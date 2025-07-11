import argparse
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
    RichModelSummary,
)
from functools import partial
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy, FSDPStrategy
from torch.optim.lr_scheduler import (
    LinearLR,
    SequentialLR,
    CosineAnnealingLR,
)

try:
    from swanlab.integration.pytorch_lightning import SwanLabLogger
except Exception:  # pragma: no cover - optional dependency
    SwanLabLogger = None

from VAE_model import VAE
from guided_diffusion.console_logger import ConsoleLogger
from guided_diffusion.lightning_data import CellDataModule


torch.set_float32_matmul_precision("medium")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Finetune Scimilarity")
    parser.add_argument(
        "--data_jsonl",
        type=str,
        default="/data1/lep/Workspace/guided-diffusion/data/tabula_muris/all.h5ad",
    )
    parser.add_argument("--loss_ae", type=str, default="mse")
    parser.add_argument("--decoder_activation", type=str, default="ReLU")
    parser.add_argument("--resume_checkpoint", action="store_true")
    parser.add_argument("--split_seed", type=int, default=1234)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strategy", type=str, default="ddp")
    parser.add_argument("--hparams", type=str, default="")
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--max_minutes", type=int, default=3000)
    parser.add_argument("--checkpoint_freq", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--file_size", type=int, default=100)
    parser.add_argument("--state_dict", type=str, default=None)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="../output/ae_checkpoint/muris_AE",
    )
    parser.add_argument("--sweep_seeds", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--use_swanlab", action="store_true", help="Enable swanlab logger")
    parser.add_argument("--swanlab_project", type=str, default="VAE-Lightning-Project")
    parser.add_argument("--swanlab_name", type=str, default=None)
    parser.add_argument("--swanlab_id", type=str, default="")
    parser.add_argument("--no_progress", action="store_false", dest="progress")
    parser.add_argument("--enable_schd", action="store_true", help="Enable CosineAnnealingLR scheduler")
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--T_max", type=int, default=0, help="CosineAnnealingLR: max number of steps")
    parser.add_argument("--eta_min", type=float, default=0.0, help="CosineAnnealingLR: min lr")
    return vars(parser.parse_args())


class LightningVAE(pl.LightningModule):
    def __init__(
        self,
        loss_ae,
        hidden_dim,
        decoder_activation,
        warmup_steps=0,
        lr=1e-3,
        enable_schd=False,
        T_max=1000,
        eta_min=0.0,
    ):
        super().__init__()
        self.VAE = partial(
            VAE,
            loss_ae=loss_ae,
            hidden_dim=hidden_dim,
            decoder_activation=decoder_activation,
        )
        self.save_hyperparameters(ignore=["vae"])
        self.lr = lr
        self.vae = None

    def configure_model(self):
        if self.vae is None:
            ds = self.trainer.datamodule
            self.vae = self.VAE(num_genes=ds.gene_dim)

    def training_step(self, batch, batch_idx):
        self.configure_model()
        genes, _ = batch
        gene_reconstructions = self.vae(genes)
        loss = self.vae.loss_autoencoder(gene_reconstructions, genes)
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("lr", lr, on_step=True, prog_bar=True, logger=False)
        return loss

    def configure_optimizers(self):
        self.configure_model()
        params = list(self.vae.encoder.parameters()) + list(self.vae.decoder.parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=self.vae.hparams["autoencoder_lr"],
            weight_decay=self.vae.hparams["autoencoder_wd"],
        )

        if getattr(self.hparams, "enable_schd", False):
            schedulers = []
            milestones = []
            if self.hparams.warmup_steps > 0:
                start_factor = 1.0 / float(self.hparams.warmup_steps)
                warmup = LinearLR(
                    optimizer,
                    start_factor=start_factor,
                    end_factor=1.0,
                    total_iters=self.hparams.warmup_steps,
                )
                schedulers.append(warmup)
                milestones.append(self.hparams.warmup_steps)

            if self.hparams.T_max > self.hparams.warmup_steps:
                schedulers.append(
                    CosineAnnealingLR(
                        optimizer,
                        T_max=self.hparams.T_max - self.hparams.warmup_steps,
                        eta_min=self.hparams.eta_min,
                    )
                )

            if len(schedulers) == 1:
                sched = schedulers[0]
            else:
                sched = SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": sched,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        return optimizer


def train_vae_lightning(args):

    seed_everything(args["seed"])

    if args["T_max"] <= 0:
        args["T_max"] = args["max_steps"]

    data_module = CellDataModule(
        spec_path=args["data_jsonl"],
        batch_size=args["batch_size"],
        file_size=args["file_size"],
        use_controlnet=False,
        keep_nz_ratio=1.0,
        keep_z_ratio=1.0,
        use_prompts=False,
        use_smiles=False,
    )

    model = LightningVAE(
        loss_ae=args["loss_ae"],
        hidden_dim=128,
        decoder_activation=args["decoder_activation"],
        warmup_steps=args.get("warmup_steps", 0),
        lr=args.get("lr", 1e-3),
        enable_schd=args.get("enable_schd", False),
        T_max=args.get("T_max", 1000),
        eta_min=args.get("eta_min", 0.0),
    )

    os.makedirs(args["save_dir"], exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=args["save_dir"],
            every_n_train_steps=args["checkpoint_freq"],
            save_top_k=-1,
            save_on_train_epoch_end=True,
            filename="period-{step}",
        ),
        ModelCheckpoint(
            dirpath=args["save_dir"],
            monitor="train_loss",
            save_top_k=1,
            mode="min",
            filename="best",
        ),
        LearningRateMonitor(logging_interval="step"),
        RichModelSummary(max_depth=5),
    ]
    theme = RichProgressBarTheme()
    theme.metrics_format = ".3e"
    if args["progress"]:
        callbacks.append(RichProgressBar(theme=theme))

    log_dir = args["log_dir"]
    if log_dir is None:
        log_dir = os.path.join(args["save_dir"], "log")
        os.makedirs(log_dir, exist_ok=True)
    loggers = []
    tb_logger = TensorBoardLogger(save_dir=log_dir, name=args["swanlab_name"])
    loggers.append(tb_logger)
    if args.get("use_swanlab", False) and SwanLabLogger is not None:
        wandb_logger = SwanLabLogger(
            project=args["swanlab_project"],
            experiment_name=args["swanlab_name"],
            save_dir=log_dir,
            config=args,
            id=args.get("swanlab_id") if args["resume_checkpoint"] else None,
            resume="allow" if args["resume_checkpoint"] else None,
        )
        loggers.append(wandb_logger)
    if not args["progress"]:
        concole_logger = ConsoleLogger(precision=4)
        loggers.append(concole_logger)

    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    num_nodes = world_size // max(local_world_size, 1)
    num_gpus = torch.cuda.device_count()

    if num_gpus > 1:
        accelerator = "gpu"
        strategy = FSDPStrategy() if args["strategy"] == "fsdp" else DDPStrategy()
    elif num_gpus == 1:
        accelerator = "gpu"
        strategy = "auto"
    else:
        accelerator = "cpu"
        strategy = "auto"

    trainer = pl.Trainer(
        max_epochs=-1,
        max_steps=args["max_steps"],
        max_time={"minutes": args["max_minutes"]},
        callbacks=callbacks,
        default_root_dir=args["save_dir"],
        enable_checkpointing=True,
        log_every_n_steps=50,
        accelerator=accelerator,
        devices=local_world_size,
        num_nodes=num_nodes,
        logger=loggers,
        strategy=strategy,
    )

    ckpt_path = None
    if args["resume_checkpoint"]:
        ckpt_path = os.path.join(args["save_dir"], "last.ckpt")

    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
    return model


if __name__ == "__main__":
    args = parse_arguments()
    train_vae_lightning(args)

