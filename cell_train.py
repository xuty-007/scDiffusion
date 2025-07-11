"""
Train a diffusion model on cells using PyTorch Lightning.
"""
import argparse
import os
import torch as th

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
    RichModelSummary,
)
from pytorch_lightning.loggers import (
    CSVLogger,
    TensorBoardLogger,
    WandbLogger,
)
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy, FSDPStrategy
from pytorch_lightning import seed_everything

from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_controlled_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.lightning_module import DiffusionLitModule
from guided_diffusion.lightning_data import CellDataModule
from guided_diffusion.ema_callback import EMACallback

from console_logger import ConsoleLogger

def main():

    args = create_argparser().parse_args()
    seed_everything(args.seed)

    if args.use_controlnet:
        model, diffusion = create_controlled_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
    else:
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    lit_model = DiffusionLitModule(
        model=model,
        diffusion=diffusion,
        lr=args.lr,
        weight_decay=args.weight_decay,
        schedule_sampler=schedule_sampler,
        lr_anneal_steps=args.lr_anneal_steps,
        warmup_steps=args.warmup_steps,
        lr_scheduler=args.lr_scheduler,
        vae_path=args.vae_path,
        model_path=args.model_path,
        hidden_dim=128,
        resume_checkpoint=args.resume_checkpoint,
    )

    data_module = CellDataModule(
        spec_path=args.data_jsonl,
        batch_size=args.batch_size,
        file_size=args.file_size,
        use_controlnet=args.use_controlnet,
        keep_nz_ratio=args.keep_nz_ratio,
        keep_z_ratio=args.keep_z_ratio,
        use_prompts=args.use_prompts,
        use_smiles=args.use_smiles,
    )

    period_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(args.save_dir, args.model_name),
        every_n_train_steps=args.save_interval,
        save_last=True,
    )
    best_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(args.save_dir, args.model_name),
        monitor="loss",
        mode="min",
        filename="best-{step}"
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = RichModelSummary()
    progress_bar = RichProgressBar() if not args.no_progress else None
    ema_callback = EMACallback(ema_rate=float(args.ema_rate))
    csv_logger = CSVLogger(args.save_dir, name=args.model_name)
    tb_logger = TensorBoardLogger(save_dir=args.save_dir, name=f"{args.model_name}_tb")
    wandb_logger = WandbLogger(name=args.model_name, save_dir=args.save_dir, project=args.model_name)
    console_logger = ConsoleLogger() if args.no_progress else None

    num_gpus = th.cuda.device_count()
    local_rank = os.environ.get("LOCAL_RANK")
    if num_gpus > 1:
        accelerator = "gpu"
        if local_rank is None:
            devices = num_gpus
        else:
            # running under torchrun -> each process handles one GPU
            devices = 1
        if args.strategy == "fsdp":
            strategy = FSDPStrategy()
        else:
            strategy = DDPStrategy()
    else:
        accelerator = "gpu" if num_gpus == 1 else "cpu"
        devices = 1 if num_gpus == 1 else None
        strategy = "auto"

    callbacks = [period_checkpoint, best_checkpoint, lr_monitor, model_summary, ema_callback]
    if progress_bar is not None:
        callbacks.insert(3, progress_bar)

    loggers = [csv_logger, tb_logger, wandb_logger]
    if console_logger is not None:
        loggers.append(console_logger)

    trainer_kwargs = dict(
        max_steps=args.lr_anneal_steps,
        log_every_n_steps=args.log_interval,
        logger=loggers,
        callbacks=callbacks,
        accelerator=accelerator,
        strategy=strategy,
        reload_dataloaders_every_n_epochs=1,
        enable_progress_bar=progress_bar is not None,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision=args.precision,
    )
    if devices is not None:
        trainer_kwargs["devices"] = devices

    trainer = pl.Trainer(**trainer_kwargs)

    ckpt_path = None
    if args.resume_checkpoint:
        ckpt_path = os.path.join(args.save_dir, args.model_name, "last.ckpt")
    else:
        checkpoint = {}
        if args.ckpt_path:
            checkpoint = th.load(args.ckpt_path, map_location="cpu")
        checkpoint.setdefault("state_dict", {})
        lit_model.on_load_checkpoint(checkpoint)
        if checkpoint["state_dict"]:
            lit_model.load_state_dict(checkpoint["state_dict"], strict=False)
        if args.ckpt_path:
            ema_callback.on_load_checkpoint(None, lit_model, checkpoint)

    trainer.fit(lit_model, datamodule=data_module, ckpt_path=ckpt_path)


def create_argparser():
    defaults = dict(
        data_jsonl="/path/to/spec.jsonl",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0001,
        lr_anneal_steps=500000,
        warmup_steps=0,
        lr_scheduler="lambda",
        batch_size=128,
        file_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=200000,
        ckpt_path="",
        model_path="",
        gradient_clip_val=0.0,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=1,
        precision="32",
        vae_path="output/Autoencoder_checkpoint/muris_AE/model_seed=0_step=0.pt",
        use_controlnet=False,
        keep_nz_ratio=0.5,
        keep_z_ratio=0.0,
        use_prompts=True,
        use_smiles=True,
        seed=1234,
        no_progress=False,
        strategy="ddp",
        model_name="muris_diffusion",
        save_dir="output/diffusion_checkpoint",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--resume_checkpoint", action="store_true", help="resume from last checkpoint in save_dir")
    return parser


if __name__ == "__main__":
    main()
