"""
Train a diffusion model on cells using PyTorch Lightning.
"""
import argparse
import os

import pytorch_lightning as pl
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
from pytorch_lightning.utilities.seed import seed_everything

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


def main():
    seed_everything(1234)
    args = create_argparser().parse_args()

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
        vae_path=args.vae_path,
        train_vae=False,
        hidden_dim=128,
    )

    data_module = CellDataModule(
        spec_path=args.data_jsonl,
        batch_size=args.batch_size,
        file_size=args.file_size,
        use_controlnet=args.use_controlnet,
        keep_ratio=args.keep_ratio,
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
    progress_bar = RichProgressBar()
    model_summary = RichModelSummary()
    ema_callback = EMACallback(ema_rate=float(args.ema_rate))
    csv_logger = CSVLogger(args.save_dir, name=args.model_name)
    tb_logger = TensorBoardLogger(save_dir=args.save_dir, name=f"{args.model_name}_tb")
    wandb_logger = WandbLogger(name=args.model_name, save_dir=args.save_dir, project=args.model_name)

    trainer = pl.Trainer(
        max_steps=args.lr_anneal_steps,
        log_every_n_steps=args.log_interval,
        logger=[csv_logger, tb_logger, wandb_logger],
        callbacks=[
            period_checkpoint,
            best_checkpoint,
            lr_monitor,
            progress_bar,
            model_summary,
            ema_callback,
        ],
        accelerator="auto",
        devices=1,
        reload_dataloaders_every_n_epochs=1,
    )
    trainer.fit(lit_model, datamodule=data_module)


def create_argparser():
    defaults = dict(
        data_jsonl="/path/to/spec.jsonl",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0001,
        lr_anneal_steps=500000,
        batch_size=128,
        file_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=200000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        vae_path="output/Autoencoder_checkpoint/muris_AE/model_seed=0_step=0.pt",
        use_controlnet=False,
        keep_ratio=0.5,
        model_name="muris_diffusion",
        save_dir="output/diffusion_checkpoint",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
