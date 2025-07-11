import argparse
import os
import torch as th
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy, FSDPStrategy
from pytorch_lightning import seed_everything
import numpy as np
import scanpy as sc
import pandas as pd

from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_controlled_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.lightning_module import DiffusionLitModule
from guided_diffusion.lightning_data import CellDataModule
from torch.utils.data import DataLoader

th.set_float32_matmul_precision('medium')

class _DummyModule(pl.LightningDataModule):
    def __init__(self, total, batch_size, gene_dim):
        self.total = total
        self.batch_size = batch_size
        self.gene_dim = gene_dim

    def predict_dataloader(self):
        return DataLoader(
            list(range(self.total)),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda b: len(b),
        )


def _save_to_h5ad(decoded, save_dir, gene_names=None):
    """Store decoded samples in an ``h5ad`` file."""
    decoded = decoded.cpu().numpy()
    if gene_names is None:
        gene_names = [f"gene{i}" for i in range(decoded.shape[1])]
    adata = sc.AnnData(X=decoded, var=pd.DataFrame(index=gene_names))
    os.makedirs(save_dir, exist_ok=True)
    adata.write_h5ad(os.path.join(save_dir, "samples.h5ad"))
    return adata

def main():
    args = create_argparser().parse_args()
    seed_everything(args.seed)

    if args.data_jsonl:
        model, diffusion = create_controlled_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
    else:
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )

    num_samples_per_cond = args.num_samples if args.data_jsonl else 1

    lit_model = DiffusionLitModule(
        model=model,
        diffusion=diffusion,
        schedule_sampler=None,
        hidden_dim=128,
        prompt_arg=args.prompts if args.prompts else None,
        smiles_arg=args.smiles if args.smiles else None,
        num_samples_per_cond=num_samples_per_cond,
    )

    ckpt = th.load(args.ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        lit_model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        lit_model.load_state_dict(ckpt, strict=False)
    lit_model.eval()

    if args.data_jsonl:
        data_module = CellDataModule(
            spec_path=args.data_jsonl,
            batch_size=args.batch_size,
            file_size=args.file_size,
            use_controlnet=True,
            keep_nz_ratio=args.keep_nz_ratio,
            keep_z_ratio=args.keep_z_ratio,
            use_prompts=(args.prompts == ""),
            use_smiles=(args.smiles == ""),
        )
        data_module.setup()
        first_file = data_module.dataset.files[0]
        ad = sc.read_h5ad(first_file, backed="r")
        gene_names = list(ad.var_names)
        ad.file.close()
        gene_dim = data_module.gene_dim
    else:
        if args.gene_name:
            ad = sc.read_h5ad(args.gene_name, backed="r")
            gene_names = list(ad.var_names)
            gene_dim = len(gene_names)
            ad.file.close()
        else:
            gene_names = None
            gene_dim = args.gene_dim
        data_module = _DummyModule(args.num_samples, args.batch_size, gene_dim)

    num_gpus = th.cuda.device_count()
    if num_gpus > 1:
        accelerator = "gpu"
        devices = num_gpus
        strategy = FSDPStrategy() if args.strategy == "fsdp" else DDPStrategy()
    elif num_gpus == 1:
        accelerator = "gpu"
        devices = num_gpus
        strategy = "auto"
    else:
        accelerator = "cpu"
        devices = None
        strategy = "auto"

    trainer = pl.Trainer(accelerator=accelerator, strategy=strategy, devices=devices)

    outputs = trainer.predict(lit_model, datamodule=data_module)
    latents_list, samples_list = zip(*outputs)

    samples = th.cat(samples_list, dim=0)

    if gene_names is not None and gene_dim == samples.shape[1]:
        valid_names = gene_names
    else:
        valid_names = None

    _save_to_h5ad(samples, args.save_dir, valid_names)

    if latents_list:
        latents = th.cat(latents_list, dim=0)
        os.makedirs(args.save_dir, exist_ok=True)
        np.savez(os.path.join(args.save_dir, "latents.npz"), cell_gen=latents.numpy())


def create_argparser():
    defaults = dict(
        num_samples=1000,
        batch_size=1000,
        ckpt_path="/path/to/ckpt.ckpt",
        gene_name="",
        gene_dim=0,
        data_jsonl="",
        save_dir="output/predict",
        prompts="",
        smiles="",
        file_size=1,
        keep_nz_ratio=1.0,
        keep_z_ratio=1.0,
        strategy="ddp",
        seed=42,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
