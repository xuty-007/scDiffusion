import json
import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


def _collate_with_optional(batch):
    """Custom collate that keeps lists for keys with ``None`` values."""
    arrays, conds = zip(*batch)
    arrays = default_collate(arrays)
    cond_batch = {}
    keys = conds[0].keys()
    for key in keys:
        vals = [d[key] for d in conds]
        if any(v is None for v in vals):
            cond_batch[key] = list(vals)
        else:
            cond_batch[key] = default_collate(vals)
    return arrays, cond_batch

from .cell_datasets_loader import CellDataset


class CellDataModule(pl.LightningDataModule):
    """Lightning DataModule for cell datasets.

    Parameters
    ----------
    spec_path : str
        Path to a JSONL file or directory of JSONL files. Each line contains a
        JSON object with ``data_path`` pointing to an H5AD file (not a
        directory), ``n_obs`` or ``num_cell`` giving the number of cells,
        ``gene_dim`` specifying the number of genes, ``prompt_template`` for
        building textual prompts, and ``smiles`` naming the column with SMILES
        strings.
    batch_size : int
        Number of samples per batch.
    use_controlnet : bool, optional
        Whether to prepare ControlNet inputs.
    keep_ratio : float, optional
        Ratio of expression values kept for ControlNet inputs.
    """

    def __init__(self, spec_path, batch_size, use_controlnet=False, keep_ratio=0.5):
        super().__init__()
        self.spec_path = spec_path
        self.batch_size = batch_size
        self.use_controlnet = use_controlnet
        self.keep_ratio = keep_ratio

    def setup(self, stage=None):
        specs = []
        paths = []
        if os.path.isdir(self.spec_path):
            for fname in sorted(os.listdir(self.spec_path)):
                if fname.endswith(".jsonl"):
                    paths.append(os.path.join(self.spec_path, fname))
        else:
            paths.append(self.spec_path)

        for p in paths:
            with open(p, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        specs.append(json.loads(line))

        files = []
        lengths = []
        prompt_templates = []
        smiles_cols = []
        self.gene_dim = None

        for spec in specs:
            files.append(spec["data_path"])
            length = spec.get("n_obs")
            if length is None:
                length = spec.get("num_cell")
            lengths.append(length)
            prompt_templates.append(spec.get("prompt_template"))
            smiles_cols.append(spec.get("smiles"))
            if self.gene_dim is None:
                self.gene_dim = spec.get("gene_dim")

        self.dataset = CellDataset(
            files,
            lengths,
            prompt_templates=prompt_templates,
            smiles_cols=smiles_cols,
            use_controlnet=self.use_controlnet,
            keep_ratio=self.keep_ratio,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
            drop_last=True,
            collate_fn=_collate_with_optional,
        )

