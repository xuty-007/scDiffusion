import pytorch_lightning as pl
from torch.utils.data import DataLoader
import json
import os
from torch.utils.data.dataloader import default_collate

from .cell_datasets_loader import StreamingCellDataset

import warnings
from anndata import ImplicitModificationWarning

# 忽略所有 ImplicitModificationWarning
warnings.filterwarnings('ignore', category=ImplicitModificationWarning)


class CellDataModule(pl.LightningDataModule):
    """Lightning DataModule for cell datasets.

    Parameters
    ----------
    spec_path : str
        Path to a JSONL file describing datasets or a directory containing
        multiple ``.jsonl`` files. Each line in a JSONL file should be a JSON
        object with ``data_dir`` pointing to the dataset location, ``data_type``
        of either ``h5ad`` or ``mtx``, ``prompt_template`` for formatting
        prompts (or ``None`` to skip PubMedBERT), and ``smiles`` containing the
        column name with SMILES strings (or ``None`` to skip ChemBERT).
    batch_size : int
        Number of samples per batch.
    use_controlnet : bool, optional
        Whether to prepare ControlNet inputs.
    keep_ratio : float, optional
        Ratio of expression values kept for ControlNet inputs.
    """

    def __init__(self, spec_path, batch_size, file_size=1, num_workers=12, use_controlnet=False, keep_ratio=0.5, path_replace=None):
        super().__init__()
        self.spec_path = spec_path
        self.batch_size = batch_size
        self.file_size = file_size
        self.num_workers = num_workers
        self.use_controlnet = use_controlnet
        self.keep_ratio = keep_ratio
        if path_replace is None:
            self.path_replace = ["", ""]
        else:
            self.path_replace = path_replace

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
            data_path = spec["data_path"].replace(*self.path_replace)
            if not os.path.exists(data_path):
                continue
            files.append(data_path)
            length = spec["n_obs"]
            lengths.append(length)
            prompt_templates.append(spec.get("prompt_template"))
            smiles_cols.append(spec.get("smiles"))
            if self.gene_dim is None:
                self.gene_dim = spec.get("gene_dim")

        self.files_all = files
        self.lengths_all = lengths
        self.prompts_all = prompt_templates
        self.smiles_all = smiles_cols

        self.dataset = StreamingCellDataset(
            self.files_all,
            self.lengths_all,
            prompt_templates=self.prompts_all,
            smiles_cols=self.smiles_all,
            file_size=self.file_size,
            use_controlnet=self.use_controlnet,
            keep_ratio=self.keep_ratio,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            prefetch_factor=2,
            collate_fn=self._collate_with_optional,
        )

    @staticmethod
    def _collate_with_optional(batch):
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
