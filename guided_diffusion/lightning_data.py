import pytorch_lightning as pl
from torch.utils.data import DataLoader
import json
import os
import yaml
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
        of either ``h5ad`` or ``mtx``, ``prompt_template_file`` pointing to a
        YAML file containing a list of prompt templates (or ``None`` to skip
        PubMedBERT), and ``smiles`` containing the column name with SMILES
        strings (or ``None`` to skip ChemBERT).
    batch_size : int
        Number of samples per batch.
    use_controlnet : bool, optional
        Whether to prepare ControlNet inputs.
    keep_nz_ratio : float, optional
        Ratio of non-zero expression values kept for ControlNet inputs.
    keep_z_ratio : float, optional
        Ratio of zero expression values kept for ControlNet inputs.
    use_prompts : bool, optional
        Whether to return prompt strings with each sample.
    use_smiles : bool, optional
        Whether to return SMILES strings with each sample.
    """

    def __init__(
        self,
        spec_path,
        batch_size,
        file_size=1,
        num_workers=12,
        use_controlnet=False,
        keep_nz_ratio=0.5,
        keep_z_ratio=0.0,
        use_prompts=True,
        use_smiles=True,
        path_replace=None,
    ):
        super().__init__()
        self.spec_path = spec_path
        self.batch_size = batch_size
        self.file_size = file_size
        self.num_workers = num_workers
        self.use_controlnet = use_controlnet
        self.keep_nz_ratio = keep_nz_ratio
        self.keep_z_ratio = keep_z_ratio
        self.use_prompts = use_prompts
        self.use_smiles = use_smiles
        if path_replace is None:
            self.path_replace = ["", ""]
        else:
            self.path_replace = path_replace

    def setup(self, stage=None):
        specs = []
        paths = []
        if isinstance(self.spec_path, str):
            spec_paths = [sp.strip() for sp in self.spec_path.split(',') if sp.strip()]
        else:
            spec_paths = list(self.spec_path)

        for path_idx, sp in enumerate(spec_paths):
            if os.path.isdir(sp):
                for fname in sorted(os.listdir(sp)):
                    if fname.endswith(".jsonl"):
                        paths.append((os.path.join(sp, fname), path_idx))
            else:
                paths.append((sp, path_idx))

        for p, ds_idx in paths:
            with open(p, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        spec = json.loads(line)
                        specs.append((spec, ds_idx))

        files = []
        lengths = []
        prompt_templates = []
        smiles_cols = []
        self.gene_dim = None

        ds_ids = []
        for spec, ds_idx in specs:
            data_path = spec["data_path"].replace(*self.path_replace)
            if not os.path.exists(data_path):
                continue
            files.append(data_path)
            length = spec["n_obs"]
            lengths.append(length)
            ds_ids.append(ds_idx)
            if self.use_prompts:
                tmpl_file = spec.get("prompt_template_file")
                if tmpl_file is not None:
                    tmpl_file = tmpl_file.replace(*self.path_replace)
                    with open(tmpl_file, "r") as f:
                        templates = yaml.safe_load(f)
                    if not isinstance(templates, list):
                        raise ValueError(
                            f"YAML template file {tmpl_file} must contain a list"
                        )
                else:
                    templates = spec.get("prompt_template")
            else:
                templates = None
            prompt_templates.append(templates)

            if self.use_smiles:
                smiles_cols.append(spec.get("smiles"))
            else:
                smiles_cols.append(None)

            if self.gene_dim is None:
                self.gene_dim = spec.get("gene_dim")

        self.files_all = files
        self.lengths_all = lengths
        self.prompts_all = prompt_templates
        self.smiles_all = smiles_cols
        self.ds_ids_all = ds_ids
        from collections import Counter
        self.ds_counts = {k: v for k, v in Counter(ds_ids).items()}

        self.dataset = StreamingCellDataset(
            self.files_all,
            self.lengths_all,
            prompt_templates=self.prompts_all,
            smiles_cols=self.smiles_all,
            ds_ids=self.ds_ids_all,
            ds_counts=self.ds_counts,
            file_size=self.file_size,
            use_controlnet=self.use_controlnet,
            keep_nz_ratio=self.keep_nz_ratio,
            keep_z_ratio=self.keep_z_ratio,
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

    def predict_dataloader(self):
        """Return DataLoader for prediction."""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            prefetch_factor=2,
            collate_fn=self._collate_with_optional,
        )

    @staticmethod
    def _collate_with_optional(batch):
        arrays, conds, ds_ids = zip(*batch)
        arrays = default_collate(arrays)
        cond_batch = {}
        keys = conds[0].keys()
        for key in keys:
            vals = [d[key] for d in conds]
            if any(v is None for v in vals):
                cond_batch[key] = list(vals)
            else:
                cond_batch[key] = default_collate(vals)
        ds_ids = default_collate(ds_ids)
        return arrays, cond_batch, ds_ids
