import math
import random

from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset, IterableDataset
import pandas as pd
import glob
import os
from functools import lru_cache

import scanpy as sc
import torch
import sys
sys.path.append('..')
from VAE.VAE_model import VAE
from sklearn.preprocessing import LabelEncoder


PROMPT_TEMPLATE = (
    "In this cell-based experiment, {cell_name} cells (cell line {cell_line_id}) "
    "were treated with {drug} at {drug_concentration_with_unit} "
    "[PubChem CID {pubchem_cid}; SMILES {canonical_smiles}]. "
    "The broad mechanism of action was {moa_broad} and the fine mechanism was {moa_fine}; "
    "reported targets: {targets}. "
    "Following treatment, cells showed gene count {gene_count}, "
    "transcript count {tscp_count}, mitochondrial read count {mread_count}, "
    "mitochondrial fraction {pcnt_mito:.4f}, S-phase score {S_score:.4f}, "
    "G2M-phase score {G2M_score:.4f}, and were assigned to the {phase} phase. "
    "The drug {approved_str} and {trials_str}. "
    "Additional notes: {gpt_notes_approval}"
)

@lru_cache(maxsize=8)
def read_h5ad_backed(path: str):
    """Load an AnnData object in backed read-only mode with caching."""
    return sc.read_h5ad(path, backed="r")

def stabilize(expression_matrix):
    ''' Use Anscombes approximation to variance stabilize Negative Binomial data
    See https://f1000research.com/posters/4-1041 for motivation.
    Assumes columns are samples, and rows are genes
    '''
    from scipy import optimize
    phi_hat, _ = optimize.curve_fit(lambda mu, phi: mu + phi * mu ** 2, expression_matrix.mean(1), expression_matrix.var(1))

    return np.log(expression_matrix + 1. / (2 * phi_hat[0]))

def load_VAE(vae_path, num_gene, hidden_dim):
    autoencoder = VAE(
        num_genes=num_gene,
        device='cuda',
        seed=0,
        loss_ae='mse',
        hidden_dim=hidden_dim,
        decoder_activation='ReLU',
    )
    autoencoder.load_state_dict(torch.load(vae_path))
    return autoencoder


def load_data(
    *,
    data_dir,
    batch_size,
    vae_path=None,
    deterministic=False,
    train_vae=False,
    hidden_dim=128,
):
    """
    For a dataset, create a generator over (cells, kwargs) pairs.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param vae_path: the path to save autoencoder / read autoencoder checkpoint.
    :param deterministic: if True, yield results in a deterministic order.
    :param train_vae: train the autoencoder or use the autoencoder.
    :param hidden_dim: the dimensions of latent space. If use pretrained weight, set 128
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    if os.path.isdir(data_dir):
        h5ad_files = glob.glob(os.path.join(data_dir, "*.h5ad"))
        if not h5ad_files:
            raise ValueError(f"no h5ad files found in {data_dir}")
        adatas = [sc.read_h5ad(f) for f in h5ad_files]
        adata = sc.concat(
            adatas,
            axis=0,
            join="outer",
            label="batch",
            keys=[os.path.basename(f)[:-5] for f in h5ad_files],
            fill_value=0,
            index_unique="-",
        )
    else:
        adata = sc.read_h5ad(data_dir)
    
    # preporcess the data. modify this part if use your own dataset. the gene expression must first norm1e4 then log1p
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_genes=10)
    adata.var_names_make_unique()

    # if generate ood data, left this as the ood data
    # selected_cells = (adata.obs['organ'] != 'mammary') | (adata.obs['celltype'] != 'B cell')  
    # adata = adata[selected_cells, :]  

    classes = adata.obs['celltype'].values
    label_encoder = LabelEncoder()
    labels = classes
    label_encoder.fit(labels)
    classes = label_encoder.transform(labels)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    cell_data = adata.X.toarray()

    # Prepare prompts and SMILES strings for later conditioning
    prompts = []
    smiles = []
    for _, row in adata.obs.iterrows():
        canonical = row.get("canonical_smiles")
        if canonical is not None and not pd.isna(canonical):
            smiles.append(canonical)
            canonical_prompt = canonical
        else:
            smiles.append(None)
            canonical_prompt = "None"
        prompt = PROMPT_TEMPLATE.format(
            cell_name=row["cell_name"],
            cell_line_id=row["cell_line_id"],
            drug=row["drug"],
            drug_concentration_with_unit=row["drug_concentration_with_unit"].replace(" ", ""),
            pubchem_cid=row["pubchem_cid"],
            canonical_smiles=canonical_prompt,
            moa_broad=row["moa-broad"],
            moa_fine=row["moa-fine"],
            targets=row.get("targets", "unknown"),
            gene_count=row["gene_count"],
            tscp_count=row["tscp_count"],
            mread_count=row["mread_count"],
            pcnt_mito=row["pcnt_mito"],
            S_score=row["S_score"],
            G2M_score=row["G2M_score"],
            phase=row["phase"],
            approved_str="is human-approved" if row["human-approved"] == "yes" else "is not human-approved",
            trials_str="is in clinical trials" if row["clinical-trials"] == "yes" else "is not in clinical trials",
            gpt_notes_approval=row["gpt-notes-approval"],
        )
        prompts.append(prompt)

    # turn the gene expression into latent space. use this if training the diffusion backbone.
    if not train_vae:
        num_gene = cell_data.shape[1]
        autoencoder = load_VAE(vae_path, num_gene, hidden_dim)
        cell_data = autoencoder(torch.tensor(cell_data).cuda(), return_latent=True)
        cell_data = cell_data.cpu().detach().numpy()
    
    dataset = CellDataset(
        cell_data,
        classes,
        prompts,
        smiles,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


class CellDataset(Dataset):
    def __init__(
        self,
        files,
        lengths,
        prompt_templates=None,
        smiles_cols=None,
        use_controlnet=False,
        keep_ratio=0.5,
    ):
        """Dataset that lazily reads rows from one or more H5AD files.

        Parameters
        ----------
        files : list[str]
            Paths to backed ``.h5ad`` files.
        lengths : list[int]
            Number of cells in each file.
        prompt_templates : list[str] or None
            Prompt format strings corresponding to ``files``. ``None`` disables
            prompt generation for that file.
        smiles_cols : list[str] or None
            Column names for SMILES strings. ``None`` disables ChemBERT
            conditioning for that file.
        """

        super().__init__()
        self.files = files
        self.lengths = lengths
        self.cumsum = np.cumsum(lengths)
        self.total_len = int(self.cumsum[-1])
        if prompt_templates is None:
            prompt_templates = [None] * len(files)
        if smiles_cols is None:
            smiles_cols = [None] * len(files)
        assert len(prompt_templates) == len(files)
        assert len(smiles_cols) == len(files)
        self.prompt_templates = list(prompt_templates)
        self.smiles_cols = list(smiles_cols)
        self.use_controlnet = use_controlnet
        self.keep_ratio = keep_ratio

    def __len__(self):
        return self.total_len

    def _locate(self, idx):
        file_idx = int(np.searchsorted(self.cumsum, idx, side="right"))
        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - int(self.cumsum[file_idx - 1])
        return file_idx, self.files[file_idx], local_idx

    def __getitem__(self, idx):
        file_idx, path, local_idx = self._locate(idx)
        adata = read_h5ad_backed(path)
        row = adata[local_idx]
        arr = row.X
        if hasattr(arr, "toarray"):
            arr = arr.toarray().ravel().astype(np.float32)
        else:
            arr = np.array(arr, dtype=np.float32)
        out_dict = {}

        # smiles string
        canonical = None
        col = self.smiles_cols[file_idx]
        if col and col in row.obs.columns:
            canonical = row.obs[col].values[0]
            if pd.isna(canonical):
                canonical = None
        out_dict["smiles"] = canonical

        # prompt construction
        template = self.prompt_templates[file_idx]
        if template is not None:
            row_dict = row.obs.iloc[0].to_dict()
            canonical_prompt = canonical if canonical is not None else "None"
            row_dict.setdefault("canonical_smiles", canonical_prompt)
            if "drug_concentration_with_unit" in row_dict:
                row_dict["drug_concentration_with_unit"] = str(row_dict["drug_concentration_with_unit"]).replace(" ", "")
            prompt = template.format(**row_dict)
        else:
            prompt = None
        out_dict["prompt"] = prompt

        if self.use_controlnet:
            control = np.zeros_like(arr, dtype=np.float32)
            nz = np.where(arr != 0)[0]
            if len(nz) > 0:
                mask = np.random.rand(len(nz)) < self.keep_ratio
                keep_idx = nz[mask]
                control[keep_idx] = arr[keep_idx]
            out_dict["control"] = control

        return arr, out_dict


class MemoryCellDataset(Dataset):
    """Dataset that fully loads selected H5AD files into memory."""

    def __init__(
        self,
        files,
        prompt_templates=None,
        smiles_cols=None,
        use_controlnet=False,
        keep_ratio=0.5,
    ):
        super().__init__()
        if prompt_templates is None:
            prompt_templates = [None] * len(files)
        if smiles_cols is None:
            smiles_cols = [None] * len(files)
        assert len(prompt_templates) == len(files)
        assert len(smiles_cols) == len(files)

        self.files = list(files)
        arrays = []
        prompts = []
        smiles = []
        for path, tmpl, col in zip(files, prompt_templates, smiles_cols):
            adata = sc.read_h5ad(path)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            arr = adata.X
            if hasattr(arr, "toarray"):
                arr = arr.toarray().astype(np.float32)
            else:
                arr = np.array(arr, dtype=np.float32)
            arrays.append(arr)
            for _, row in adata.obs.iterrows():
                canonical = None
                if col and col in row.index:
                    canonical = row[col]
                    if pd.isna(canonical):
                        canonical = None
                smiles.append(canonical)
                if tmpl is not None:
                    row_dict = row.to_dict()
                    cp = canonical if canonical is not None else "None"
                    row_dict.setdefault("canonical_smiles", cp)
                    if "drug_concentration_with_unit" in row_dict:
                        row_dict["drug_concentration_with_unit"] = str(
                            row_dict["drug_concentration_with_unit"]
                        ).replace(" ", "")
                    prompt = tmpl.format(**row_dict)
                else:
                    prompt = None
                prompts.append(prompt)

        self.arrays = np.concatenate(arrays, axis=0) if len(arrays) > 1 else arrays[0]
        self.prompts = prompts
        self.smiles = smiles
        self.use_controlnet = use_controlnet
        self.keep_ratio = keep_ratio

    def __len__(self):
        return self.arrays.shape[0]

    def __getitem__(self, idx):
        arr = self.arrays[idx].astype(np.float32)
        out = {
            "prompt": self.prompts[idx],
            "smiles": self.smiles[idx],
        }
        if self.use_controlnet:
            control = np.zeros_like(arr, dtype=np.float32)
            nz = np.where(arr != 0)[0]
            if len(nz) > 0:
                mask = np.random.rand(len(nz)) < self.keep_ratio
                keep_idx = nz[mask]
                control[keep_idx] = arr[keep_idx]
            out["control"] = control
        return arr, out


class StreamingCellDataset(IterableDataset):
    """Stream ``file_size`` H5AD files at a time until all are exhausted."""

    def __init__(
        self,
        files,
        lengths,
        prompt_templates=None,
        smiles_cols=None,
        file_size=1,
        use_controlnet=False,
        keep_ratio=0.5,
    ):
        super().__init__()
        if prompt_templates is None:
            prompt_templates = [None] * len(files)
        if smiles_cols is None:
            smiles_cols = [None] * len(files)
        assert len(prompt_templates) == len(files)
        assert len(smiles_cols) == len(files)

        self.files = list(files)
        self.lengths = list(lengths)
        self.prompt_templates = list(prompt_templates)
        self.smiles_cols = list(smiles_cols)
        self.file_size = file_size
        self.use_controlnet = use_controlnet
        self.keep_ratio = keep_ratio

    def __len__(self):
        return sum(self.lengths)

    def _iter_file(self, file_idx):
        path = self.files[file_idx]
        length = self.lengths[file_idx]
        tmpl = self.prompt_templates[file_idx]
        col = self.smiles_cols[file_idx]
        adata = read_h5ad_backed(path)
        order = np.random.permutation(length)
        for local_idx in order:
            row = adata[int(local_idx)]
            arr = row.X
            if hasattr(arr, "toarray"):
                arr = arr.toarray().ravel().astype(np.float32)
            else:
                arr = np.array(arr, dtype=np.float32)

            out = {}
            canonical = None
            if col and col in row.obs.columns:
                canonical = row.obs[col].values[0]
                if pd.isna(canonical):
                    canonical = None
            out["smiles"] = canonical

            if tmpl is not None:
                row_dict = row.obs.iloc[0].to_dict()
                cp = canonical if canonical is not None else "None"
                row_dict.setdefault("canonical_smiles", cp)
                if "drug_concentration_with_unit" in row_dict:
                    row_dict["drug_concentration_with_unit"] = str(
                        row_dict["drug_concentration_with_unit"]
                    ).replace(" ", "")
                prompt = tmpl.format(**row_dict)
            else:
                prompt = None
            out["prompt"] = prompt

            if self.use_controlnet:
                control = np.zeros_like(arr, dtype=np.float32)
                nz = np.where(arr != 0)[0]
                if len(nz) > 0:
                    mask = np.random.rand(len(nz)) < self.keep_ratio
                    keep_idx = nz[mask]
                    control[keep_idx] = arr[keep_idx]
                out["control"] = control

            yield arr, out

    def __iter__(self):
        order = np.random.permutation(len(self.files))
        for start in range(0, len(order), self.file_size):
            subset = order[start : start + self.file_size]
            mem_ds = MemoryCellDataset(
                [self.files[i] for i in subset],
                prompt_templates=[self.prompt_templates[i] for i in subset],
                smiles_cols=[self.smiles_cols[i] for i in subset],
                use_controlnet=self.use_controlnet,
                keep_ratio=self.keep_ratio,
            )
            for idx in range(len(mem_ds)):
                yield mem_ds[idx]

