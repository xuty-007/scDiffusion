import random

import numpy as np
from torch.utils.data import DataLoader, Dataset, IterableDataset
import pandas as pd
import glob
import os
from functools import lru_cache
import json

import scanpy as sc
import torch
import torch.distributed as dist
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
    data_dir=None,
    data_jsonl=None,
    batch_size,
    vae_path=None,
    deterministic=False,
    hidden_dim=128,
    file_size=1,
):
    """
    For a dataset, create a generator over (cells, kwargs) pairs.

    :param data_dir: a dataset directory (legacy).
    :param data_jsonl: JSONL specification of datasets.
    :param batch_size: the batch size of each returned pair.
    :param vae_path: the path to save autoencoder / read autoencoder checkpoint.
    :param deterministic: if True, yield results in a deterministic order.
    :param hidden_dim: the dimensions of latent space. If use pretrained weight, set 128
    """

    if data_jsonl is not None:
        specs = []
        paths = []
        if os.path.isdir(data_jsonl):
            for fname in sorted(os.listdir(data_jsonl)):
                if fname.endswith(".jsonl"):
                    paths.append(os.path.join(data_jsonl, fname))
        else:
            paths = [p.strip() for p in str(data_jsonl).split(',') if p.strip()]

        for path_idx, p in enumerate(paths):
            with open(p, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        specs.append((json.loads(line), path_idx))

        files = []
        lengths = []
        ds_ids = []
        for spec, path_idx in specs:
            dp = spec["data_path"]
            if not os.path.exists(dp):
                continue
            files.append(dp)
            lengths.append(spec["n_obs"])
            ds_ids.append(path_idx)

        from collections import Counter
        ds_counts = {k: v for k, v in Counter(ds_ids).items()}

        dataset = StreamingCellDataset(
            files,
            lengths,
            prompt_templates=[None] * len(files),
            smiles_cols=[None] * len(files),
            ds_ids=ds_ids,
            ds_counts=ds_counts,
            file_size=file_size,
            use_controlnet=False,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=not deterministic,
            num_workers=1,
            drop_last=True,
        )
        while True:
            yield from loader

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

    # convert the gene expression to latent space
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
        ds_ids=None,
        use_controlnet=False,
        keep_nz_ratio=0.5,
        keep_z_ratio=0.0,
    ):
        """Dataset that lazily reads rows from one or more H5AD files.

        Parameters
        ----------
        files : list[str]
            Paths to backed ``.h5ad`` files.
        lengths : list[int]
            Number of cells in each file.
        prompt_templates : list[str | list[str]] or None
            Prompt format strings or lists of templates corresponding to
            ``files``. ``None`` disables prompt generation for that file.
        smiles_cols : list[str] or None
            Column names for SMILES strings. ``None`` disables ChemBERT
            conditioning for that file.
        ds_ids : list[int] or None
            Integer dataset identifiers for ``files``. ``None`` defaults to 0 for all.
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
        if ds_ids is None:
            ds_ids = [0] * len(files)
        assert len(ds_ids) == len(files)
        self.ds_ids = list(ds_ids)
        if ds_ids is None:
            ds_ids = [0] * len(files)
        assert len(ds_ids) == len(files)
        self.ds_ids = list(ds_ids)
        self.use_controlnet = use_controlnet
        self.keep_nz_ratio = keep_nz_ratio
        self.keep_z_ratio = keep_z_ratio

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
        tmpl_spec = self.prompt_templates[file_idx]
        if isinstance(tmpl_spec, list):
            template = random.choice(tmpl_spec) if len(tmpl_spec) > 0 else None
        else:
            template = tmpl_spec
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

        ds_id = self.ds_ids[file_idx]
        if self.use_controlnet:
            control = np.full_like(arr, -10.0, dtype=np.float32)
            nz = np.where(arr != 0)[0]
            if len(nz) > 0:
                mask = np.random.rand(len(nz)) < self.keep_nz_ratio
                keep_idx = nz[mask]
                control[keep_idx] = arr[keep_idx]
            z = np.where(arr == 0)[0]
            if len(z) > 0 and self.keep_z_ratio > 0:
                mask = np.random.rand(len(z)) < self.keep_z_ratio
                keep_idx = z[mask]
                control[keep_idx] = arr[keep_idx]
            out_dict["control"] = control

        return arr, out_dict, ds_id


class MemoryCellDataset(Dataset):
    """Dataset that fully loads selected H5AD files into memory."""

    def __init__(
        self,
        files,
        prompt_templates=None,
        smiles_cols=None,
        ds_ids=None,
        use_controlnet=False,
        keep_nz_ratio=0.5,
        keep_z_ratio=0.0,
    ):
        super().__init__()
        if prompt_templates is None:
            prompt_templates = [None] * len(files)
        if smiles_cols is None:
            smiles_cols = [None] * len(files)
        assert len(prompt_templates) == len(files)
        assert len(smiles_cols) == len(files)

        self.files = list(files)
        if ds_ids is None:
            ds_ids = [0] * len(files)
        assert len(ds_ids) == len(files)
        arrays = []
        prompts = []
        smiles = []
        ds_all = []
        for path, tmpl, col, ds in zip(files, prompt_templates, smiles_cols, ds_ids):
            adata = sc.read_h5ad(path)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            arr = adata.X
            if hasattr(arr, "toarray"):
                arr = arr.toarray().astype(np.float32)
            else:
                arr = np.array(arr, dtype=np.float32)
            arrays.append(arr)
            ds_all.append(np.full(arr.shape[0], ds, dtype=np.int64))
            for _, row in adata.obs.iterrows():
                canonical = None
                if col and col in row.index:
                    canonical = row[col]
                    if pd.isna(canonical):
                        canonical = None
                smiles.append(canonical)
                if isinstance(tmpl, list):
                    template = random.choice(tmpl) if len(tmpl) > 0 else None
                else:
                    template = tmpl
                if template is not None:
                    row_dict = row.to_dict()
                    cp = canonical if canonical is not None else "None"
                    row_dict.setdefault("canonical_smiles", cp)
                    if "drug_concentration_with_unit" in row_dict:
                        row_dict["drug_concentration_with_unit"] = str(
                            row_dict["drug_concentration_with_unit"]
                        ).replace(" ", "")
                    prompt = template.format(**row_dict)
                else:
                    prompt = None
                prompts.append(prompt)

        self.arrays = np.concatenate(arrays, axis=0) if len(arrays) > 1 else arrays[0]
        self.prompts = prompts
        self.smiles = smiles
        self.ds_ids = np.concatenate(ds_all) if len(ds_all) > 1 else ds_all[0]
        self.use_controlnet = use_controlnet
        self.keep_nz_ratio = keep_nz_ratio
        self.keep_z_ratio = keep_z_ratio

    def __len__(self):
        return self.arrays.shape[0]

    def __getitem__(self, idx):
        arr = self.arrays[idx].astype(np.float32)
        out = {
            "prompt": self.prompts[idx],
            "smiles": self.smiles[idx],
        }
        if self.use_controlnet:
            control = np.full_like(arr, -10.0, dtype=np.float32)
            nz = np.where(arr != 0)[0]
            if len(nz) > 0:
                mask = np.random.rand(len(nz)) < self.keep_nz_ratio
                keep_idx = nz[mask]
                control[keep_idx] = arr[keep_idx]
            z = np.where(arr == 0)[0]
            if len(z) > 0 and self.keep_z_ratio > 0:
                mask = np.random.rand(len(z)) < self.keep_z_ratio
                keep_idx = z[mask]
                control[keep_idx] = arr[keep_idx]
            out["control"] = control
        ds_id = int(self.ds_ids[idx]) if hasattr(self, "ds_ids") else 0
        return arr, out, ds_id


class StreamingCellDataset(IterableDataset):
    """Stream ``file_size`` H5AD files at a time with support for DataLoader
    workers and distributed training."""

    def __init__(
        self,
        files,
        lengths,
        prompt_templates=None,
        smiles_cols=None,
        ds_ids=None,
        ds_counts=None,
        file_size=1,
        use_controlnet=False,
        keep_nz_ratio=0.5,
        keep_z_ratio=0.0,
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
        if ds_ids is None:
            ds_ids = [0] * len(files)
        assert len(ds_ids) == len(files)
        self.ds_ids = list(ds_ids)
        self.ds_counts = ds_counts if ds_counts is not None else {}
        self.file_size = file_size
        self.use_controlnet = use_controlnet
        self.keep_nz_ratio = keep_nz_ratio
        self.keep_z_ratio = keep_z_ratio

    def __len__(self):
        return sum(self.lengths)

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        num_shards = num_workers * world_size
        shard = rank * num_workers + worker_id

        if self.ds_counts and len(self.ds_counts) > 1:
            weights = np.array([1.0 / self.ds_counts.get(ds, 1) for ds in self.ds_ids])
            prob = weights / weights.sum()
            order = np.random.choice(len(self.files), size=len(self.files), replace=True, p=prob)
        else:
            order = np.random.permutation(len(self.files))

        pad = (-len(order)) % (self.file_size * num_shards)
        if pad > 0:
            order = np.concatenate([order, order[:pad]])

        order = order[shard::num_shards]

        files = [self.files[i] for i in order]
        prompts = [self.prompt_templates[i] for i in order]
        smiles_cols = [self.smiles_cols[i] for i in order]
        ds_ids = [self.ds_ids[i] for i in order]

        for start in range(0, len(files), self.file_size):
            subset_files = files[start : start + self.file_size]
            mem_ds = MemoryCellDataset(
                subset_files,
                prompt_templates=prompts[start : start + self.file_size],
                smiles_cols=smiles_cols[start : start + self.file_size],
                ds_ids=ds_ids[start : start + self.file_size],
                use_controlnet=self.use_controlnet,
                keep_nz_ratio=self.keep_nz_ratio,
                keep_z_ratio=self.keep_z_ratio,
            )
            for idx in range(len(mem_ds)):
                yield mem_ds[idx]


class _StreamingCellIterator:
    def __init__(self, files, prompt_templates, smiles_cols, ds_ids, file_size, use_controlnet, keep_nz_ratio, keep_z_ratio):
        self.files = list(files)
        self.prompt_templates = list(prompt_templates)
        self.smiles_cols = list(smiles_cols)
        self.ds_ids = list(ds_ids)
        self.file_size = file_size
        self.use_controlnet = use_controlnet
        self.keep_nz_ratio = keep_nz_ratio
        self.keep_z_ratio = keep_z_ratio

        self.rng = np.random.default_rng()
        self.file_order = np.arange(len(self.files))
        self.file_order_idx = 0

        self.chunk_buffers = []
        self.chunk_block_idxs = []
        self.chunk_block_order = []
        self.curr_idx = 0

        if len(self.file_order) == 0:
            raise ValueError("No input files for StreamingCellDataset")

        self._load_n_chunks()

    def _load_chunk(self, idx):
        ds = MemoryCellDataset(
            [self.files[idx]],
            prompt_templates=[self.prompt_templates[idx]],
            smiles_cols=[self.smiles_cols[idx]],
            ds_ids=[self.ds_ids[idx]],
            use_controlnet=self.use_controlnet,
            keep_nz_ratio=self.keep_nz_ratio,
            keep_z_ratio=self.keep_z_ratio,
        )
        return ds, len(ds)

    def _load_n_chunks(self):
        if self.file_order_idx >= len(self.file_order):
            self.file_order = self.rng.permutation(len(self.files))
            self.file_order_idx = 0

        self.chunk_buffers = []
        self.chunk_block_idxs = []

        remaining = len(self.file_order) - self.file_order_idx
        if remaining < self.file_size:
            self.file_order = self.rng.permutation(len(self.files))
            self.file_order_idx = 0

        for i in range(self.file_size):
            idx = self.file_order[self.file_order_idx]
            self.file_order_idx += 1
            chunk, n_blocks = self._load_chunk(idx)
            self.chunk_buffers.append(chunk)
            self.chunk_block_idxs.extend([(i, j) for j in range(n_blocks)])

        order = self.rng.permutation(len(self.chunk_block_idxs))
        self.chunk_block_order = order.tolist()
        self.curr_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr_idx >= len(self.chunk_block_order):
            self._load_n_chunks()
        order_idx = self.chunk_block_order[self.curr_idx]
        chunk_id, sample_id = self.chunk_block_idxs[order_idx]
        sample = self.chunk_buffers[chunk_id][sample_id]
        self.curr_idx += 1
        return sample

