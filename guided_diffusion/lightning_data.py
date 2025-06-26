import pytorch_lightning as pl
from torch.utils.data import DataLoader
import scanpy as sc
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder

from .cell_datasets_loader import CellDataset, PROMPT_TEMPLATE


class CellDataModule(pl.LightningDataModule):
    """Lightning DataModule for cell datasets."""

    def __init__(self, data_dir, batch_size, vae_path=None, train_vae=False, hidden_dim=128, use_controlnet=False, keep_ratio=0.5):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.vae_path = vae_path
        self.train_vae = train_vae
        self.hidden_dim = hidden_dim
        self.use_controlnet = use_controlnet
        self.keep_ratio = keep_ratio

    def setup(self, stage=None):
        adata = sc.read_h5ad(self.data_dir)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.filter_cells(adata, min_genes=10)
        adata.var_names_make_unique()

        classes = adata.obs["celltype"].values
        label_encoder = LabelEncoder()
        labels = classes
        label_encoder.fit(labels)
        classes = label_encoder.transform(labels)

        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        cell_data = adata.X.toarray()

        prompts = []
        smiles = adata.obs["canonical_smiles"].tolist()
        for _, row in adata.obs.iterrows():
            prompt = PROMPT_TEMPLATE.format(
                drug=row["drug"],
                pubchem_cid=row["pubchem_cid"],
                canonical_smiles=row["canonical_smiles"],
                drug_concentration=row["drug concentration with unit"],
                cell_line_id=row["cell_line_id"],
                gene_count=row["gene_count"],
                tscp_count=row["tscp_count"],
                pcnt_mito=row["pcnt_mito"],
                mread_count=row["mread_count"],
                moa_broad=row["moa-broad"],
                moa_fine=row["moa-fine"],
                targets=row["targets"],
                human_approved=row["human-approved"],
                clinical_trials=row["clinical-trials"],
                gpt_notes_approval=row["gpt-notes-approval"],
            )
            prompts.append(prompt)
        self.dataset = CellDataset(
            cell_data,
            classes,
            prompts,
            smiles,
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
        )
