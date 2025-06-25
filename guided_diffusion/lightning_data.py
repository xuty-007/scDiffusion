import pytorch_lightning as pl
from torch.utils.data import DataLoader
import scanpy as sc
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder

from .cell_datasets_loader import CellDataset, load_VAE


class CellDataModule(pl.LightningDataModule):
    """Lightning DataModule for cell datasets."""

    def __init__(self, data_dir, batch_size, vae_path=None, train_vae=False, hidden_dim=128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.vae_path = vae_path
        self.train_vae = train_vae
        self.hidden_dim = hidden_dim

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

        if not self.train_vae:
            num_gene = cell_data.shape[1]
            autoencoder = load_VAE(self.vae_path, num_gene, self.hidden_dim)
            cell_data = autoencoder(torch.tensor(cell_data).cuda(), return_latent=True)
            cell_data = cell_data.cpu().detach().numpy()

        self.dataset = CellDataset(cell_data, classes)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
            drop_last=True,
        )
