import math
import random

from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset

import scanpy as sc
import torch
import sys
sys.path.append('..')
from VAE.VAE_model import VAE
from sklearn.preprocessing import LabelEncoder


PROMPT_TEMPLATE = (
    "Drug name: {drug} (PubChem CID: {pubchem_cid}) has the canonical SMILES {canonical_smiles}. "
    "The drug was tested at {drug_concentration} on the cell line {cell_line_id}. "
    "Biological metrics include gene count of {gene_count:.2f}, transcript count of {tscp_count:.2f}, "
    "mitochondrial read percentage of {pcnt_mito:.2%}, and mRNA read count of {mread_count:.2f}. "
    "The mechanism of action (MOA) is broadly {moa_broad}, specifically {moa_fine}, and targets are currently {targets}. "
    "Clinical status: approved by humans - {human_approved}, in clinical trials - {clinical_trials}. "
    "Additional notes: {gpt_notes_approval}"
)

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
        cell_data,
        class_name,
        prompts=None,
        smiles=None,
        use_controlnet=False,
        keep_ratio=0.5,
    ):
        super().__init__()
        self.data = cell_data
        self.class_name = class_name
        self.prompts = prompts
        self.smiles = smiles
        self.use_controlnet = use_controlnet
        self.keep_ratio = keep_ratio

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        arr = self.data[idx]
        out_dict = {}
        if self.class_name is not None:
            out_dict["y"] = np.array(self.class_name[idx], dtype=np.int64)
        if self.prompts is not None:
            out_dict["prompt"] = self.prompts[idx]
        if self.smiles is not None:
            out_dict["smiles"] = self.smiles[idx]
        if self.use_controlnet:
            control = np.full_like(arr, -1.0, dtype=np.float32)
            nz = np.where(arr != 0)[0]
            if len(nz) > 0:
                mask = np.random.rand(len(nz)) < self.keep_ratio
                keep_idx = nz[mask]
                control[keep_idx] = arr[keep_idx]
            out_dict["control"] = control
        return arr, out_dict

