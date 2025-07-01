import pytorch_lightning as pl
import torch as th
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel

from .cell_datasets_loader import load_VAE, read_h5ad_backed

PUBMED_MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
CHEMBERT_MODEL_NAME = "ChemBERTa-2"

from .resample import LossAwareSampler, UniformSampler


class DiffusionLitModule(pl.LightningModule):
    """LightningModule for training diffusion models."""

    def __init__(
        self,
        model,
        diffusion,
        lr=1e-4,
        weight_decay=0.0,
        schedule_sampler=None,
        lr_anneal_steps=0,
        vae_path=None,
        train_vae=False,
        hidden_dim=128,
    ):
        super().__init__()
        self.model = model
        self.diffusion = diffusion
        # ensure learning rate is a float. some callers may accidentally pass a
        # schedule sampler here which would break the optimizer
        self.lr = float(lr)
        self.weight_decay = weight_decay
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.lr_anneal_steps = lr_anneal_steps
        self.vae_path = vae_path
        self.train_vae = train_vae
        self.hidden_dim = hidden_dim
        self.autoencoder = None

        # Preload tokenizers and encoders for conditional embeddings
        self.pubmed_tokenizer = AutoTokenizer.from_pretrained(PUBMED_MODEL_NAME)
        self.pubmed_encoder = AutoModel.from_pretrained(PUBMED_MODEL_NAME)
        self.chem_tokenizer = AutoTokenizer.from_pretrained(CHEMBERT_MODEL_NAME)
        self.chem_token_encoder = AutoModel.from_pretrained(CHEMBERT_MODEL_NAME)
        self.pubmed_encoder.eval()
        self.chem_token_encoder.eval()
        for p in self.pubmed_encoder.parameters():
            p.requires_grad_(False)
        for p in self.chem_token_encoder.parameters():
            p.requires_grad_(False)
        self.embed_dim = self.pubmed_encoder.config.hidden_size

        # mapper for combined PubMed and Chem embeddings
        self.cond_mapper = th.nn.Linear(1536, 768)
        # initialize to average the two halves
        with th.no_grad():
            w = th.zeros(768, 1536)
            eye = th.eye(768)
            w[:, :768] = 0.5 * eye
            w[:, 768:] = 0.5 * eye
            self.cond_mapper.weight.copy_(w)
            self.cond_mapper.bias.zero_()

        # mapper for standalone Chem embeddings
        self.chem_mapper = th.nn.Linear(768, 768)
        with th.no_grad():
            self.chem_mapper.weight.copy_(th.eye(768))
            self.chem_mapper.bias.zero_()

    def configure_model(self, stage=None):
        if not self.train_vae and self.autoencoder is None:
            dm = self.trainer.datamodule
            num_gene = getattr(dm, "gene_dim", None)
            if num_gene is None:
                ds = dm.dataset
                path = ds.files[0]
                num_gene = read_h5ad_backed(path).shape[1]
            self.autoencoder = load_VAE(self.vae_path, num_gene, self.hidden_dim)
            self.autoencoder.eval()
            for p in self.autoencoder.parameters():
                p.requires_grad_(False)

    def on_fit_start(self):
        self.configure_model()

    def training_step(self, batch, batch_idx):
        x, cond = batch
        if not self.train_vae:
            with th.no_grad():
                x = self.autoencoder(x, return_latent=True)
        prompts = cond.pop("prompt", None)
        smiles = cond.pop("smiles", None)
        cond_emb = None
        emb_pub = None
        batch_size = x.shape[0]
        all_prompt_none = prompts is None or all(p is None for p in prompts)
        all_smiles_none = smiles is None or all(s is None for s in smiles)
        if prompts is not None:
            valid_idx = [i for i, p in enumerate(prompts) if p is not None]
            if valid_idx:
                with th.no_grad():
                    tok = self.pubmed_tokenizer(
                        [prompts[i] for i in valid_idx],
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    )
                    tok = {k: v.to(self.device) for k, v in tok.items()}
                    emb = self.pubmed_encoder(**tok).last_hidden_state[:, 0]
                emb_pub = th.zeros(len(prompts), emb.shape[1], device=self.device)
                emb_pub[valid_idx] = emb
            else:
                emb_pub = th.zeros(batch_size, self.embed_dim, device=self.device)
        else:
            emb_pub = th.zeros(batch_size, self.embed_dim, device=self.device)

        emb_chem = None
        if smiles is not None:
            valid_idx = [i for i, s in enumerate(smiles) if s is not None]
            if valid_idx:
                with th.no_grad():
                    tok = self.chem_tokenizer(
                        [smiles[i] for i in valid_idx],
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    )
                    tok = {k: v.to(self.device) for k, v in tok.items()}
                    emb = self.chem_token_encoder(**tok).last_hidden_state[:, 0]
                emb_chem = th.zeros(len(smiles), emb.shape[1], device=self.device)
                emb_chem[valid_idx] = emb
            else:
                emb_chem = th.zeros(batch_size, self.embed_dim, device=self.device)
        else:
            emb_chem = th.zeros(batch_size, self.embed_dim, device=self.device)

        if not (all_prompt_none and all_smiles_none):
            if emb_pub is not None and emb_chem is not None:
                cond_emb = self.cond_mapper(th.cat([emb_pub, emb_chem], dim=1))
            elif emb_pub is not None and not all_prompt_none:
                cond_emb = emb_pub
            elif emb_chem is not None and not all_smiles_none:
                cond_emb = self.chem_mapper(emb_chem)

        if cond_emb is not None:
            cond["cond_emb"] = cond_emb

        t, weights = self.schedule_sampler.sample(x.shape[0], x.device)
        losses = self.diffusion.training_losses(self.model, x, t, model_kwargs=cond)

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

        # map timesteps to quartile ids without leaving the device
        quartiles = (t * 4 // self.diffusion.num_timesteps).long()
        # weight each loss term the same way as the original TrainLoop
        for key, val in losses.items():
            weighted = val * weights
            mean_val = weighted.mean()
            self.log(key, mean_val, prog_bar=key in ["loss", "mse"], on_step=True, on_epoch=False)

            # Log quartile statistics to match the original logger behaviour
            for q in range(4):
                mask = quartiles == q
                if mask.any():
                    self.log(
                        f"{key}_q{q}",
                        weighted[mask].mean().item(),
                        on_step=True,
                        on_epoch=False,
                    )

        return (losses["loss"] * weights).mean()

    def on_after_backward(self):
        grad_norm = 0.0
        param_norm = 0.0
        for p in self.model.parameters():
            param_norm += p.data.norm(2).item() ** 2
            if p.grad is not None:
                grad_norm += p.grad.norm(2).item() ** 2
        self.log("grad_norm", grad_norm ** 0.5, on_step=True, on_epoch=False)
        self.log("param_norm", param_norm ** 0.5, on_step=True, on_epoch=False)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        batch_size = batch[0].size(0)
        samples = (self.global_step + 1) * batch_size
        self.log("samples", float(samples), on_step=True, on_epoch=False)
        self.log("step", float(self.global_step + 1), on_step=True, on_epoch=False)


    def configure_optimizers(self):
        opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.lr_anneal_steps > 0:
            scheduler = th.optim.lr_scheduler.LambdaLR(
                opt,
                lambda step: 1 - min(step, self.lr_anneal_steps) / float(self.lr_anneal_steps),
            )
            return [opt], [
                {
                    "scheduler": scheduler,
                    "interval": "step",
                }
            ]
        return opt
