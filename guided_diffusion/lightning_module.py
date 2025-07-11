import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pytorch_lightning as pl
import torch as th
import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    LambdaLR,
    SequentialLR,
    LinearLR,
    CosineAnnealingLR,
)
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
        warmup_steps=0,
        lr_scheduler="lambda",
        vae_path=None,
        model_path=None,
        hidden_dim=128,
        prompt_arg=None,
        smiles_arg=None,
        num_samples_per_cond=1,
        resume_checkpoint=False,
    ):
        super().__init__()
        lr = float(lr)
        schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.save_hyperparameters(
            ignore=["model", "diffusion"]
        )
        self.model = model
        self.diffusion = diffusion
        self.autoencoder = None
        self._model_loaded = False
        try:
            self.input_dim = model.layers[0].fc.in_features
        except AttributeError:
            self.input_dim = model.unet.layers[0].fc.in_features

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

        self.use_controlnet = hasattr(self.model, "control_net")
        if self.use_controlnet:
            for p in self.model.unet.parameters():
                p.requires_grad_(False)

        # projector for combined PubMed and Chem embeddings
        self.cond_mapper = th.nn.Sequential(
            th.nn.Linear(1536, 1536),
            th.nn.SiLU(),
            th.nn.Linear(1536, 768),
            th.nn.SiLU(),
            th.nn.Linear(768, 768),
        )
        # initialize to approximately average the two halves at start
        with th.no_grad():
            w = th.zeros(768, 1536)
            eye = th.eye(768)
            w[:, :768] = 0.5 * eye
            w[:, 768:] = 0.5 * eye
            self.cond_mapper[0].weight.copy_(th.eye(1536))
            self.cond_mapper[0].bias.zero_()
            self.cond_mapper[2].weight.copy_(w)
            self.cond_mapper[2].bias.zero_()
            self.cond_mapper[4].weight.copy_(th.eye(768))
            self.cond_mapper[4].bias.zero_()

    def configure_model(self, stage=None):
        if self.autoencoder is None:
            dm = self.trainer.datamodule
            num_gene = getattr(dm, "gene_dim", None)
            if num_gene is None:
                ds = dm.dataset
                path = ds.files[0]
                num_gene = read_h5ad_backed(path).shape[1]
            self.autoencoder = load_VAE(self.hparams.vae_path, num_gene, self.hparams.hidden_dim)
            self.autoencoder.eval()
            for p in self.autoencoder.parameters():
                p.requires_grad_(False)

        if self.use_controlnet and self.hparams.model_path and not self._model_loaded:
            ckpt = th.load(self.hparams.model_path, map_location="cpu")
            sd = ckpt.get("state_dict", ckpt)
            src_prefix = "model.unet." if any(k.startswith("model.unet") for k in sd) else "model."
            tgt_prefix = "model.unet."
            unet_sd = {k[len(src_prefix):]: v for k, v in sd.items() if k.startswith(src_prefix)}
            self.model.unet.load_state_dict(unet_sd, strict=False)
            self._model_loaded = True

    def on_load_checkpoint(self, checkpoint):
        """Remap state dict keys when ControlNet settings differ."""
        state_dict = checkpoint.get("state_dict")
        if state_dict is None:
            return

        ckpt_has_control = any(k.startswith("model.control_net") for k in state_dict)
        ckpt_has_unet_prefix = any(k.startswith("model.unet") for k in state_dict)
        current_has_control = hasattr(self.model, "control_net")

        if getattr(self.hparams, "resume_checkpoint", False) and ckpt_has_control != current_has_control:
            raise ValueError(
                "ControlNet setting mismatch between checkpoint and current model when resuming"
            )

        src_prefix = "model.unet." if ckpt_has_unet_prefix else "model."
        tgt_prefix = "model.unet." if current_has_control else "model."

        new_state = {}
        for k, v in state_dict.items():
            if k.startswith("model.control_net") and not current_has_control:
                continue
            if k.startswith(src_prefix):
                new_key = tgt_prefix + k[len(src_prefix):]
            else:
                new_key = k
            new_state[new_key] = v

        # update in place so Lightning loads the remapped dict
        state_dict.clear()
        state_dict.update(new_state)

        # override autoencoder weights when a path is given
        if self.hparams.vae_path:
            vae_ckpt = th.load(self.hparams.vae_path, map_location="cpu")
            vae_sd = vae_ckpt.get("state_dict", vae_ckpt)
            for k, v in vae_sd.items():
                state_dict[f"autoencoder.{k}"] = v

        # override model weights when a path is specified
        if self.hparams.model_path and current_has_control:
            m_ckpt = th.load(self.hparams.model_path, map_location="cpu")
            m_sd = m_ckpt.get("state_dict", m_ckpt)
            src_pre = "model.unet." if any(k.startswith("model.unet") for k in m_sd) else "model."
            tgt_pre = "model.unet." if current_has_control else "model."
            for k, v in m_sd.items():
                if k.startswith("model.control_net") and not current_has_control:
                    continue
                if k.startswith(src_pre):
                    new_k = tgt_pre + k[len(src_pre):]
                elif k.startswith("diffusion"):
                    new_k = k
                else:
                    continue
                state_dict[new_k] = v

        # drop optimizer state if parameter counts do not match
        opt_states = checkpoint.get("optimizer_states")
        if opt_states:
            try:
                ckpt_params = sum(len(g["params"]) for g in opt_states[0]["param_groups"])
                curr_params = sum(1 for _ in self.model.parameters())
            except Exception:
                ckpt_params = curr_params = -1
            if ckpt_params != curr_params:
                checkpoint.pop("optimizer_states", None)
                checkpoint.pop("lr_schedulers", None)

    def on_fit_start(self):
        self.configure_model()

    def training_step(self, batch, batch_idx):
        x, cond, _ = batch
        with th.no_grad():
            x = self.autoencoder(x, return_latent=True)
            if "control" in cond and cond["control"] is not None:
                ctrl = cond["control"]
                ctrl = self.autoencoder(ctrl, return_latent=True)
                cond["control"] = ctrl
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
                cond_emb = emb_chem

        if cond_emb is not None:
            cond["cond_emb"] = cond_emb

        t, weights = self.hparams.schedule_sampler.sample(x.shape[0], x.device)
        losses = self.diffusion.training_losses(self.model, x, t, model_kwargs=cond)

        if isinstance(self.hparams.schedule_sampler, LossAwareSampler):
            self.hparams.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

        # map timesteps to quartile ids without leaving the device
        quartiles = (t * 4 // self.diffusion.num_timesteps).long()
        # weight each loss term the same way as the original TrainLoop
        for key, val in losses.items():
            weighted = val * weights
            mean_val = weighted.mean()
            self.log(
                key,
                mean_val,
                prog_bar=key in ["loss", "mse"],
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )

            # Log quartile statistics to match the original logger behaviour
            for q in range(4):
                mask = quartiles == q
                if mask.any():
                    self.log(
                        f"{key}_q{q}",
                        weighted[mask].mean().item(),
                        on_step=True,
                        on_epoch=False,
                        sync_dist=True,
                    )

        return (losses["loss"] * weights).mean()

    def on_after_backward(self):
        grad_norm = 0.0
        param_norm = 0.0
        for p in self.model.parameters():
            param_norm += p.data.norm(2).item() ** 2
            if p.grad is not None:
                grad_norm += p.grad.norm(2).item() ** 2
        self.log(
            "grad_norm",
            grad_norm ** 0.5,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        self.log(
            "param_norm",
            param_norm ** 0.5,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        batch_size = batch[0].size(0)
        samples = (self.global_step + 1) * batch_size
        self.log("samples", float(samples), on_step=True, on_epoch=False, sync_dist=True)
        self.log("step", float(self.global_step + 1), on_step=True, on_epoch=False, sync_dist=True)


    def configure_optimizers(self):
        params = []
        if self.use_controlnet:
            params.extend(self.model.control_net.parameters())
        else:
            params.extend(self.model.parameters())

        # always train the conditional mapper
        params.extend(self.cond_mapper.parameters())


        opt = AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        if self.hparams.lr_anneal_steps > 0 or self.hparams.warmup_steps > 0:
            schedulers = []
            milestones = []

            if self.hparams.warmup_steps > 0:
                start_factor = 1.0 / float(self.hparams.warmup_steps)
                warmup = LinearLR(
                    opt,
                    start_factor=start_factor,
                    end_factor=1.0,
                    total_iters=self.hparams.warmup_steps,
                )
                schedulers.append(warmup)
                milestones.append(self.hparams.warmup_steps)

            if self.hparams.lr_anneal_steps > self.hparams.warmup_steps:
                decay_steps = self.hparams.lr_anneal_steps - self.hparams.warmup_steps
                if self.hparams.lr_scheduler.lower() == "cosine":
                    scheduler = CosineAnnealingLR(opt, T_max=decay_steps)
                else:
                    def lr_lambda(step: int):
                        progress = min(float(step) / float(decay_steps), 1.0)
                        return 1.0 - progress

                    scheduler = LambdaLR(opt, lr_lambda)
                schedulers.append(scheduler)

            if schedulers:
                if len(schedulers) == 1:
                    sched = schedulers[0]
                else:
                    sched = SequentialLR(opt, schedulers=schedulers, milestones=milestones)
                return [opt], [{"scheduler": sched, "interval": "step"}]

        return opt

    def on_predict_start(self):
        self.configure_model()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Generate samples for the given conditioning batch."""
        if isinstance(batch, tuple):
            if len(batch) == 3:
                arr, cond, _ = batch
            else:
                arr, cond = batch
            if hasattr(arr, "shape"):
                batch_size = arr.shape[0]
            else:
                batch_size = len(arr)
        elif isinstance(batch, int):
            cond = {}
            batch_size = batch
        else:
            cond = batch if batch else {}
            if cond:
                val = next(iter(cond.values()))
                if isinstance(val, list):
                    batch_size = len(val)
                else:
                    batch_size = val.shape[0]
            else:
                dm = getattr(self.trainer, "datamodule", None)
                batch_size = getattr(dm, "batch_size", 1)

        prompt_arg = self.hparams.prompt_arg
        smiles_arg = self.hparams.smiles_arg

        if prompt_arg is not None:
            cond["prompt"] = [prompt_arg] * batch_size
        if smiles_arg is not None:
            cond["smiles"] = [smiles_arg] * batch_size

        if "control" in cond and cond["control"] is not None:
            with th.no_grad():
                ctrl = cond["control"].to(self.device)
                cond["control"] = self.autoencoder(ctrl, return_latent=True)

        prompts = cond.pop("prompt", None)
        smiles = cond.pop("smiles", None)
        cond_emb = None
        emb_pub = None
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
                cond_emb = emb_chem

        if cond_emb is not None:
            cond["cond_emb"] = cond_emb

        repeat = self.hparams.num_samples_per_cond
        if repeat > 1:
            for k, v in list(cond.items()):
                if th.is_tensor(v):
                    cond[k] = v.repeat_interleave(repeat, dim=0)
                else:
                    cond[k] = [vv for vv in v for _ in range(repeat)]
            batch_size *= repeat

        sample_fn = self.diffusion.p_sample_loop
        latents, _ = sample_fn(
            self.model,
            (batch_size, self.input_dim),
            clip_denoised=True,
            model_kwargs=cond,
            device=self.device,
            progress=False,
            start_time=self.diffusion.betas.shape[0],
        )

        with th.no_grad():
            decoded = self.autoencoder(latents, return_decoded=True)

        return latents.detach().cpu(), decoded.detach().cpu()
