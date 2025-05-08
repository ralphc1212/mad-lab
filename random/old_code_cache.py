# vq_pl_model_wrapper.py

import torch
import typing as tp
import pytorch_lightning as pl
import torchmetrics as met
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from mad.metrics import Accuracy

class VQPLModelWrap(pl.LightningModule):
    """
    PyTorch Lightning wrapper for VQ-based language models.

    Args:
        model (nn.Module): VQTransformerMAD or similar model.
        mad_config (MADConfig): configuration object.
        metrics (list, optional): list of metrics to compute (e.g., ["acc", "ppl"]).
    """
    def __init__(self, model, mad_config, metrics: list = ['acc', 'ppl']):
        super().__init__()
        self.model = model
        self.mad_config = mad_config
        self.loss_fn = nn.NLLLoss(ignore_index=self.mad_config.target_ignore_index)
        self.vq_loss_weight = getattr(self.mad_config, 'vq_loss_weight', 0.01)
        self.instantiate_metrics(metrics)
        self.save_hyperparameters('mad_config')

    def instantiate_metrics(self, metrics: list) -> None:
        mets = []
        for m in metrics:
            if m == 'acc':
                mets.append(Accuracy(
                    num_classes=self.model.vocab_size,
                    ignore_index=self.mad_config.target_ignore_index
                ))
            elif m == 'ppl':
                mets.append(met.text.Perplexity(ignore_index=self.mad_config.target_ignore_index))
            elif isinstance(m, met.Metric):
                mets.append(m)
            else:
                raise ValueError(f"Invalid metric: {m}. Use 'acc', 'ppl', or torchmetrics.Metric.")

        mets = met.MetricCollection(mets)
        self.train_metrics = mets.clone(prefix='train/')
        self.test_metrics = mets.clone(prefix='test/')

    def forward(self, input_ids, loss_mask=None, targets=None):
        return self.model(input_ids, loss_mask=loss_mask, targets=targets)

    def step(self, batch: tuple, batch_idx: int):
        input_ids, targets = batch
        out_dict = self(input_ids)
        log_probs = out_dict['logprobs']

        # Language modeling loss
        model_out = self(input_ids, loss_mask=(input_ids != self.mad_config.target_ignore_index), targets=targets)
        total_loss = model_out["loss"]
        log_probs = model_out["logprobs"]

        return total_loss, log_probs, targets, {
            'l_ce': model_out['l_lm_unscaled'].detach(),
            'l_vq': (model_out['l_commit'] + model_out['l_codebook']).detach(),
            **{f"metrics/{k}": v.item() for k, v in out_dict['metrics'].items()}
        }

    def phase_step(self, batch, batch_idx, phase='train'):
        loss, log_probs, targets, extras = self.step(batch, batch_idx)
        self.log(f'{phase}/Loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{phase}/Loss_CE', extras['l_ce'], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{phase}/Loss_VQ', extras['l_vq'], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        metrics = getattr(self, f'{phase}_metrics')(log_probs, targets)
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log_dict({f"{phase}/{k}": v for k, v in extras.items() if k.startswith('metrics/')},
                      on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        return {'loss': loss, 'outputs': log_probs, 'targets': targets}

    def training_step(self, batch, batch_idx):
        return self.phase_step(batch, batch_idx, phase='train')

    def validation_step(self, batch, batch_idx):
        return self.phase_step(batch, batch_idx, phase='test')

    def test_step(self, batch, batch_idx):
        return self.phase_step(batch, batch_idx, phase='test')

    def configure_optimizers(self):
        if self.mad_config.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.mad_config.lr,
                weight_decay=self.mad_config.weight_decay
            )
        elif self.mad_config.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.mad_config.lr,
                weight_decay=self.mad_config.weight_decay
            )
        else:
            raise ValueError(f"invalid optimizer: {self.mad_config.optimizer}")

        if self.mad_config.scheduler == 'none':
            return optimizer
        elif self.mad_config.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.mad_config.epochs,
                eta_min=self.mad_config.min_lr,
                last_epoch=-1
            )
            return {'optimizer': optimizer, 'scheduler': scheduler}
        elif self.mad_config.scheduler == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=self.mad_config.plateau_patience,
                factor=self.mad_config.plateau_factor,
                min_lr=self.mad_config.min_lr,
                verbose=True
            )
            return {'optimizer': optimizer, 'scheduler': scheduler, 'monitor': "test/Loss_epoch"}
        else:
            raise ValueError(f"invalid scheduler: {self.mad_config.scheduler}")

# vq_language_model.py

# Compatible MAD-LAB-style VQ Transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# from mad.model.layers.vq_attn import VQAttention  # your custom implementation
# from mad.model.layers.ops.vq_module import LearnableVQ
from mad.model.layers.ops.norm.rmsnorm import RMSNorm


class VQTransformer(nn.Module):
    def __init__(self, vocab_size, layers, layer_cfgs, dim=128, max_length=1024,
                 norm=RMSNorm, position_embeds=None, embed_drop_rate=0.0, c_beta=0.0001):
        super().__init__()
        assert len(layer_cfgs) == len(layers)
        assert all(cfg['dim'] == dim for cfg in layer_cfgs)

        self.vocab_size = vocab_size
        self.token_embeds = nn.Embedding(vocab_size, dim)
        self.c_beta = c_beta

        if position_embeds is not None:
            position_embeds = position_embeds(max_length, dim)
        self.position_embeds = (
            position_embeds.weight if isinstance(position_embeds, nn.Embedding) else position_embeds
        )
        self.drop_embed = nn.Dropout(embed_drop_rate)

        self.model = nn.ModuleList([])
        for layer_cls, layer_cfg in zip(layers, layer_cfgs):
            self.model.append(nn.Sequential(norm(layer_cfg['dim']), layer_cls(**layer_cfg)))

        self.unembed = nn.Sequential(norm(layer_cfgs[-1]['dim']), nn.Linear(dim, vocab_size))
        self.apply(self._init_weights)

    def embed(self, input_ids, position_ids=None):
        B, T = input_ids.shape
        x = self.token_embeds(input_ids)
        if self.position_embeds is not None:
            if position_ids is None:
                position_ids = torch.arange(T, dtype=torch.long, device=input_ids.device)
            pos_emb = self.position_embeds[position_ids]
            x = x + pos_emb.to(x.device)
        return self.drop_embed(x)

    def forward(self, input_ids, loss_mask, targets=None):
        x = self.embed(input_ids)
        aux_loss = {}

        for layer in self.model:
            # print('0\n', layer[0], '\n1\n', layer[1])
            x_normed = layer[0](x)  # apply norm
            # print(layer[1].forward.__code__.co_varnames)
            if 'aux_loss' in layer[1].forward.__code__.co_varnames:
                x_, return_dict = layer[1](x_normed, loss_mask, aux_loss=aux_loss)
                aux_loss = return_dict['aux_loss']
                x = x + x_
            else:
                x = x + layer[1](x_normed)

        logits = self.unembed(x)
        log_probs = F.log_softmax(logits, dim=-1)

        if targets is not None:
            gather_index = targets.clone()
            gather_index[targets == -100] = 0  # safe dummy index
            l_lm_premask = -torch.gather(log_probs, dim=-1, index=gather_index.unsqueeze(-1)).squeeze(-1)
            l_lm_premask = l_lm_premask * (targets != -100)
            l_lm_unscaled = l_lm_premask.sum() / (loss_mask.sum() + 1e-6)
        else:
            l_lm_unscaled = torch.tensor(0.0, device=log_probs.device)

        # attach aux loss components
        return {
            'logprobs': log_probs,
            'l_commit': aux_loss.get('l_commit', torch.tensor(0.0, device=x.device)),
            'l_codebook': aux_loss.get('l_codebook', torch.tensor(0.0, device=x.device)),
            'l_lm_unscaled': l_lm_unscaled,
            'loss': l_lm_unscaled + self.c_beta * aux_loss.get('l_commit', 0.0) + aux_loss.get('l_codebook', 0.0),
            'metrics': {k.split('metrics/')[1]: v for k, v in aux_loss.items() if k.startswith('metrics/')}
        }

    def _init_weights(self, m, initializer_range=0.02):
        if isinstance(m, nn.Linear):
            if m.bias is not None and not getattr(m.bias, '_no_reinit', False):
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=initializer_range)

vq_attn.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mad.model.layers.ops.vq_module import LearnableVQ  # you need to plug in the version we wrote earlier

MASK_INFTY_APPROX = 1e30

class VQAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_code: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        causal: bool = True,
        c_gamma: float = 0.99,
        *args, **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.n_code = n_code
        self.causal = causal

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.vq = LearnableVQ(n_code=n_code, d_k=self.head_dim, n_head=num_heads, c_gamma=c_gamma)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, loss_mask=None, is_train=True):
        B, L, D = x.shape
        H = self.num_heads
        q = self.q_proj(x).view(B, L, H, self.head_dim).transpose(1, 2)  # [B, H, L, d]
        k = self.k_proj(x).view(B, L, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, self.head_dim).transpose(1, 2)

        # Quantize keys
        vq_out = self.vq(k, loss_mask=loss_mask, is_train=is_train, return_metrics=True)
        k_hat = vq_out['quantized']  # [B, H, L, d]
        shortcodes = vq_out['shortcodes']
        l_commit = vq_out['l_commit']
        l_codebook = vq_out['l_codebook']
        metrics = vq_out['metrics']

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn_logits = torch.einsum("bhlk,bhtk->bhlt", q, k_hat) * scale

        if self.causal:
            causal_mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))
            attn_logits = attn_logits.masked_fill(~causal_mask[None, None, :, :], -MASK_INFTY_APPROX)

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.einsum("bhlt,bhtd->bhld", attn_weights, v)  # [B, H, L, d]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)

        output = self.out_proj(attn_output)
        return output, dict(
            l_commit=l_commit,
            l_codebook=l_codebook,
            shortcodes=shortcodes,
            metrics=metrics  # <- for logging/debugging
        )


class VQTransformerLayer(nn.Module):
    def __init__(self, 
                dim: int,
                n_code: int,
                num_heads: int = 8,
                dropout: float = 0.1,
                causal: bool = True,
                c_gamma: float = 0.99,
                *args, **kwargs
        ):
        super().__init__()
        self.attn1 = VQAttention(
            dim=dim, n_code=n_code,
            num_heads=num_heads, dropout=dropout
        )
        self.attn2 = VQAttention(
            dim=dim, n_code=n_code,
            num_heads=num_heads, dropout=dropout
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, loss_mask=None, aux_loss=None):
        out1, metrics1 = self.attn1(x, loss_mask, is_train=self.training)
        x = x + self.dropout1(out1)
        out2, metrics2 = self.attn2(x, loss_mask, is_train=self.training)
        x = x + self.dropout2(out2)

        l_commit = metrics1['l_commit'] + metrics2['l_commit']
        l_codebook = metrics1['l_codebook'] + metrics2['l_codebook']
        metrics = {
            key: (metrics1['metrics'][key] + metrics2['metrics'][key]) / 2
            for key in metrics1['metrics']
        }

        # accumulate VQ loss in aux_loss dictionary
        if aux_loss is not None:
            aux_loss['l_commit'] = aux_loss.get('l_commit', 0.0) + l_commit
            aux_loss['l_codebook'] = aux_loss.get('l_codebook', 0.0) + l_codebook
            for k, v in metrics.items():
                aux_loss[f'metrics/{k}'] = aux_loss.get(f'metrics/{k}', 0.0) + v

        return x, dict(
            # l_commit=l_commit,
            # l_codebook=l_codebook,
            # shortcodes=shortcodes,
            aux_loss=aux_loss,
            metrics=metrics  # <- for logging/debugging
        )


# vq = VQAttention({'dim': 128, 'n_code': 16, 'max_length': 1280})
# vecs = torch.randn(4, 64, 8*128)  # [B, H, L, d_k]
# loss_mask = torch.ones(4, 64)
# out = vq(vecs, loss_mask, is_train=True)

vq_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableVQ(nn.Module):
    def __init__(self, n_code, d_k, n_head, dtype=torch.float32, c_gamma=0.99):
        super().__init__()
        self.n_code = n_code
        self.d_k = d_k
        self.n_head = n_head
        self.dtype = dtype
        self.c_gamma = c_gamma

        # EMA statistics for each head's codebook
        self.register_buffer('c_sum', torch.zeros(n_head, n_code, d_k, dtype=dtype))
        self.register_buffer('c_count', torch.ones(n_head, n_code, dtype=dtype))  # prevent div0

    def get_codebook(self):
        # Prevent division by small values
        count = torch.clamp(self.c_count, min=1e-2).unsqueeze(-1)  # [H, S, 1]
        codebook = self.c_sum / count  # [H, S, d_k]
        return codebook.detach()  # stop gradient like sg()

    def get_shortcodes(self, vecs, codebook):
        # vecs: [B, H, L, d_k], codebook: [H, S, d_k]
        # B, H, L, d_k = vecs.shape
        # S = codebook.shape[1]

        # ||x - c||^2 = ||x||^2 - 2x.c + ||c||^2
        x_sq = (vecs ** 2).sum(-1, keepdim=True)  # [B, H, L, 1]
        c_sq = (codebook ** 2).sum(-1)  # [H, S]
        x_dot_c = torch.einsum("bhlk,hsk->bhls", vecs, codebook)  # [B, H, L, S]
        dist2 = x_sq - 2 * x_dot_c + c_sq[None, :, None, :]  # [B, H, L, S]

        shortcodes = dist2.argmin(dim=-1)  # [B, H, L]
        errs2 = F.relu(dist2.min(dim=-1).values)  # [B, H, L]
        return shortcodes, errs2

    def get_codewords(self, shortcodes, codebook):
        # shortcodes: [B, H, L], codebook: [H, S, d_k]
        # B, H, L = shortcodes.shape
        S, d_k = codebook.shape[1], codebook.shape[2]
        one_hot = F.one_hot(shortcodes, num_classes=S).float()  # [B, H, L, S]
        codewords = torch.einsum("bhls,hsk->bhlk", one_hot, codebook)  # [B, H, L, d_k]
        return codewords

    def update_ema(self, vecs, shortcodes, loss_mask, n_device=1, n_block_per_update=1):
        # vecs: [B, H, L, d_k], shortcodes: [B, H, L], loss_mask: [B, L]
        # B, H, L, d_k = vecs.shape
        S = self.n_code
        loss_mask = loss_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, L, 1]
        one_hot = F.one_hot(shortcodes, num_classes=S).float()  # [B, H, L, S]

        r = one_hot * loss_mask  # mask out padding tokens
        r_sum = r.sum(dim=2)  # [B, H, S]
        # r_vec = torch.einsum("bhls,bhlk->bhsd", r, vecs)  # [B, H, S, d_k]
        r_vec = torch.einsum("bhls,bhlk->bhsk", r, vecs)

        scale = n_device * n_block_per_update
        c_sum_hat = scale * r_vec.sum(dim=0)  # [H, S, d_k]
        c_count_hat = scale * r_sum.sum(dim=0)  # [H, S]

        self.c_sum.data = (1 - self.c_gamma) * c_sum_hat + self.c_gamma * self.c_sum.data
        self.c_count.data = (1 - self.c_gamma) * c_count_hat + self.c_gamma * self.c_count.data

    def forward(self, vecs, loss_mask, is_train=True, return_metrics=False):
        # vecs: [B, H, L, d_k]
        # B, H, L, d_k = vecs.shape
        codebook = self.get_codebook()  # [H, S, d_k]

        shortcodes, errs2 = self.get_shortcodes(vecs, codebook)
        quantized = self.get_codewords(shortcodes, codebook)  # [B, H, L, d_k]

        # Straight-through estimator
        vecs_hat = quantized.detach() + (vecs - vecs.detach())

        if is_train:
            # Commitment loss
            if loss_mask is None:
                # default to all valid positions
                loss_mask = torch.ones(errs2.shape[0], errs2.shape[2], device=errs2.device)

            commit_loss = (loss_mask.unsqueeze(1) * errs2).sum() / (loss_mask.sum() + 1e-6)

            # Codebook update (loss not used directly, but gradients could be)
            self.update_ema(vecs.detach(), shortcodes.detach(), loss_mask)

            codebook_loss = torch.tensor(0.0, device=vecs.device)  # placeholder
        else:
            commit_loss = torch.tensor(0.0, device=vecs.device)
            codebook_loss = torch.tensor(0.0, device=vecs.device)

        output = {
            "quantized": vecs_hat,
            "shortcodes": shortcodes,
            "l_commit": commit_loss,
            "l_codebook": codebook_loss,
        }

        if return_metrics:
            # Add metric calculations here if desired
            output["metrics"] = {}

        return output


# vq = LearnableVQ(n_code=512, d_k=128, n_head=8)
# vecs = torch.randn(4, 8, 64, 128)  # [B, H, L, d
