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
        # self.attn2 = VQAttention(
        #     dim=dim, n_code=n_code,
        #     num_heads=num_heads, dropout=dropout
        # )
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, loss_mask=None, aux_loss=None):
        out1, metrics1 = self.attn1(x, loss_mask, is_train=self.training)
        x = x + self.dropout1(out1)
        # x = x + out1

        # out2, metrics2 = self.attn2(x, loss_mask, is_train=self.training)
        # x = x + self.dropout2(out2)

        l_commit = metrics1['l_commit']
        l_codebook = metrics1['l_codebook']
        metrics = {
            key: metrics1['metrics'][key]
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
