import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# from mad.model.layers.vq_attn import VQAttention  # your custom implementation
# from mad.model.layers.ops.vq_module import LearnableVQ
from mad.model.layers.ops.norm.rmsnorm import RMSNorm


class VQTransformer(nn.Module):
    def __init__(self, vocab_size, layers, layer_cfgs, dim=128, max_length=1024,
                 norm=RMSNorm, position_embeds=None, embed_drop_rate=0.0, c_beta=1e-5):
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

    def forward(self, input_ids, loss_mask=None, targets=None):
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

        if not torch.isfinite(log_probs).all():
            print("NaNs or -inf in log_probs")
            print("logits stats:", logits.min(), logits.max(), logits.mean())
            exit()

        if targets is not None:
            l_lm_premask = F.nll_loss(
                log_probs.permute(0, 2, 1),  # [B, vocab, T]
                targets,
                ignore_index=-100,
                reduction="none"
            )
            assert loss_mask is not None, "loss_mask must be provided during training"
            l_lm_unscaled = (l_lm_premask * loss_mask).sum() / (loss_mask.sum() + 1e-6)
        else:
            l_lm_unscaled = torch.tensor(0.0, device=log_probs.device)

        # print(aux_loss['l_commit'])

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
