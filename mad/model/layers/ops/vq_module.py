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

        # Random initialization of codebook
        with torch.no_grad():
            nn.init.normal_(self.c_sum, mean=0.0, std=1.0)
            self.c_count.fill_(10.0)

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
        """
        shortcodes: [B, H, L]
        codebook: [H, S, d_k]
        Return: [B, H, L, d_k]
        """
        B, H, L = shortcodes.shape
        S, d_k = codebook.shape[1], codebook.shape[2]

        # Flatten for easier indexing
        flat_shortcodes = shortcodes.reshape(-1)  # [B * H * L]

        # Index codebook with proper advanced indexing
        head_idx = torch.arange(H, device=shortcodes.device).view(1, H, 1).expand(B, H, L).reshape(-1)  # [B * H * L]

        # Gather codewords: codebook[head_idx, shortcode] → [B * H * L, d_k]
        selected = codebook[head_idx, flat_shortcodes]  # [B * H * L, d_k]

        # Reshape to [B, H, L, d_k]
        codewords = selected.view(B, H, L, d_k)

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
            output["metrics"] = {
                "codebook_usage": (shortcodes.float().mean().item()),  # or entropy
            }

        # print(output['quantized'].shape)
        # print(output['shortcodes'].shape)
        # print(output['l_commit'])
        # exit()

        return output
