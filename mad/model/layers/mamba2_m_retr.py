"""
mamba2-minimal
==============

A minimal, single-file implementation of the Mamba-2 model in PyTorch.

> **Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality**
> Authors: Tri Dao, Albert Gu
> Paper: https://arxiv.org/abs/2405.21060
"""

import json
from dataclasses import dataclass
from typing import Iterable, NamedTuple, TypeAlias, cast

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import LongTensor, Tensor, nn

Device: TypeAlias = str | torch.device | None


@dataclass
class Mamba2Config:
    d_model: int  # model dimension (D)
    n_layer: int = 24  # number of Mamba-2 layers in the language model
    d_state: int = 128  # state dimension (N)
    d_conv: int = 4  # convolution kernel size
    expand: int = 2  # expansion factor (E)
    headdim: int = 64  # head dimension (P)
    chunk_size: int = 64  # matrix partition size (Q)
    vocab_size: int = 50277
    pad_vocab_size_multiple: int = 16

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                self.pad_vocab_size_multiple
                - self.vocab_size % self.pad_vocab_size_multiple
            )


class InferenceCache(NamedTuple):
    conv_state: Tensor  # (batch, d_inner + 2 * d_state, d_conv)
    ssm_state: Tensor  # (batch, nheads, headdim, d_state)

    @staticmethod
    def alloc(batch_size: int, args: Mamba2Config, device: Device = None):
        return InferenceCache(
            torch.zeros(
                batch_size, args.d_inner + 2 * args.d_state, args.d_conv, device=device
            ),
            torch.zeros(
                batch_size, args.nheads, args.headdim, args.d_state, device=device
            ),
        )


class Mamba2LMHeadModel(nn.Module):
    def __init__(self, args: Mamba2Config, device: Device = None):
        super().__init__()
        self.args = args
        self.device = device

        self.backbone = nn.ModuleDict(
            dict(
                embedding=nn.Embedding(args.vocab_size, args.d_model, device=device),
                layers=nn.ModuleList(
                    [
                        nn.ModuleDict(
                            dict(
                                mixer=Mamba2_m_retr(args, device=device),
                                norm=RMSNorm(args.d_model, device=device),
                            )
                        )
                        for _ in range(args.n_layer)
                    ]
                ),
                norm_f=RMSNorm(args.d_model, device=device),
            )
        )
        self.lm_head = nn.Linear(
            args.d_model, args.vocab_size, bias=False, device=device
        )
        self.lm_head.weight = self.backbone.embedding.weight

    @staticmethod
    def from_pretrained(huggingface_model_id: str, device: Device = None):
        from transformers.utils import CONFIG_NAME, WEIGHTS_NAME
        from transformers.utils.hub import cached_file

        config_path = cached_file(huggingface_model_id, CONFIG_NAME)
        assert config_path, "Failed to get huggingface config file"
        state_dict_path = cached_file(huggingface_model_id, WEIGHTS_NAME)
        assert state_dict_path, "Failed to get huggingface state dict file"

        config = json.load(open(config_path))
        args = Mamba2Config(
            d_model=config["d_model"],
            n_layer=config["n_layer"],
            vocab_size=config["vocab_size"],
            pad_vocab_size_multiple=config["pad_vocab_size_multiple"],
        )

        map_location = "cpu" if device is None else device
        state_dict = torch.load(
            state_dict_path, weights_only=True, map_location=map_location, mmap=True
        )
        model = Mamba2LMHeadModel(args, device=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def forward(
        self, input_ids: LongTensor, h: list[InferenceCache] | list[None] | None = None
    ) -> tuple[LongTensor, list[InferenceCache]]:
        """
        Arguments
            input_ids: (batch, seqlen) tokens from EleutherAI/gpt-neox-20b tokenizer
            h: hidden states for inference step. If present the constant-time
               (wrt sequence length) inference path will be taken, input_ids
               should have shape (batch, 1) containing the next batch of prompt
               token.

        Return (logits, h)
            logits: (batch, seqlen, vocab_size)
            h: updated inference cache after processing input_ids
        """
        seqlen = input_ids.shape[1]

        if h is None:
            h = [None for _ in range(self.args.n_layer)]

        x = self.backbone.embedding(input_ids)
        for i, layer in enumerate(self.backbone.layers):
            y, h[i] = layer.mixer(layer.norm(x), h[i])
            x = y + x

        x = self.backbone.norm_f(x)
        logits = self.lm_head(x)
        return logits[:, :seqlen], cast(list[InferenceCache], h)

    def generate(
        self,
        input_ids: LongTensor,
        max_new_length: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        eos_token_id: int = 0,
    ) -> Iterable[tuple[int, list[InferenceCache]]]:
        prefix, tokens = input_ids[:-1], input_ids[-1:].unsqueeze(0)

        # Process prompt
        # The input sequence to forward (non-inference path) must have length multiple that of chunk_size.
        # We split out excess tokens so that n_chunked tokens can be processed by one forward call and
        # process the rest in multiple inference steps.
        n_chunked = (prefix.shape[0] // self.args.chunk_size) * self.args.chunk_size
        if n_chunked > 0:
            _, h = self(prefix[:n_chunked].unsqueeze(0), None)
        else:
            h = [
                InferenceCache.alloc(1, self.args, device=self.device)
                for _ in range(self.args.n_layer)
            ]
        for i in range(n_chunked, prefix.shape[0]):
            _, h = self(prefix[i : i + 1].unsqueeze(0), h)

        # Generate
        for _ in range(max_new_length):
            with torch.no_grad():
                out, h = self(tokens, h)
            logits = out[0, -1]
            if temperature != 1.0:
                logits = logits / temperature
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, k=top_k)[0][-1]
                logits[indices_to_remove] = -torch.inf
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > 0.5
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = -torch.inf
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            if next_token.item() == eos_token_id:
                return
            tokens = next_token.unsqueeze(0)
            yield cast(int, next_token.item()), h


def segment_and_cosine_similarity_batched(sequence, chunk_size):
    # sequence: (batch_size, sequence_length, dim)
    batch_size, sequence_length, dim = sequence.size()
    num_chunks = sequence_length - chunk_size + 1  # Number of chunks for stride 1 convolution

    # Step 1: Segment the sequence into sliding chunks of size chunk_size (stride 1)
    # Chunks will have shape (batch_size, num_chunks, chunk_size, dim)
    chunks = sequence.unfold(1, chunk_size, 1)  # (batch_size, num_chunks, chunk_size, dim)

    # Step 2: Create convolution kernels by stacking the chunks
    # Reshape for grouped conv1d: (batch_size * num_chunks, dim, chunk_size)
    kernels = chunks.permute(0, 1, 3, 2).contiguous().view(batch_size * num_chunks, dim, chunk_size)

    # Step 3: Reshape the sequence for convolution (batch_size * dim, 1, sequence_length)
    sequence = sequence.permute(0, 2, 1).contiguous().view(batch_size * dim, 1, sequence_length)

    # Step 4: Perform group convolution
    # grouped_kernels shape: (batch_size * num_chunks, dim, chunk_size)
    # grouped_sequence shape: (1, batch_size * dim, sequence_length)
    grouped_kernels = kernels
    grouped_sequence = sequence.view(1, -1, sequence_length)

    grouped_conv_output = F.conv1d(grouped_sequence, grouped_kernels, groups=batch_size)
    # Reshape conv_output: (batch_size, num_chunks, dim, output_size)
    grouped_conv_output = grouped_conv_output.view(batch_size, num_chunks, -1)

    # Step 5: Create a lower triangular mask matrix to zero out unwanted positions
    output_size = grouped_conv_output.size(2)  # Size of the output for each chunk convolution
    mask = torch.tril(torch.ones(num_chunks, output_size, device=grouped_conv_output.device)).unsqueeze(0).expand(batch_size, -1, output_size)

    # Step 6: Apply the mask to the convolution output (this still gives dot products)
    masked_output = grouped_conv_output * mask  # (batch_size, num_chunks, dim, output_size)

    # Step 7: Calculate the norms of the sequence chunks and the corresponding sequence segments
    # Norm of each chunk (for cosine similarity) across the chunk_size and dim dimensions
    chunk_norms = chunks.norm(dim=(2, 3), keepdim=True)  # (batch_size, num_chunks, 1)

    # Unfold sequence segments and compute the norms
    sequence_segments = sequence.view(batch_size, dim, sequence_length).permute(0, 2, 1)  # (batch_size, sequence_length, dim)
    sequence_segments = sequence_segments.unfold(1, chunk_size, 1)  # (batch_size, num_chunks, dim, chunk_size)
    sequence_norms = sequence_segments.norm(dim=(2, 3), keepdim=True)  # (batch_size, num_chunks, 1)

    # Step 8: Multiply norms to get the denominator for cosine similarity
    norms_product = (chunk_norms * sequence_norms.transpose(1, 2)).squeeze(-1)  # (batch_size, num_chunks, output_size)

    # Step 9: Calculate cosine similarity
    cosine_similarity_output = masked_output / (norms_product + 1e-8)  # (batch_size, num_chunks, output_size)

    ### version 1 ###
    # Step 10: Pad the results to the left with chunk_size - 1 zeros (because of convolution alignment)
    left_padding = chunk_size - 1
    padded_output = F.pad(cosine_similarity_output, (left_padding, 0), mode='constant', value=0)  # (batch_size, num_chunks, sequence_length)

    # Step 11: Now pad the top to make it a square matrix of shape LxL (L = sequence_length)
    top_padding = sequence_length - num_chunks
    final_output = F.pad(padded_output, (0, 0, top_padding, 0), mode='constant', value=0)  # (batch_size, sequence_length, sequence_length)

    # Step 12: Correctly set the diagonal to 1 instead of adding to the matrix
    eye_mask = torch.eye(sequence_length, device=final_output.device).unsqueeze(0).expand(batch_size, -1, -1)
    final_output = final_output.masked_fill(eye_mask.bool(), 1)

    return final_output


class Mamba2_m_retr(nn.Module):
    def __init__(self, 
                #  d_model,
                 dim,
                 d_state=128,
                 d_conv=4,
                 expand=2,
                 headdim=4,
                 chunk_size=32,
                 device: Device = "cuda",
                *args, **kwargs
                 ):
        super().__init__()
        self.d_model = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.chunk_size = chunk_size

        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim

        self.device = device

        # Order: (z, x, B, C, dt)
        d_in_proj = 2 * self.d_inner + 2 * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False, device=device)

        conv_dim = self.d_inner + 2 * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=self.d_conv,
            groups=conv_dim,
            padding=self.d_conv - 1,
            device=device,
        )

        self.dt_bias = nn.Parameter(torch.empty(self.nheads, device=device))
        self.A_log = nn.Parameter(torch.empty(self.nheads, device=device))
        self.D = nn.Parameter(torch.empty(self.nheads, device=device))
        self.norm = RMSNorm(self.d_inner, device=device)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, device=device)
        self.mask_scaling_factor = nn.Parameter(torch.tensor([0.], device=device), requires_grad=False)
        self.relevant_keys = None

    def retrieval(self, x, split_point, window_size=3, step_size=1, mode='l2'):

        if window_size == 1:
            assert mode == 'l2'
            assert step_size == 1

            if split_point != 0:
                context = x[:, :split_point, :]
                generated_text = x[:, split_point:, :]

            dot_products = torch.bmm(context, generated_text.transpose(1,2))
            norm_k = torch.norm(context, dim=2)
            norm_q = torch.norm(generated_text, dim=2)
            epsilon = 1e-8

            denominator = torch.clamp(norm_k.unsqueeze(2) * norm_q.unsqueeze(1), min=epsilon)
            cosine_similarity = dot_products / denominator

            cosine_similarity[:, 1:, :] += cosine_similarity[:, :-1, :]
            cosine_similarity /= 2

        else:
            cosine_similarity = segment_and_cosine_similarity_batched(x, window_size)

        return cosine_similarity

    def forward(self, u: Tensor, h: InferenceCache | None = None, similarity_matrix: Tensor | None = None):
        """
        Arguments
            u: (batch, seqlen, d_model) input. seqlen should be a multiple of chunk_size.
            h: hidden states for inference step. Initialized to 0s if not present.

        Return (y, h)
            y: (batch, seqlen, d_model) output
            h: updated inference cache after processing u
        """
        if h:
            return self.step(u, h)

        if similarity_matrix is None:
            cosine_similarity = self.retrieval(u, 16)
        else:
            cosine_similarity = similarity_matrix

        # L_mask = place_matrices_left_bottom(torch.clamp(cosine_similarity, min=0.0, max=0.1), u.shape[1]).unsqueeze(1)
        L_mask = place_matrices_left_bottom(cosine_similarity, u.shape[1]).unsqueeze(1)
        # self.relevant_keys = [(cosine_similarity > 0.1).int().cpu(), (cosine_similarity > 0.3).int().cpu(), (cosine_similarity > 0.5).int().cpu()]

        A = - torch.exp(self.A_log)  # (nheads,)
        zxbcdt = self.in_proj(u)  # (batch, seqlen, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.d_inner,
                self.d_inner + 2 * self.d_state,
                self.nheads,
            ],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)  # (batch, seqlen, nheads)

        # Pad or truncate xBC seqlen to d_conv
        conv_state = F.pad(
            rearrange(xBC, "b l d -> b d l"), (self.d_conv - u.shape[1], 0)
        )
        
        xBC = silu(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, : u.shape[1], :]
        )  # (batch, seqlen, d_inner + 2 * d_state))

        x, B, C = torch.split(
            xBC, [self.d_inner, self.d_state, self.d_state], dim=-1
        )
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)

        y, ssm_state = ssd(
            x * dt.unsqueeze(-1),
            A * dt,
            rearrange(B, "b l n -> b l 1 n"),
            rearrange(C, "b l n -> b l 1 n"),
            self.chunk_size,
            device=self.device,
            L_mask=L_mask,
            mask_scaling_factor = self.mask_scaling_factor,
        )
        y = y + x * self.D.unsqueeze(-1)
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        h = InferenceCache(conv_state, ssm_state)
        return y, h, cosine_similarity


    def step(self, u: Tensor, h: InferenceCache) -> tuple[Tensor, InferenceCache]:
        """Take a single inference step for the current input and hidden state

        Unlike attention-based models, RNN-based models (eg Mamba) does not need
        to look back at all the past tokens to generate a new token. Instead a
        hidden state (initialized to 0s initially) is updated for each input and
        passed to the next inference step. This means that the total inference
        time is linear with respect to the sequence length instead of quadratic
        in attention's case.

        Arguments
            u: (batch, 1, d_model)
            h: initial/running hidden state

        Return (y, h)
            y: (batch, 1, d_model)
            h: updated hidden state
        """
        assert u.shape[1] == 1, "Only one token can be decoded per inference step"

        zxbcdt = self.in_proj(u.squeeze(1))  # (batch, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.d_inner,
                self.d_inner + 2 * self.d_state,
                self.nheads,
            ],
            dim=-1,
        )

        # Advance convolution input
        h.conv_state.copy_(torch.roll(h.conv_state, shifts=-1, dims=-1))
        h.conv_state[:, :, -1] = xBC
        # Convolution step
        xBC = torch.sum(
            h.conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
        )
        xBC += self.conv1d.bias
        xBC = silu(xBC)

        x, B, C = torch.split(
            xBC, [self.d_inner, self.d_state, self.d_state], dim=-1
        )
        A = -torch.exp(self.A_log)  # (nheads,)

        # SSM step
        dt = F.softplus(dt + self.dt_bias)  # (batch, nheads)
        dA = torch.exp(dt * A)  # (batch, nheads)
        x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
        dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)
        h.ssm_state.copy_(h.ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
        y = torch.einsum("bhpn, bn -> bhp", h.ssm_state, C)
        y = y + rearrange(self.D, "h -> h 1") * x
        y = rearrange(y, "b h p -> b (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        return y.unsqueeze(1), h


def segsum(x: Tensor, device: Device = None) -> Tensor:
    """Stable segment sum calculation.

    exp(segsum(A)) produces a 1-semiseparable matrix, which is equivalent to a scalar SSM.

    Source: https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L23-L32
    """
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def decompose_diagonal_and_lower_tri(M, chunk_size):
    original_shape = M.shape
    T = M.shape[-1]
    assert T % chunk_size == 0

    k = T // chunk_size

    reshaped = M.view(*original_shape[:-2], k, chunk_size, k, chunk_size)

    diagonal_blocks = reshaped.diagonal(dim1=-4, dim2=-2).reshape(*original_shape[:-2], k, chunk_size, chunk_size)

    # Create a mask for the lower triangular part (i.e., row index > column index)
    row_idx, col_idx = torch.tril_indices(k, k, offset=-1, device=M.device)
    
    # Use advanced indexing to get the lower triangular blocks
    lower_triangle_blocks = reshaped[..., row_idx, :, col_idx, :].permute(1, 2, 0, 3, 4).contiguous()

    return diagonal_blocks, lower_triangle_blocks, row_idx, col_idx


def ssd(x, A, B, C, chunk_size, initial_states=None, device: Device = None, L_mask = None, mask_scaling_factor = None):
    """Structed State Space Duality (SSD) - the core of Mamba-2

    This is almost the exact same minimal SSD code from the blog post.

    Arguments
        x: (batch, seqlen, n_heads, d_head)
        A: (batch, seqlen, n_heads)
        B: (batch, seqlen, n_heads, d_state)
        C: (batch, seqlen, n_heads, d_state)

    Return
        y: (batch, seqlen, n_heads, d_head)

    Source
     1. https://tridao.me/blog/2024/mamba2-part3-algorithm/
     2. https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L34-L78
    """
    # print(x.shape)
    # print(chunk_size)
    assert x.shape[1] % chunk_size == 0

    # Rearrange into chunks
    # Step 1, 2 and 4 of SSD can be computed in parallel for each chunk across devices (sequence parallel)
    # This is not implemented and left as an exercise for the reader 😜

    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A, device=device))

    if L_mask is not None:
        L_mask_diag, lower_tri, lower_row_idx, lower_col_idx = decompose_diagonal_and_lower_tri(L_mask, chunk_size=chunk_size)
        L = L + mask_scaling_factor * L_mask_diag

    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    # x. Calculate a retrieval affected output Y_off_retr
    Y_off_retr_tmp = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C[:, lower_row_idx, ...], B[:, lower_col_idx, ...], mask_scaling_factor * lower_tri, x[:, lower_col_idx, ...])
    Y_off_retr = torch.zeros_like(x, device=x.device)
    Y_off_retr.scatter_add_(1, lower_row_idx.view(1, -1, 1, 1, 1).expand(Y_off_retr_tmp.shape), Y_off_retr_tmp)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device))
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off + Y_off_retr, "b c l h p -> b (c l) h p")

    return Y, final_state


def place_matrices_left_bottom(original_matrices, N):
    """
    Places each matrix in original_matrices into the left bottom corner of an N x N zero matrix.

    Args:
        original_matrices (torch.Tensor): Tensor of shape [batch_size, p, q]
        N (int): Size of the square matrices

    Returns:
        torch.Tensor: Tensor of shape [batch_size, N, N] with original matrices placed at the left bottom corner.
    """
    batch_size, p, q = original_matrices.shape

    if N < max(p, q):
        raise ValueError("N must be greater than or equal to both dimensions of the original matrices.")

    # Create a batch of N x N zero matrices
    result = torch.zeros(batch_size, N, N, dtype=original_matrices.dtype, device=original_matrices.device)

    # Compute indices for placement
    row_start = N - p
    row_end = N
    col_start = 0
    col_end = q

    # Place each original matrix into the left bottom corner
    result[:, row_start:row_end, col_start:col_end] = original_matrices

    return result


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device: Device = None):
        """Gated Root Mean Square Layer Normalization

        Paper: https://arxiv.org/abs/1910.07467
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))

    def forward(self, x, z=None):
        if z is not None:
            x = x * silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def silu(x):
    """Applies the Sigmoid Linear Unit (SiLU), element-wise.

    Define this manually since torch's version doesn't seem to work on MPS.
    """
    return x * F.sigmoid(x)