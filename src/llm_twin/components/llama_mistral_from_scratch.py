"""
First-Principles LLaMA & Mistral Architecture in PyTorch.
Reference: Umar Jamil — "LLaMA & Mistral Architectures from Scratch in PyTorch"

Implements the four foundational mathematical building blocks of modern open-weights LLMs:
1. RMSNorm (Root Mean Square Layer Normalization — Zhang & Sennrich, 2019)
2. RoPE (Rotary Positional Embeddings — Su et al., 2021)
3. SwiGLU Feed-Forward Network (Shazeer, 2020)
4. Grouped-Query Attention (GQA — Ainslie et al., 2023) with KV Cache & Mistral Sliding Window Attention
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    Replaces standard LayerNorm by omitting mean-centering:
        RMS(x) = sqrt( (1 / d) * sum_{i=1}^d x_i^2 + eps )
        y_i = (x_i / RMS(x)) * gamma_i
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


def precompute_theta_pos_frequencies(
    head_dim: int, max_seq_len: int, theta: float = 10000.0, device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Precomputes complex exponential frequencies e^(i * m * theta_j) for Rotary Positional Embeddings (RoPE):
        theta_j = 10000^(-2(j-1)/d),  j in {1, 2, ..., d/2}
    Returns a complex tensor of shape (max_seq_len, head_dim // 2).
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

    theta_numerator = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    freqs = 1.0 / (theta ** (theta_numerator / head_dim))
    m = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs_outer = torch.outer(m, freqs)  # (max_seq_len, head_dim // 2)
    freqs_complex = torch.polar(torch.ones_like(freqs_outer), freqs_outer)
    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor) -> torch.Tensor:
    """
    Applies Rotary Positional Embeddings (RoPE) to query or key tensor x.
    Args:
        x: Tensor of shape (batch_size, seq_len, n_heads, head_dim)
        freqs_complex: Complex tensor of shape (seq_len, head_dim // 2)
    Returns:
        Rotated tensor of shape (batch_size, seq_len, n_heads, head_dim)
    """
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_reshaped = freqs_complex.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim // 2)
    x_rotated = x_complex * freqs_reshaped
    x_out = torch.view_as_real(x_rotated).reshape(*x.shape)
    return x_out.type_as(x)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats Key/Value heads n_rep times to match the number of Query heads in Grouped-Query Attention (GQA).
    Args:
        x: Tensor of shape (batch_size, seq_len, n_kv_heads, head_dim)
        n_rep: Group size G = n_heads // n_kv_heads
    Returns:
        Tensor of shape (batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    """
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) with RoPE, KV Cache, and Mistral Sliding Window Attention (SWA).
    When n_kv_heads == n_heads -> Standard Multi-Head Attention (MHA)
    When n_kv_heads == 1       -> Multi-Query Attention (MQA)
    When 1 < n_kv_heads < n_heads -> Grouped-Query Attention (GQA)
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        sliding_window: Optional[int] = None,
    ):
        super().__init__()
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = dim // n_heads
        self.sliding_window = sliding_window

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def _build_causal_sliding_window_mask(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Builds a causal attention mask, optionally restricted to a Mistral sliding window W:
        Token i attends to token j iff j <= i and (i - j) < W.
        """
        idx = torch.arange(seq_len, device=device)
        row_idx = idx.unsqueeze(1)
        col_idx = idx.unsqueeze(0)
        valid = col_idx <= row_idx
        if self.sliding_window is not None and self.sliding_window > 0:
            valid = valid & ((row_idx - col_idx) < self.sliding_window)

        mask = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
        mask = mask.masked_fill(~valid, float("-inf"))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    def forward(
        self,
        x: torch.Tensor,
        freqs_complex: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        xq = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        xq = apply_rotary_embeddings(xq, freqs_complex[:seq_len])
        xk = apply_rotary_embeddings(xk, freqs_complex[:seq_len])

        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)

        # Transpose to (batch_size, n_heads, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = self._build_causal_sliding_window_mask(seq_len, x.device, scores.dtype)
        scores = scores + mask

        attn_weights = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(attn_weights, values)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output), attn_weights


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network used in LLaMA-2/3 and Mistral:
        SwiGLU(x) = (SiLU(x W_gate) * (x W_up)) W_down
    where SiLU(z) = z * sigmoid(z).
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class LLaMAMistralBlock(nn.Module):
    """
    Single Decoder Transformer Block combining Pre-RMSNorm, Grouped-Query Attention (with RoPE + SWA),
    and SwiGLU Feed-Forward Network.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        hidden_dim: int,
        norm_eps: float = 1e-5,
        sliding_window: Optional[int] = None,
    ):
        super().__init__()
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.attention = GroupedQueryAttention(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            sliding_window=sliding_window,
        )
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.feed_forward = SwiGLUFeedForward(dim=dim, hidden_dim=hidden_dim)

    def forward(
        self, x: torch.Tensor, freqs_complex: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attention(self.attention_norm(x), freqs_complex)
        h = x + attn_out
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, attn_weights


class MiniLLaMAMistralForCausalLM(nn.Module):
    """
    Complete First-Principles LLaMA/Mistral Decoder Language Model for architectural verification.
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        hidden_dim: int = 384,
        max_seq_len: int = 128,
        sliding_window: Optional[int] = 32,
        norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [
                LLaMAMistralBlock(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    hidden_dim=hidden_dim,
                    norm_eps=norm_eps,
                    sliding_window=sliding_window,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(dim, eps=norm_eps)
        self.output = nn.Linear(dim, vocab_size, bias=False)

        freqs = precompute_theta_pos_frequencies(
            head_dim=dim // n_heads,
            max_seq_len=max_seq_len * 2,
            theta=rope_theta,
        )
        self.register_buffer("freqs_complex", freqs, persistent=False)

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.tok_embeddings(tokens)
        last_attn = None
        for layer in self.layers:
            h, last_attn = layer(h, self.freqs_complex.to(h.device))
        h = self.norm(h)
        logits = self.output(h)
        return logits, last_attn
