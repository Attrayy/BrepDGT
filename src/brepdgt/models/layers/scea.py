from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class SCEA(nn.Module):
    """Shared Cross-head Efficient Attention.

    SCEA keeps one query projection per attention head while sharing keys and
    values within head groups. After the row-wise softmax, a learnable matrix
    mixes attention maps across query heads. This preserves cross-head
    communication while reducing K/V projection parameters and computation.

    Args:
        dim: Token feature dimension.
        num_heads: Number of query and output heads.
        kv_heads: Number of shared key/value heads. By default, one K/V head is
            shared by every pair of query heads.
        dropout: Dropout probability applied to attention probabilities.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kv_heads: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )

        kv_heads = max(1, num_heads // 2) if kv_heads is None else kv_heads
        if kv_heads <= 0 or num_heads % kv_heads != 0:
            raise ValueError(
                "kv_heads must be positive and divide num_heads exactly; "
                f"got kv_heads={kv_heads}, num_heads={num_heads}"
            )

        self.dim = dim
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.head_dim = dim // num_heads
        self.groups = num_heads // kv_heads
        self.dropout = float(dropout)

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

        # This key name keeps trained BRepDGT checkpoints directly loadable.
        self.w_post = nn.Parameter(torch.eye(num_heads))

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, length, channels = x.shape
        if channels != self.dim:
            raise ValueError(f"expected input dim {self.dim}, got {channels}")

        q = self.q_proj(x).view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(x).view(
            batch, length, self.kv_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            batch, length, self.kv_heads, self.head_dim
        ).transpose(1, 2)

        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.head_dim**-0.5
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch, length):
                raise ValueError(
                    "key_padding_mask must have shape "
                    f"({batch}, {length}), got {tuple(key_padding_mask.shape)}"
                )
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :].bool(), float("-inf")
            )

        attention = torch.softmax(scores, dim=-1)
        attention = F.dropout(attention, p=self.dropout, training=self.training)

        # out_head[g] = sum_h w_post[h, g] * attention_head[h]
        attention = torch.einsum("bhnm,hg->bgnm", attention, self.w_post)
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.out_proj(output)
