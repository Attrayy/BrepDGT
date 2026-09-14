import torch
from torch import nn


class DynamicTanh(nn.Module):
    """Dynamic Tanh (DyT) feature transformation.

    DyT replaces token-wise normalization with a learned, bounded transform:

        y = gamma * tanh(alpha * x) + beta

    Alpha is shared across channels, while gamma and beta are learned
    independently for every feature channel.
    """

    def __init__(self, dim: int, init_alpha: float = 0.5):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * torch.tanh(self.alpha * x) + self.beta
