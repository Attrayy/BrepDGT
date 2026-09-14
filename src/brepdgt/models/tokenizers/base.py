from __future__ import annotations
from typing import Tuple, Dict, Any, Union
from dataclasses import dataclass, field

from brepdgt.models.base_model import BaseModule
from brepdgt.config import Config
from brepdgt.utils.acc_logger import AccLogger
from brepdgt.primitives.uvgrid import UvGrid
import torch


# Don't know why but torch.compile can't handle dataclass
# @dataclass(frozen=True)
# class Tokens:
#     """
#     Used as output of the tokenizer and input to the detokenizer
#     """
#
#     sample: torch.Tensor  # Tensor of B x n_prims x x_dim, only continuous sample
#     condition: torch.Tensor = None  # Tensor of B x n_prims x z_dim
#     mask: torch.Tensor = None  # Tensor of B x n_prims
#     mean: torch.Tensor = None  # Tensor of B x n_prims x x_dim
#     log_var: torch.Tensor = None  # Tensor of B x n_prims x x_dim
#     labels: Union[torch.Tensor, None] = None  # Tensor of B x n_prims
#     aux_outputs: Dict = field(default_factory=dict)
#     center_coord: torch.Tensor = (
#         None  # Tensor of B x n_prims x 3, used for flow matching
#     )

KEYS_TO_CLONE = {
    "sample",
    "condition",
    "mask",
    "mean",
    "log_var",
    "labels",
    "center_coord",
}


class Tokens:
    """
    Used as output of the tokenizer and input to the detokenizer
    """

    def __init__(
        self,
        sample: torch.Tensor,
        condition: torch.Tensor = None,
        mask: torch.Tensor = None,
        mean: torch.Tensor = None,
        log_var: torch.Tensor = None,
        labels: Union[torch.Tensor, None] = None,  # Tensor of B x n_prims
        aux_outputs: Dict = None,
        center_coord: torch.Tensor = None,  # Tensor of B x n_prims x 3, used for flow matching
    ):
        self.sample = sample
        self.condition = condition
        self.mask = mask
        self.mean = mean
        self.log_var = log_var
        self.labels = labels
        self.aux_outputs = aux_outputs
        self.center_coord = center_coord

    def clone(self) -> Tokens:
        data = {}
        for k in KEYS_TO_CLONE:
            if hasattr(self, k) and getattr(self, k) is not None:
                data[k] = getattr(self, k).clone()

        return Tokens(**data)


class Tokenizer(BaseModule):
    def __init__(self, config: Config, acc_loger: AccLogger):
        super().__init__(config, acc_loger)

    def forward(self, labels: torch.Tensor, uvgrids: UvGrid) -> Tokens:
        raise NotImplementedError()

    @staticmethod
    def reparameterize(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        # reparameterization trick
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon
        return z

    def reg_loss(
        self, tokens: Tokens, prefix: str
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # apply kld loss
        mean, log_var = tokens.mean, tokens.log_var
        loss_reg = -0.5 * torch.mean(
            (1 + log_var - mean.pow(2) - log_var.exp()), dim=2
        ).mean(
            1
        )  # B
        assert len(loss_reg.shape) == 1
        log_dict = {
            f"loss/{prefix}/kld": loss_reg.tolist(),
            f"token/var_{prefix}": log_var.exp().mean(2).mean(1).tolist(),
            f"token/mean_norm_{prefix}": mean.norm(dim=2).mean(1).tolist(),
        }
        return loss_reg, log_dict

    def sample_empty_embeddings(self, shape) -> Union[torch.Tensor, None]:
        return None
