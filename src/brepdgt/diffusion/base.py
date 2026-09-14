from dataclasses import dataclass
from typing import Tuple, Dict, List, Union
import torch
from torch import nn
from brepdgt.config import Config
from brepdgt.models.tokenizers import Tokens


@dataclass(frozen=True)
class DiffusionSample:
    x: torch.Tensor
    l: torch.Tensor

    def get_x_for_detokenizer(self) -> torch.Tensor:
        raise NotImplementedError()


class Diffusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        tokens: Tokens,
        z: torch.Tensor,
        mask: torch.Tensor,
        prefix: str,
        empty_embeddings: Union[torch.Tensor, None],
        return_timesteps: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        raise NotImplementedError()

    @torch.no_grad()
    def p_sample_loop(
        self,
        batch_size: int,
        return_traj: bool = False,
        z: Union[torch.Tensor, None] = None,
        cfg_scale: float = 1.0,
        traj_stride: int = 100,
    ) -> Tuple[DiffusionSample, List[DiffusionSample]]:
        raise NotImplementedError()
