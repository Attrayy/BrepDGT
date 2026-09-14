import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import os
from typing import List, Dict, Any, Tuple, Union
from PIL import Image, ImageDraw
from dataclasses import dataclass
from tqdm import tqdm

from brepdgt.models.base_model import BaseModel
from brepdgt.models.tokenizers import (
    Tokenizer,
    TOKENIZERS,
    Tokens,
)
from brepdgt.models.detokenizers import (
    Detokenizer,
    DETOKENIZERS,
    DetokenizerOutput,
)
from brepdgt.config import Config
from brepdgt.utils.vis import concat_h_pil, concat_v_pil
from brepdgt.primitives.uvgrid import UvGrid
from brepdgt.datasets.abc_dataset import ABCDatasetOutput
from brepdgt.utils.vis import save_and_render_uvgrid


@dataclass(frozen=True)
class UvVaeOutput:
    type_logits: torch.Tensor
    uvgrid_preds: UvGrid
    tokens: Tokens


class UvVae(BaseModel):
    name = "uv_vae"
    """
    Wrapper around the tokenizer and detokenizer
    """

    def __init__(self, config: Config, acc_logger):
        super().__init__(config, acc_logger)
        self.tokenizer: Tokenizer = TOKENIZERS[self.config.tokenizer.name](
            self.config, acc_logger
        )
        self.detokenizer: Detokenizer = DETOKENIZERS[self.config.detokenizer.name](
            self.config, acc_logger
        )
        self.params_error = None

    def forward(self, batch) -> UvVaeOutput:
        """
        :param labels: Tensor of B x n_slicers
        :param uvgrids: UVGrid
        :return:

        """
        labels = batch.uvgrid.prim_type
        uvgrids = batch.uvgrid
        tokens = self.tokenizer(labels, uvgrids)
        out: DetokenizerOutput = self.detokenizer(tokens)
        return UvVaeOutput(
            type_logits=out.type_logits,
            uvgrid_preds=out.uvgrid,
            tokens=tokens,
        )

    def encode(self, labels: torch.Tensor, uvgrid: UvGrid) -> Tokens:
        tokens = self.tokenizer(labels, uvgrid)
        return tokens
