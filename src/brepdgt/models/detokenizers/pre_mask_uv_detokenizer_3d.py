from typing import Union, List
import torch
import torch.nn as nn
from brepdgt.models.tokenizers import Tokens
from brepdgt.utils.acc_logger import AccLogger
from brepdgt.config import Config
from brepdgt.models.detokenizers.base import DetokenizerOutput, Detokenizer
from brepdgt.primitives.uvgrid import UvGrid, uv_coord2normal
from brepdgt.utils.geometry import (
    compute_normals_from_grids,
    adjust_normals_direction,
)


class PreMaskUVDetokenizer3D(Detokenizer):
    name = "pre_mask_uv_detokenizer_3d"

    def __init__(self, config: Config, acc_logger: AccLogger):
        super().__init__(config, acc_logger)
        # trainable parameters are required for torch lightning
        self.dummy = nn.Linear(1, 1)
        self.per_points_dim = self.config.data_dim
        self.per_points_dim += 1  # grid mask
        self.sanity_check_x_dim()

    def sanity_check_x_dim(self):
        assert (
            self.config.x_dim
            == self.config.n_grid * self.config.n_grid * self.per_points_dim
        ), f"x_dim: {self.config.x_dim}, n_grid: {self.config.n_grid}, uvgrid_dim: {self.per_points_dim}"

    def forward(self, tokens: Tokens) -> DetokenizerOutput:
        batch_size, n_prims, head_dim = tokens.sample.shape
        token_sample = tokens.sample

        empty_mask = tokens.mask
        uvgrids = token_sample.view(
            batch_size,
            n_prims,
            self.config.n_grid,
            self.config.n_grid,
            self.per_points_dim,
        )
        # update the last idx
        end_idx = self.config.data_dim
        coords = uvgrids[:, :, :, :, :end_idx]
        grid_mask = uvgrids[:, :, :, :, end_idx : end_idx + 1]
        # map to B x n_prims x n_grid x n_grid
        grid_mask = grid_mask.squeeze(dim=-1) > 0  # values should range [-1, 1]
        end_idx += 1

        # flat n_prims and batch to batch_dim and reshape afterwards
        normals = uv_coord2normal(coords.reshape(-1, *coords.shape[2:])).view(
            *coords.shape
        )

        uvgrids = UvGrid(
            coord=coords, empty_mask=empty_mask, normal=normals, grid_mask=grid_mask
        )
        return DetokenizerOutput(type_logits=None, uvgrid=uvgrids)

    def decode(
        self,
        tokens: Tokens,
        max_n_prims: int,
    ) -> UvGrid:
        """
        Generate zone graphs from tokens and auxiliary arguments
        :param tokens: Tokens
        :param max_n_prims: maximum number of slicers
        :return:
            uvgrids: uvgrid
        """
        out: DetokenizerOutput = self(tokens)
        return out.uvgrid


class DedupDetokenizer3D(PreMaskUVDetokenizer3D):
    name = "dedup_uv_detokenizer_3d"
    """
    Uses deduplication to deduplicate duplicated masks.
    Does not use empty mask
    """

    def decode(
        self,
        tokens: Tokens,
        max_n_prims: int,
        deduplicate: bool = True,
    ) -> UvGrid:
        uvgrid = self(tokens).uvgrid
        if not deduplicate:
            return uvgrid

        feat = uvgrid.coord

        empty_mask = self.get_empty_mask_by_removing_duplicates(
            feat,
            self.config.tokenizer_dedup_thresh,
            self.config.tokenizer_dedup_rel_thresh,
            empty_mask=uvgrid.empty_mask,
        )
        uvgrid.empty_mask = empty_mask
        return uvgrid
