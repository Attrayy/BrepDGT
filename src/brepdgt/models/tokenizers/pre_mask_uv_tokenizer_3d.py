import torch
from brepdgt.models.tokenizers.base import Tokens, Tokenizer
from brepdgt.primitives.uvgrid import UvGrid
import torch.nn.functional as F


class PreMaskUvTokenizer3D(Tokenizer):
    name = "pre_mask_uv_tokenizer_3d"
    """
    Append an empty mask to the token
    """

    def forward(self, labels: torch.Tensor, uvgrids: UvGrid) -> Tokens:
        """
        :param labels: Tensor of B x max_n_prims
            one hot encoding of slicers
        :param uvgrids: UVGrid of coordinates having B x max_n_prims x n_grid x n_grid x data_dim
            could also have normals
        :return:
        """
        batch_size, max_n_prims, n_grid, _, _ = uvgrids.coord.shape
        # note that 1 indicates empty and -1 indicates not empty
        uvgrid_empty_mask = (2 * uvgrids.empty_mask - 1) > 0  # B x n_prims x 1

        # tensorize
        uvgrid_tensor = [uvgrids.coord]
        # B x n_prims x n_grid x n_grid x 1
        grid_mask = 2 * uvgrids.grid_mask.unsqueeze(-1).float() - 1
        uvgrid_tensor.append(grid_mask)
        uvgrid_tensor = torch.cat(uvgrid_tensor, dim=-1)
        uvgrid_tensor = uvgrid_tensor.view(batch_size, max_n_prims, -1)

        condition = None
        empty_mask = uvgrids.empty_mask
        if self.config.diffusion.z_conditioning:
            num_prims_condition = F.one_hot(
                self.config.max_n_prims - empty_mask.sum(dim=-1) - 1,
                num_classes=self.config.max_n_prims,
            )
            condition = num_prims_condition.unsqueeze(dim=1).float()

        # center_coord used for flow matching
        center_coord = torch.stack(
            [
                uvgrids.coord[:, :, 0, 0],
                uvgrids.coord[:, :, 0, n_grid - 1],
                uvgrids.coord[:, :, n_grid - 1, 0],
                uvgrids.coord[:, :, n_grid - 1, n_grid - 1],
            ],
            dim=-1,
        ).mean(dim=-1)

        tokens = Tokens(
            sample=uvgrid_tensor,
            mask=uvgrid_empty_mask,
            condition=condition,
            center_coord=center_coord,
        )
        return tokens
