from enum import Enum
from dataclasses import dataclass
from typing import Tuple, List
from brepdgt.primitives.uvgrid import UvGrid

import torch
import numpy as np


def slerp(z1: torch.Tensor, z2: torch.Tensor, alpha: float):
    theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
    return (
        torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
        + torch.sin(alpha * theta) / torch.sin(theta) * z2
    )


def lerp(x1: torch.Tensor, x2: torch.Tensor, alpha: float):
    return (1 - alpha) * x1 + alpha * x2


class InterpolationType(Enum):
    LERP: str = "lerp"
    SLERP: str = "slerp"


INTERPOLATION_TYPE_MAP = {x: i for i, x in enumerate(InterpolationType)}
INTERPOLATION_TYPE_INVMAP = {i: x for i, x in enumerate(InterpolationType)}

INTERPOLATION_TYPE_FN = {InterpolationType.LERP: lerp, InterpolationType.SLERP: slerp}


@dataclass
class InterpolationConfig:
    interp_t: int = 999
    interp_type: InterpolationType = InterpolationType.LERP
    interp_ot_match: bool = True
    interp_batch_size: int = 32


def get_optimal_idxs(x0: torch.Tensor, x1: torch.Tensor, replace: bool = False):
    r"""Compute the OT plan $\pi$ (wrt squared Euclidean cost) between a source and a target
    minibatch and draw source and target samples from pi $(x,z) \sim \pi$

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the source minibatch
    replace : bool
        represents sampling or without replacement from the OT plan

    Returns
    -------
    i : Tensor, shape (bs, *dim)
        represents the remapped indices of x0 drawn from $\pi$
    j : Tensor, shape (bs, *dim)
        represents the remapped indices of x1 drawn from $\pi$
    """
    from torchcfm import OTPlanSampler

    ot_sampler = OTPlanSampler(method="exact")
    pi = ot_sampler.get_map(x0, x1)
    i, j = ot_sampler.sample_map(pi, x0.shape[0], replace=replace)
    return i, j


def get_best_matching(
    uvgrids0: List[UvGrid], uvgrids1: List[UvGrid]
) -> Tuple[int, int]:
    # Given two list of uv grids (over time), compute the best matching/assignment
    # NB: this assumes empty_mask is constant within the list!

    assert len(uvgrids0) == len(uvgrids1)

    # Get non_empty_masks i.e., which face to compare to which one
    non_zeros = ~uvgrids0[0].empty_mask[0] | ~uvgrids1[0].empty_mask[0]
    n_faces = torch.count_nonzero(non_zeros).item()

    device = uvgrids0[0].coord.device
    non_zeros = non_zeros.to(device)

    # Standard strategy
    all_x0s = []
    all_x1s = []
    for uvgrid0, uvgrid1 in zip(uvgrids0, uvgrids1):
        all_x0s.append(uvgrid0.coord.cuda()[0, non_zeros].view(n_faces, -1))
        all_x1s.append(uvgrid1.coord.cuda()[0, non_zeros].view(n_faces, -1))
    all_x0s = torch.cat(all_x0s, dim=1)
    all_x1s = torch.cat(all_x1s, dim=1)

    i, j = get_optimal_idxs(all_x0s, all_x1s, replace=False)
    non_zeros_indices = torch.argwhere(non_zeros).squeeze(1)
    return non_zeros_indices[i], non_zeros_indices[j]


def extract_best_matching(
    uvgrid0: UvGrid, uvgrid1: UvGrid, i: torch.Tensor, j: torch.Tensor
) -> Tuple[UvGrid, UvGrid]:

    assert len(i) == len(j)

    n_faces = len(i)

    uvgrid0_new = uvgrid0.clone()
    uvgrid0_new.to_tensor("cuda")
    uvgrid0_new.coord[0, :n_faces] = uvgrid0_new.coord[0, i]
    uvgrid0_new.empty_mask[0, :n_faces] = False
    if uvgrid0_new.normal is not None:
        uvgrid0_new.normal[0, :n_faces] = uvgrid0_new.normal[0, i]
    if uvgrid0_new.grid_mask is not None:
        uvgrid0_new.grid_mask[0, :n_faces] = uvgrid0_new.grid_mask[0, i]
    if uvgrid0_new.prim_type is not None:
        uvgrid0_new.prim_type[:0, n_faces] = uvgrid0_new.prim_type[0, i]

    uvgrid1_new = uvgrid1.clone()
    uvgrid1_new.to_tensor("cuda")
    uvgrid1_new.coord[0, :n_faces] = uvgrid1_new.coord[0, j]
    uvgrid1_new.empty_mask[0, :n_faces] = False
    if uvgrid1_new.normal is not None:
        uvgrid1_new.normal[0, :n_faces] = uvgrid1_new.normal[0, j]
    if uvgrid1_new.grid_mask is not None:
        uvgrid1_new.grid_mask[0, :n_faces] = uvgrid1_new.grid_mask[0, j]
    if uvgrid1_new.prim_type is not None:
        uvgrid1_new.prim_type[0, :n_faces] = uvgrid1_new.prim_type[0, j]

    return uvgrid0_new, uvgrid1_new


def ot_match_uvgrid(uvgrid0: UvGrid, uvgrid1: UvGrid) -> Tuple[UvGrid, UvGrid]:
    """
    :param uvgrid0: coord with shape n_seq x n_grid x n_grid x 3
    :param uvgrid1: coord with shape n_seq x n_grid x n_grid x 3
    :return:
        Uv idx reordered uvgrids using ot
    """
    assert uvgrid0.coord.shape[0] == 1 and uvgrid1.coord.shape[0] == 1
    # TODO: warning this isn't always valid!
    n_faces = max(torch.sum(~uvgrid0.empty_mask), torch.sum(~uvgrid1.empty_mask))

    # Convert uvgrid to cost vectors and obtain indices
    x0 = uvgrid0.coord[0, :n_faces].view(n_faces, -1)
    x1 = uvgrid1.coord[0, :n_faces].view(n_faces, -1)
    # NB: this needs to be done without replacement! (otherwise, we'll lose faces)
    i, j = get_optimal_idxs(x0, x1, replace=False)

    # remap uvgrid order inplace
    uvgrid0_new = uvgrid0.clone()
    uvgrid0_new.coord[0, :n_faces] = uvgrid0_new.coord[0, i]
    uvgrid0_new.empty_mask[0, :n_faces] = False
    if uvgrid0_new.normal is not None:
        uvgrid0_new.normal[0, :n_faces] = uvgrid0_new.normal[0, i]
    if uvgrid0_new.grid_mask is not None:
        uvgrid0_new.grid_mask[0, :n_faces] = uvgrid0_new.grid_mask[0, i]
    if uvgrid0_new.prim_type is not None:
        uvgrid0_new.prim_type[:0, n_faces] = uvgrid0_new.prim_type[0, i]

    uvgrid1_new = uvgrid1.clone()
    uvgrid1_new.coord[0, :n_faces] = uvgrid1_new.coord[0, j]
    uvgrid1_new.empty_mask[0, :n_faces] = False
    if uvgrid1_new.normal is not None:
        uvgrid1_new.normal[0, :n_faces] = uvgrid1_new.normal[0, j]
    if uvgrid1_new.grid_mask is not None:
        uvgrid1_new.grid_mask[0, :n_faces] = uvgrid1_new.grid_mask[0, j]
    if uvgrid1_new.prim_type is not None:
        uvgrid1_new.prim_type[0, :n_faces] = uvgrid1_new.prim_type[0, j]

    return uvgrid0_new, uvgrid1_new
