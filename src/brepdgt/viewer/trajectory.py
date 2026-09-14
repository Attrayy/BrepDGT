from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from copy import deepcopy

import torch
import numpy as np

from brepdgt.primitives.uvgrid import (
    UvGrid,
    concat_uvgrids,
    uncat_uvgrids,
)
from brepdgt.models.tokenizers.base import Tokens
from brepdgt.models.detokenizers.pre_mask_uv_detokenizer_3d import DedupDetokenizer3D
from brepdgt.models.brepdgt import BRepDGT
from brepdgt.viewer.interpolation import (
    InterpolationType,
    INTERPOLATION_TYPE_FN,
    ot_match_uvgrid,
    get_best_matching,
    extract_best_matching,
)
from brepdgt.viewer.denoiser_config import DenoiserConfig


class TrajectoryType(Enum):
    FORWARD: str = "forward"  # Noising
    BACKWARD: str = "backward"  # Denoising
    INPAINTING: str = "inpainting"  # Inpainting
    INTERPOLATING: str = "interpolating"  # Interpolating


@dataclass
class Trajectory:
    """
    A trajectory stores all the Tokens/UvGrid/Config for both Forward and Backward trajectories.
    This allows to store all parameters and potentially export trajectories for reproducibility.
    """

    # Stores base tokens used for inference (across all ts)
    base_tokens: Tokens

    # Frozen Primitives
    frozen_prims: List[bool]

    # Indexed by t (as done for sampling)
    traj_uv_grids: Dict[int, UvGrid]

    # Config file to track parameters used during sampling
    denoiser_config: DenoiserConfig

    # Stores current `i_t` = different from i_t because i_t directly index into the set of
    # possible  whereas i_step is used to keep track of denoising steps! NB: this means
    # that the "renoiser" will be in charge of manually incrementing i_t to display the new
    # state but this should be much safer!
    i_t: int
    i_step: int

    # Stores current slider `i_batch`
    i_batch: int

    # Keymap mapping i_t -> t (it's actually a list)
    keymap: List[int]
    inv_keymap: Dict[int, int]

    # Trajectory Type
    trajectory_type: TrajectoryType

    # (Optional) Reconstructed Brep
    brep_mesh_vertices: List[Optional[np.ndarray]]
    brep_mesh_faces: List[Optional[np.ndarray]]
    brep_solid: List[Optional[Any]]  # Not serialized!

    def _is_exportable_t(self, t: int):
        """Specifies whether t can be exported or not! First and last grid are always exported!"""
        return (
            t == 0
            or t == 1000
            or t == 999
            or t == max(self.keymap)
            or t % self.denoiser_config.export_stride == 0
        )

    @property
    def batch_size(self):
        return next(iter(self.traj_uv_grids.values())).coord.shape[0]

    @torch.no_grad()
    def serialize(self) -> Dict[str, Any]:
        data = {}
        data["frozen_prims"] = self.frozen_prims
        data["denoiser_config"] = self.denoiser_config.to_yaml()
        data["i_step"] = self.i_step
        data["i_t"] = self.i_t
        data["keymap"] = [t for t in self.keymap if self._is_exportable_t(t)]
        data["trajectory_type"] = self.trajectory_type
        # Concatenate all the uv_grids (keymap will be used to recover them later on)
        # NB: we need to sort here to map properly to the keymap above!
        all_uv_grids = [
            v for t, v in sorted(self.traj_uv_grids.items()) if self._is_exportable_t(t)
        ]
        all_uv_grids = concat_uvgrids(
            all_uv_grids, cat_dim=0
        )  # concat over batch dimension!
        data["traj_uv_grids"] = all_uv_grids.serialize()

        # Brep reconstruction
        if self.brep_mesh_vertices is not None and self.brep_mesh_faces is not None:
            data["brep_mesh_vertices"] = self.brep_mesh_vertices
            data["brep_mesh_faces"] = self.brep_mesh_faces

        return data

    @staticmethod
    def deserialize(data: Dict[str, Any], model: BRepDGT) -> Trajectory:
        keymap = data["keymap"]
        frozen_prims = data["frozen_prims"]
        i_step = data["i_step"]
        # Worse case we show the first step
        i_t = min(max(data["i_t"], 0), len(keymap) - 1) if "i_t" in data else 0
        denoiser_config = DenoiserConfig.from_yaml(data["denoiser_config"])
        uvgrids = uncat_uvgrids(
            UvGrid.deserialize(data["traj_uv_grids"]),
            cat_dim=0,
            stride=denoiser_config.batch_size,
        )
        uvgrids = {k: v for k, v in zip(keymap, uvgrids)}
        for v in uvgrids.values():
            v.to_tensor("cuda")

        # First tokenize:
        tmp_uv_grids: UvGrid = next(iter(uvgrids.values()))
        x = model.token_vae.encode(None, uvgrid=tmp_uv_grids).sample
        max_n_prims = len(frozen_prims)
        base_tokens = get_base_tokens(
            x, max_n_prims=max_n_prims, empty_mask=tmp_uv_grids.empty_mask
        )

        trajectory = Trajectory(
            t=keymap[i_t],
            base_tokens=base_tokens,
            uv_grids=uvgrids,
            frozen_prims=frozen_prims,
            denoiser_config=denoiser_config,
            i_step=i_step,
            trajectory_type=data["trajectory_type"],
            i_t=i_t,
        )

        if "brep_mesh_vertices" in data and "brep_mesh_faces" in data:

            if isinstance(data["brep_mesh_vertices"], list) and isinstance(
                data["brep_mesh_faces"], list
            ):
                trajectory.brep_mesh_vertices = data["brep_mesh_vertices"]
                trajectory.brep_mesh_faces = data["brep_mesh_faces"]
            # DEPRECATED but implemented for backward compatibility
            elif isinstance(data["brep_mesh_vertices"], np.ndarray) and isinstance(
                data["brep_mesh_faces"], np.ndarray
            ):
                trajectory.brep_mesh_vertices = [data["brep_mesh_vertices"]] + [
                    None
                ] * (tmp_uv_grids.coord.shape[0] - 1)
                trajectory.brep_mesh_faces = [data["brep_mesh_faces"]] + [None] * (
                    tmp_uv_grids.coord.shape[0] - 1
                )

        return trajectory

    def _update_key_map(self):
        self.keymap = sorted(list(self.traj_uv_grids.keys()))
        self.inv_keymap = {t: i_t for i_t, t in enumerate(self.keymap)}

    def __init__(
        self,
        t: int,
        base_tokens: Tokens,
        uv_grids: UvGrid,
        frozen_prims: List[bool],
        denoiser_config: DenoiserConfig,
        i_step: int,
        trajectory_type: TrajectoryType = TrajectoryType.BACKWARD,
        i_t: int = 0,
        i_batch: int = 0,
    ) -> None:
        # Set Trajectory Type
        self.trajectory_type = trajectory_type

        # Set Frozen Primitives
        self.frozen_prims = frozen_prims

        # Set i_step
        self.i_step = i_step

        # Set i_t
        self.i_t = i_t

        # Set a copy of the denoiser config
        self.denoiser_config = deepcopy(denoiser_config)

        # Store values
        self.base_tokens = base_tokens
        self.traj_uv_grids = uv_grids if isinstance(uv_grids, dict) else {t: uv_grids}

        # Debug: this is for sanity
        self.prev_t = t

        # Brep null initialization
        batch_size = next(iter(self.traj_uv_grids.values())).coord.shape[0]
        self.brep_mesh_vertices = [None] * batch_size
        self.brep_mesh_faces = [None] * batch_size
        self.brep_solid = [None] * batch_size

        # Set i_batch
        self.i_batch = min(batch_size, i_batch)

        self._update_key_map()

    def add(self, t: int, uv_grids: UvGrid):
        # Debug: this is for sanity
        if t in self.traj_uv_grids:
            print(
                f"WARNING: you are currently overwriting a trajectory at timestep {t}"
            )
        if (self.trajectory_type == TrajectoryType.FORWARD and t < self.prev_t) or (
            self.trajectory_type == TrajectoryType.BACKWARD and t > self.prev_t
        ):
            print(
                f"WARNING: trajectory is of type '{self.trajectory_type.value}' and prev_t={self.prev_t};t={t}"
            )
        self.prev_t = t

        # Store values
        self.traj_uv_grids[t] = uv_grids

        self._update_key_map()

    def has(self, t: int):
        test = t in self.traj_uv_grids
        if not test:
            print(f"WARNING: {t} if not in the current trajectory!")
        return test

    @property
    def size(self):
        return len(self.traj_uv_grids)

    @property
    def traj_ts(self):
        return self.keymap

    @staticmethod
    def interpolate(
        traj1: Trajectory,
        traj2: Trajectory,
        batch_size: int,
        t: int,
        model: BRepDGT,
        interpolation_type: InterpolationType = InterpolationType.LERP,
        ot_match: bool = False,
        only_coord: bool = False,
    ) -> UvGrid:
        if not (t in traj1.inv_keymap and t in traj2.inv_keymap):
            raise ValueError(f"t={t} must be in both trajectory")
        # if lbda > 1.0 or lbda < 0.0:
        #     raise ValueError(f"lbda={lbda} must be within [0, 1]")
        lbda = torch.linspace(0, 1, batch_size, device="cuda")[
            :,
            None,
            None,
        ]

        def get_ot_ts(step_size: int = 100):
            return np.arange(
                0, model.config.diffusion.training_timesteps - step_size + 1, step_size
            ).tolist() + [model.config.diffusion.training_timesteps - 1]

        # NB: extract the selected uv-grid for each trajectory!
        # NB: no need to clone, this is automatic!
        uv_grids1 = traj1.traj_uv_grids[t].extract(traj1.i_batch)
        uv_grids2 = traj2.traj_uv_grids[t].extract(traj2.i_batch)

        # Optional, perform OT
        if ot_match:
            # Uv grids used to compute distances
            dist_uv_grids1 = [
                traj1.traj_uv_grids[i].extract(traj1.i_batch)
                for i in get_ot_ts()
                if i in traj1.traj_uv_grids
            ]
            dist_uv_grids2 = [
                traj2.traj_uv_grids[i].extract(traj2.i_batch)
                for i in get_ot_ts()
                if i in traj2.traj_uv_grids
            ]

            i, j = get_best_matching(
                dist_uv_grids1,
                dist_uv_grids2,
            )
            uv_grids1, uv_grids2 = extract_best_matching(uv_grids1, uv_grids2, i, j)

        # Combine masks
        # Shouldn't be necessary (when OT is enabled!)
        empty_mask1 = uv_grids1.empty_mask[traj1.i_batch : traj1.i_batch + 1].repeat(
            (batch_size, 1)
        )
        empty_mask2 = uv_grids2.empty_mask[traj2.i_batch : traj2.i_batch + 1].repeat(
            (batch_size, 1)
        )
        empty_mask_interpolated = empty_mask1 & empty_mask2

        # Repeat by the batch_size
        uv_grids1 = uv_grids1.repeat(batch_size)
        uv_grids2 = uv_grids2.repeat(batch_size)

        # DEBUG
        # x_tokenized1 = model.token_vae.encode(None, uv_grids1).sample
        # tokens1 = Tokens(
        #     sample=x_tokenized1,
        #     mask=base_tokens.mask,
        #     condition=base_tokens.condition,
        # )
        # uv_grids_detokenized1 = model.token_vae.detokenizer.decode(
        #     tokens1, model.config.max_n_prims
        # )
        # # breakpoint()

        if only_coord:
            assert False
            lbda = lbda[..., None, None]
            # Interpolate only coordinates!
            uv_grids_interpolated = uv_grids1.clone()
            uv_grids_interpolated.coord = INTERPOLATION_TYPE_FN[interpolation_type](
                uv_grids1.coord, uv_grids2.coord, lbda
            )
            uv_grids_interpolated.empty_mask = empty_mask_interpolated
            x_interpolated = model.token_vae.encode(None, uv_grids_interpolated).sample
        else:
            # First, we need to tokenize both
            x_tokenized1 = model.token_vae.encode(None, uv_grids1).sample.cuda()
            x_tokenized2 = model.token_vae.encode(None, uv_grids2).sample.cuda()

            # Interpolate
            x_interpolated = INTERPOLATION_TYPE_FN[interpolation_type](
                x_tokenized1, x_tokenized2, lbda
            )

        tokens_interpolated = Tokens(
            sample=x_interpolated,
            mask=empty_mask_interpolated.cuda(),
            condition=get_condition(empty_mask_interpolated, model.config.max_n_prims),
        )

        # Detokenize again
        if isinstance(model.token_vae.detokenizer, DedupDetokenizer3D):
            uv_grids_interpolated = model.token_vae.detokenizer.decode(
                tokens_interpolated, model.config.max_n_prims, deduplicate=False
            )
        else:
            uv_grids_interpolated = model.token_vae.detokenizer.decode(
                tokens_interpolated, model.config.max_n_prims
            )

        return uv_grids_interpolated


def get_empty_mask(batch_size: int, n_prims: int, max_n_prims: int):
    empty_mask = torch.zeros((batch_size, max_n_prims), device="cuda", dtype=torch.bool)
    empty_mask[:, n_prims:] = True
    return empty_mask


def get_condition(empty_mask: torch.Tensor, max_n_prims: int) -> torch.Tensor:
    batch_size = empty_mask.shape[0]
    # Create one-hot condition vector for number of faces
    condition = torch.zeros(
        batch_size, max_n_prims, device="cuda"
    )  # [batch_size, max_n_prims]
    condition[:, (~empty_mask).sum(1) - 1] = 1.0


def get_base_tokens(
    x: torch.Tensor,
    max_n_prims: int,
    empty_mask,
):
    condition = get_condition(empty_mask, max_n_prims)

    return Tokens(
        sample=x, mask=empty_mask.clone(), condition=condition, center_coord=None
    )
