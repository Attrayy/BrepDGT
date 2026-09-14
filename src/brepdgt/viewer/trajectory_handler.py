from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import os
from enum import Enum
from copy import copy, deepcopy
from dataclass_wizard import YAMLWizard

import torch
import numpy as np

import polyscope as ps
import polyscope.imgui as psim

from brepdgt.models.brepdgt import BRepDGT
from brepdgt.viewer.interpolation import (
    InterpolationType,
    INTERPOLATION_TYPE_MAP,
    INTERPOLATION_TYPE_INVMAP,
)
from brepdgt.viewer.trajectory import (
    Trajectory,
)
from brepdgt.viewer.denoiser import (
    DenoiserConfig,
    ALLOWED_BATCH_SIZES,
    ALLOWED_BATCH_SIZES_INVMAP,
    ALLOWED_BATCH_SIZES_MAP,
)

# Allows to track version for backward compatibility
# 0.0.4: add batched-inference!
# 0.0.5: add interpolation parameters to config + TrajectoryType.INTERPOLATION (NB: can be
# either DDPM or DDIM)!
# 0.0.6: export subsampling stride
# 0.0.7: add i_t to distinguish stepping and UI!
# 0.0.8: add batch overrides with variable number of faces for inpainting
# 0.0.9: you can now store multiple meshes across the whole trajectory
TRAJ_VERSION: str = "0.0.9"

SLIDER_HALF_WIDTH = 100


class TrajectoryHandler:
    """
    The Trajectory Handler stores all trajectories and allows to save/replay them + export them
    """

    trajectories: Dict[int, Trajectory]

    # Maps i_traj to traj_idx (mostly for gui)
    traj_idx_map: List[int]

    def __init__(self):
        self.counter: int = -1
        self.trajectories = {}

        self.current_idx: int = -1

        # Interpolation (warning)
        self.interp_i1: int = 0
        self.interp_i2: int = 1

        self._update_traj_idx_map()

    def add(self, trajectory: Trajectory, replace_current: bool = False):
        if replace_current:
            self.trajectories[self.current_idx] = trajectory
        else:
            self.counter += 1
            self.trajectories[self.counter] = trajectory
            self.current_idx = self.counter
        self._update_traj_idx_map()

    def _get_prev_traj_idx(self, traj_idx) -> int:
        """
        If there isn't a previous one, it'll return the next one!
        """
        assert len(self.trajectories) > 1
        prev_map = {i: idx for i, idx in enumerate(self.trajectories.keys())}
        prev_invmap = {idx: i for i, idx in enumerate(self.trajectories.keys())}
        if prev_invmap[traj_idx] == 0:
            return prev_map[1]
        else:
            return prev_map[prev_invmap[traj_idx] - 1]

    def delete(self, traj_idx: int):
        if traj_idx not in self.trajectories:
            print(f"WARNING: tried to delete unknown trajectory {traj_idx:03d}!")
            return
        if len(self.trajectories) < 2:
            print(
                "WARNING: cannot remove this trajectory because there is only one trajectory!"
            )
            return

        self.current_idx = self._get_prev_traj_idx(traj_idx)
        self.trajectories.pop(traj_idx)
        self._update_traj_idx_map()

    def _update_traj_idx_map(self):
        self.traj_idx_map = sorted(list(self.trajectories.keys()))

        # Just make sure these stay valid!
        self.interp_i1 = max(min(len(self.traj_idx_map) - 1, self.interp_i1), 0)
        self.interp_i2 = max(min(len(self.traj_idx_map) - 1, self.interp_i2), 0)

    @property
    def current_trajectory(self):
        return self.trajectories[self.current_idx]

    @property
    def config(self) -> DenoiserConfig:
        return self.current_trajectory.denoiser_config

    def traj_name_selectable(
        self, label: str, idx: int, flags=psim.ImGuiSelectableFlags_None
    ) -> bool:

        clicked, _ = psim.Selectable(
            label,
            self.current_idx == idx,
            flags=flags,
        )

        if clicked and idx != self.current_idx:
            self.current_idx = idx

            return True
        return False

    def gui(self, model: BRepDGT):

        TEXT_BASE_HEIGHT = psim.GetTextLineHeightWithSpacing()
        LIST_MAX_HEIGHT = 300

        update = False

        if psim.BeginTable(
            f"Trajectories##trajectory_handler",
            5,
            psim.ImGuiTableFlags_ScrollY,
            (
                0,
                min(TEXT_BASE_HEIGHT * (len(self.trajectories) + 1.5), LIST_MAX_HEIGHT),
            ),
        ):
            psim.TableSetupColumn(
                "Name",
                psim.ImGuiTableColumnFlags_WidthStretch,
                0.0,
            )
            psim.TableSetupColumn(
                "Type",
                psim.ImGuiTableColumnFlags_WidthStretch,
                0.0,
            )
            psim.TableSetupColumn(
                "Size",
                psim.ImGuiTableColumnFlags_WidthFixed
                | psim.ImGuiTableColumnFlags_NoHide,
                0.0,
            )
            psim.TableSetupColumn(
                "Steps",
                psim.ImGuiTableColumnFlags_WidthFixed
                | psim.ImGuiTableColumnFlags_NoHide,
                0.0,
            )
            psim.TableSetupColumn(
                "Delete",
                psim.ImGuiTableColumnFlags_WidthFixed
                | psim.ImGuiTableColumnFlags_NoHide,
                0.0,
            )
            psim.TableHeadersRow()

            to_delete = -1
            for traj_idx, traj in self.trajectories.items():

                psim.TableNextRow()

                # Display Name
                psim.TableNextColumn()
                update |= self.traj_name_selectable(f"{traj_idx:03d}_traj", traj_idx)

                # Trajectory Type
                psim.TableNextColumn()
                update |= self.traj_name_selectable(
                    f"{traj.trajectory_type.value}##{traj_idx:03d}_traj", traj_idx
                )

                # Trajectory Type
                psim.TableNextColumn()
                psim.Text(f"{traj.denoiser_config.batch_size}")

                # Number of Steps
                psim.TableNextColumn()
                psim.Text(f"{traj.size}")

                # Delete
                psim.TableNextColumn()
                if psim.Button(f"X##{traj_idx:03d}_traj_trajectory_handler"):
                    to_delete = traj_idx
                    update |= True

            if to_delete >= 0:
                self.delete(to_delete)

            psim.EndTable()

        # update |= self.interpolate_gui(model=model)

        return update

    # NB: model is needed here to be able to tokenize/detokenize
    # TODO: this is kind of wobbly
    def interpolate_gui(self, model: BRepDGT) -> bool:

        uv_grids_interpolated = None

        # We can only interpolate from 2 trajectories
        if len(self.trajectories) > 1:

            psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
            if psim.TreeNode("Interpolation##trajectory_handler"):
                # Choose which trajectories to interpolate from
                psim.PushItemWidth(SLIDER_HALF_WIDTH)
                clicked, self.interp_i1 = psim.SliderInt(
                    "traj1##trajectory_handler",
                    self.interp_i1,
                    v_min=0,
                    v_max=len(self.traj_idx_map) - 1,
                    format=f"{self.traj_idx_map[self.interp_i1]}",
                )
                psim.SameLine()
                clicked, self.interp_i2 = psim.SliderInt(
                    "traj2##trajectory_handler",
                    self.interp_i2,
                    v_min=0,
                    v_max=len(self.traj_idx_map) - 1,
                    format=f"{self.traj_idx_map[self.interp_i2]}",
                )
                psim.PopItemWidth()

                # DEPRECATED
                # Choose t
                # (
                #     _,
                #     self.config.interpolation.interp_ot_match,
                # ) = psim.Checkbox(
                #     "ot_match##trajectory_handler_interp",
                #     self.config.interpolation.interp_ot_match,
                # )
                # psim.SameLine()
                # clicked, self.interp_t = psim.SliderInt(
                #     "t##trajectory_handler_interp",
                #     self.config.interpolation.interp_t,
                #     v_min=0,
                #     v_max=0,  # TODO!!!!
                # )
                # psim.SameLine()
                # clicked, interp_type_idx = psim.SliderInt(
                #     "fn##trajectory_handler_interp",
                #     INTERPOLATION_TYPE_MAP[self.config.interpolation.interp_type],
                #     v_min=0,
                #     v_max=len(InterpolationType) - 1,
                #     format=f"{self.config.interpolation.interp_type.value}",
                # )
                # if clicked:
                #     self.config.interpolation.interp_type = INTERPOLATION_TYPE_INVMAP[
                #         interp_type_idx
                #     ]
                psim.SetNextItemWidth(250)
                clicked, batch_size_idx = psim.SliderInt(
                    "interpolation points##trajectory_handler_interp",
                    ALLOWED_BATCH_SIZES_MAP[
                        self.config.interpolation.interp_batch_size
                    ],
                    v_min=0,
                    v_max=len(ALLOWED_BATCH_SIZES) - 1,
                    format=f"{self.config.interpolation.interp_batch_size}",
                )
                if clicked:
                    self.config.interpolation.interp_batch_size = (
                        ALLOWED_BATCH_SIZES_INVMAP[batch_size_idx]
                    )

                psim.BeginDisabled(self.interp_i1 == self.interp_i2)
                if psim.Button("Interpolate##trajectory_handler_interp"):
                    uv_grids_interpolated = Trajectory.interpolate(
                        traj1=self.trajectories[self.traj_idx_map[self.interp_i1]],
                        traj2=self.trajectories[self.traj_idx_map[self.interp_i2]],
                        batch_size=self.config.interpolation.interp_batch_size,
                        t=self.config.interpolation.interp_t,
                        model=model,
                        interpolation_type=self.config.interpolation.interp_type,
                        ot_match=self.config.interpolation.interp_ot_match,
                    )

                psim.EndDisabled()

                psim.TreePop()

        return uv_grids_interpolated

    def serialize(self):
        traj_dict = {}
        for traj_idx, traj in self.trajectories.items():
            traj_dict[traj_idx] = traj.serialize()

        data = {"trajectories": traj_dict, "version": TRAJ_VERSION}

        return data

    def deserialize(self, data: Dict[str, Any], model: BRepDGT):
        for traj_data in data["trajectories"].values():
            trajectory = Trajectory.deserialize(traj_data, model=model)
            self.add(trajectory)

    def save(self, cpath: str):
        import deepdish as dd

        data = self.serialize()
        # cpath = self.cpath if cpath is None else cpath
        # os.makedirs(os.path.abspath(os.path.dirname(cpath)), exist_ok=True)
        dd.io.save(cpath, data)

    def clear(self):
        to_delete = self.traj_idx_map[1:]
        for i_traj in to_delete:
            self.delete(i_traj)
