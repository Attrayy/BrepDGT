from typing import List, Dict, Tuple
from enum import Enum
from collections import defaultdict
import glob
import os
from functools import partial

import numpy as np
import torch

import polyscope as ps
import polyscope.imgui as psim

from ps_utils.ui import int_popup
from ps_utils.viewer.gizmo import (
    ControlMode,
    CONTROL_MODE_TO_PS_TRANSFORM,
    key_control_mode,
    slider_control_mode,
)
from ps_utils.structures import CUBE_VERTICES_NP, CUBE_EDGES_NP

from brepdgt.primitives.uvgrid import UvGrid, stack_uvgrids, concat_uvgrids

from brepdgt.viewer.ps_uvgrid import (
    UvGridPsData,
    uv_grid_to_ps_data,
    ps_add_uv_grid,
    apply_transform,
)


class RenderMode(Enum):
    POINT_CLOUD: str = "point_cloud"
    POINT_CLOUD_W_NORMALS = "point_cloud_w_normals"
    MESH: str = "mesh"
    NONE: str = "none"


RENDER_MODE_MAP = {x: i for i, x in enumerate(RenderMode)}
RENDER_MODE_INVMAP = {i: x for i, x in enumerate(RenderMode)}

SELECT_BBOX_COLOR = [0.0, 1.0, 0.0]
FROZEN_BBOX_COLOR = [1.0, 0.0, 0.0]

LIST_MAX_HEIGHT = 500

UV_GRIDS_SAVE_FOLDER = "./out/uvgrids"


# Lists all primitives and allows one to edit them individually
class UvGridWidget:
    def __init__(self, widget_id: int = 0):
        self.widget_id = widget_id

        self.ps_structures: Dict[int, ps.Structure] = {}
        self.ps_bbox: Dict[int, ps.Structure] = {}

        self.selected_structures_indices = set()

        self.select_bbox = None
        self.control_mode = ControlMode.TRANSLATION
        self.unmask_N = 10

        # Tracks which primitives are frozen for inpainting
        # NB: this is a permanent property
        # self.frozen_prims = [False for _ in range(self.max_n_prims)]
        self.frozen_prims: List[bool] = []

    @property
    def n_prims(self):
        return len(self.frozen_prims)

    def _adjust_frozen_prims(self, target_n: int):
        if target_n < len(self.frozen_prims):
            self.frozen_prims = self.frozen_prims[:target_n]
        else:
            self.frozen_prims += [False] * (target_n - len(self.frozen_prims))

    # =====================================================
    # Getters/Setters
    # =====================================================

    def set_uv_grids(
        self, uv_grids: UvGrid, render_mode: RenderMode, headless_mode: bool = False
    ):
        if len(uv_grids.coord.shape) != 5:
            raise ValueError("UV grids should be provided in batched format!")
        if uv_grids.coord.shape[0] != 1:
            raise ValueError("UV grids should have one sample in UvGridWidget!")

        self.uv_grids = uv_grids.clone()

        # TODO: handle batched and non-batched versions simultaneously (or at least properly)
        if self.uv_grids.coord.shape[1] != len(self.frozen_prims):
            self._adjust_frozen_prims(self.uv_grids.coord.shape[1])

        # Tracks the currently selected structure
        # self.selected_structure_idx = -1

        # Convert uv grids to polyscope data
        self.uv_grid_ps_data: List[UvGridPsData] = uv_grid_to_ps_data(self.uv_grids)

        self.update_ps_structures(render_mode=render_mode)

    def get_current(self) -> Tuple[UvGrid, List[bool]]:
        self.apply_transforms()
        return self.uv_grids.clone(), self.frozen_prims

    def append_uv_grids(self, uv_grids: UvGrid) -> None:
        """Append the (non-masked) provided uv_grids to the current (non-masked) uv_grids"""
        n_prims = self.uv_grids.coord.shape[1]
        uv_grids = stack_uvgrids([uv_grids])
        uv_grids.to_tensor("cuda")
        self.uv_grids.compactify()  # Compactify debatches
        self.uv_grids = stack_uvgrids([self.uv_grids])
        self.uv_grids = concat_uvgrids([self.uv_grids, uv_grids])
        self.uv_grids.pad(n_prims)
        self.update_ps_structures(self.current_render_mode)

    # =====================================================
    # Polyscope Utilities
    # =====================================================

    def get_structure_name(self, i_prim: int):
        return f"{self.widget_id}_uv_grid_{i_prim}"

    def get_bbox_name(self, i_prim: int):
        return f"{self.widget_id}_bbox_{i_prim}"

    def clear_ps_structures(self):
        for i_prim in range(self.n_prims):
            name = self.get_structure_name(i_prim)
            if ps.has_point_cloud(name):
                ps.remove_point_cloud(name)
            elif ps.has_surface_mesh(name):
                ps.remove_surface_mesh(name)

        self.clear_select_bbox()

    def update_ps_structures(self, render_mode: RenderMode) -> None:
        # NB: this is to ensure consistency between rendering modes since we reset transform to Identity below
        self.apply_transforms()

        # Clear current structures (necessary for masking/unmasking)
        self.clear_ps_structures()

        # Track RenderMode
        # TODO: this is unsafe!
        self.current_render_mode = render_mode

        # Create ps structures
        self.ps_structures: Dict[int, ps.Structure] = {}
        self.ps_bbox: Dict[int, ps.Structure] = {}

        # Don't show anything in BREP mode - the brep mesh will be shown by the denoiser
        if render_mode == RenderMode.NONE:
            return

        for i_prim, data in enumerate(self.uv_grid_ps_data):
            if not self.uv_grids.empty_mask[0, i_prim]:
                # =================
                # Frame
                # =================
                if render_mode == RenderMode.MESH:
                    self.ps_structures[i_prim] = ps_add_uv_grid(
                        data=data,
                        name=self.get_structure_name(i_prim),
                        render_mesh=True,
                    )

                    self.ps_structures[i_prim].set_hover_callback(
                        partial(self.hover_mesh_callback, i_prim=i_prim)
                    )
                    self.ps_structures[i_prim].set_pick_callback(
                        partial(self.pick_mesh_callback, i_prim=i_prim)
                    )
                elif (
                    render_mode == RenderMode.POINT_CLOUD
                    or render_mode == RenderMode.POINT_CLOUD_W_NORMALS
                ):
                    self.ps_structures[i_prim] = ps_add_uv_grid(
                        data=data,
                        name=self.get_structure_name(i_prim),
                        normals_enabled=render_mode == RenderMode.POINT_CLOUD_W_NORMALS,
                        render_mesh=False,
                    )

                    self.ps_structures[i_prim].set_hover_callback(
                        partial(self.hover_pc_callback, i_prim=i_prim)
                    )
                    self.ps_structures[i_prim].set_pick_callback(
                        partial(self.pick_pc_callback, i_prim=i_prim)
                    )
                else:
                    raise ValueError()

                self.ps_structures[i_prim].set_transform(np.eye(4))

                # =================
                # Bounding box
                # =================
                self.ps_bbox[i_prim] = self.create_ps_bbox(
                    self.uv_grids.coord[0, i_prim], i_prim
                )
                self.ps_bbox[i_prim].set_transform(np.eye(4))

        # Init select bbox
        self.init_select_bbox()

    # =======================
    # HOVER + PICK CALLBACKS
    # =======================

    def hover_pc_callback(self, _: int, i_prim: int):
        self.ps_structures[i_prim].set_color([1.0, 1.0, 1.0, 1.0])

    def hover_mesh_callback(self, _: int, __: int, i_prim: int):
        self.ps_structures[i_prim].set_color([1.0, 1.0, 1.0, 1.0])

    def pick_pc_callback(self, _: int, i_prim: int):
        if i_prim not in self.selected_structures_indices:
            self.selected_structures_indices.add(i_prim)
        else:
            self.selected_structures_indices.remove(i_prim)
        self.select()

    def pick_mesh_callback(self, _: int, __: int, i_prim: int):
        if i_prim not in self.selected_structures_indices:
            self.selected_structures_indices.add(i_prim)
        else:
            self.selected_structures_indices.remove(i_prim)
        self.select()

    # Reset colors to data colors
    def update_hover(self):
        for i_prim, data in enumerate(self.uv_grid_ps_data):
            if i_prim in self.ps_structures:
                if self.current_render_mode in {
                    RenderMode.MESH,
                    RenderMode.POINT_CLOUD,
                    RenderMode.POINT_CLOUD_W_NORMALS,
                }:
                    self.ps_structures[i_prim].set_color(data.single_color)

    # =======================================

    def create_ps_bbox(self, coord: torch.Tensor, i_prim: int):
        bbox_min = coord.min(dim=0)[0].min(dim=0)[0].cpu().numpy()
        bbox_max = coord.max(dim=0)[0].max(dim=0)[0].cpu().numpy()
        cube_vertices = (bbox_max - bbox_min) * CUBE_VERTICES_NP + bbox_min

        bbox = ps.register_curve_network(
            self.get_bbox_name(i_prim),
            cube_vertices,
            CUBE_EDGES_NP,
            radius=0.004,
            color=FROZEN_BBOX_COLOR,
            enabled=self.frozen_prims[i_prim],
        )
        bbox.set_transform(np.eye(4))
        return bbox

    # Move the selected point cloud with the selected bbox
    def update_transform(self):
        if len(self.selected_structures_indices) > 0 and self.select_bbox is not None:
            for selected_structure_idx in self.selected_structures_indices:
                if selected_structure_idx in self.ps_structures:
                    self.ps_structures[selected_structure_idx].set_transform(
                        self.select_bbox.get_transform()
                        @ np.linalg.inv(self.bbox_transform)
                    )

        for i_prim, bbox in self.ps_bbox.items():
            bbox.set_enabled(self.frozen_prims[i_prim])
            bbox.set_transform(np.eye(4))

    # This applies all the transformations (with Gizmos) to the corresponding UV Grids
    def apply_transforms(self) -> None:
        for i_prim, structure in self.ps_structures.items():
            self.uv_grids.coord[0, i_prim] = apply_transform(
                self.uv_grids.coord[0, i_prim],
                structure.get_transform(),
            )

            # NB: make sure to reset the transform to identity on the ps.Structure!
            structure.set_transform(np.eye(4))

        # if self.select_bbox is not None:
        #     self.init_select_bbox()

        # Don't forget to update the whole ps_data again
        self.uv_grid_ps_data: List[UvGridPsData] = uv_grid_to_ps_data(self.uv_grids)

    def init_select_bbox(self):
        if len(self.selected_structures_indices) == 0:
            return

        # TODO: handle multiple UV-grids!
        all_coords = torch.cat(
            [
                self.uv_grids.coord[0, selected_structure_idx].reshape(-1, 3)
                for selected_structure_idx in self.selected_structures_indices
            ],
            dim=0,
        )

        bbox_min = all_coords.min(dim=0).values.cpu().numpy()
        bbox_max = all_coords.max(dim=0).values.cpu().numpy()
        self.bbox_center = (bbox_max + bbox_min) / 2.0
        cube_vertices = (bbox_max - bbox_min) * (CUBE_VERTICES_NP - 0.5)
        # We need to apply a slight transform to recenter the bbox
        self.bbox_transform = np.eye(4)
        self.bbox_transform[:3, 3] = self.bbox_center

        self.select_bbox = ps.register_curve_network(
            f"{self.widget_id}_select_bbox",
            cube_vertices,
            CUBE_EDGES_NP,
            radius=0.004,
            color=SELECT_BBOX_COLOR,
        )
        self.select_bbox.set_transform(self.bbox_transform)
        self.select_bbox.set_transform_mode_gizmo(
            CONTROL_MODE_TO_PS_TRANSFORM[self.control_mode]
        )
        self.select_bbox.enable_transform_gizmo(True)

    def clear_select_bbox(self):
        if self.select_bbox is not None:
            self.select_bbox.remove()
            self.select_bbox = None

    def select(self, force_unselect: bool = False) -> None:
        if force_unselect:
            self.selected_structures_indices = set()

        self.update_ps_structures(self.current_render_mode)

        if len(self.selected_structures_indices) == 0:
            self.clear_select_bbox()
            return
        if any(
            [idx not in self.ps_structures for idx in self.selected_structures_indices]
        ):
            print(
                f"WARNING: you tried to select a uv-grid for which there is no structure!"
            )
            return

        self.init_select_bbox()

    def _unmask_N_prim(self):
        self.uv_grids.unmask_N_first(self.unmask_N)
        self.uv_grid_ps_data: List[UvGridPsData] = uv_grid_to_ps_data(self.uv_grids)
        self.update_ps_structures(self.current_render_mode)

    # Returns
    # (
    #   True if one of the structure is selected in this grid,
    #   True if uv grid has changed (BROKEN)
    #   True if the frozen prims changed
    # )
    def gui(self) -> Tuple[bool, bool]:

        uv_grid_modified = False
        frozen_prims_modified = False

        self.update_hover()

        self.update_transform()

        # ========================
        # BUTTONS
        # ========================

        # Rescale
        if psim.Button(f"Rescale (bbox)##uv_grid_widget_{self.widget_id}"):
            uv_grids, _ = self.get_current()
            rescaled_uv_grids, _, _ = uv_grids.normalize_to_cube(nonempty_only=True)
            self.set_uv_grids(rescaled_uv_grids, self.current_render_mode)
            uv_grid_modified |= True

        # Freeze/Unfreeze All
        psim.SameLine()
        freeze_all = not any(self.frozen_prims)
        freeze_all_text = "Freeze All" if freeze_all else "Unfreeze All"
        if psim.Button(f"{freeze_all_text}##uv_grid_widget_{self.widget_id}"):
            if freeze_all:
                for i_prim in range(self.n_prims):
                    self.frozen_prims[i_prim] = not self.uv_grids.empty_mask[0, i_prim]
            else:
                self.frozen_prims = [False for _ in range(self.n_prims)]

            uv_grid_modified |= True
            frozen_prims_modified |= True

        # Unselect
        psim.SameLine()
        if len(self.selected_structures_indices) > 0:
            if psim.Button(f"Unselect All##uv_grid_widget_{self.widget_id}"):
                self.select(True)
        else:
            if psim.Button(f"Select All##uv_grid_widget_{self.widget_id}"):
                self.selected_structures_indices = set(
                    [i_prim for i_prim in range(self.n_prims)]
                )
                self.select()

        # Mask N
        psim.SameLine()
        clicked, self.unmask_N = int_popup(
            f"unmask_{self.widget_id}",
            self.unmask_N,
            button_label="Unmask N",
        )
        if clicked:
            self._unmask_N_prim()
            uv_grid_modified |= True

        # ========================
        # CONTROL_MODE
        # ========================

        if self.select_bbox is not None:

            self.control_mode = slider_control_mode(
                "Control", self.select_bbox, self.control_mode
            )

            self.control_mode = key_control_mode(self.select_bbox, self.control_mode)

        # ========================
        # UV-GRIDs
        # ========================

        # TODO: add this button for the denoiser!
        # if psim.Button("Export UV Grid##uv_grid_widget"):
        #     self.get_current()[0].export_npz("editor_uv_grid.npz", 0)

        # TEXT_SIZE = psim.CalcTextSize("A")[0]
        TEXT_BASE_HEIGHT = psim.GetTextLineHeightWithSpacing()

        if psim.BeginTable(
            f"UV grids##uv_grid_widget_{self.widget_id}_table",
            5,
            psim.ImGuiTableFlags_ScrollY,
            (0, min(TEXT_BASE_HEIGHT * (self.n_prims + 1.5), LIST_MAX_HEIGHT)),
        ):
            psim.TableSetupColumn(
                "Name",
                psim.ImGuiTableColumnFlags_WidthStretch,
                0.0,
            )
            psim.TableSetupColumn(
                "Color",
                psim.ImGuiTableColumnFlags_WidthFixed
                | psim.ImGuiTableColumnFlags_NoHide,
                0.0,
            )
            psim.TableSetupColumn(
                "Mask",
                psim.ImGuiTableColumnFlags_WidthFixed
                | psim.ImGuiTableColumnFlags_NoHide,
                0.0,
            )
            psim.TableSetupColumn(
                "Selected",
                psim.ImGuiTableColumnFlags_WidthFixed
                | psim.ImGuiTableColumnFlags_NoHide,
                0.0,
            )
            psim.TableSetupColumn(
                "Frozen",
                psim.ImGuiTableColumnFlags_WidthFixed
                | psim.ImGuiTableColumnFlags_NoHide,
                0.0,
            )

            psim.TableHeadersRow()

            for i_prim in range(self.n_prims):

                psim.TableNextRow()

                # Display Name
                psim.TableNextColumn()
                psim.Text(f"{i_prim}")

                # Color Preview
                psim.TableNextColumn()
                if not self.uv_grids.empty_mask[0, i_prim]:
                    psim.ColorEdit3(
                        f"##uv_grid_widget_{self.widget_id}_color_{i_prim}",
                        self.uv_grid_ps_data[i_prim].single_color,
                        flags=psim.ImGuiColorEditFlags_NoInputs,
                    )

                # Mask
                psim.TableNextColumn()
                clicked, tmp = psim.Checkbox(
                    f"##uv_grid_widget_{self.widget_id}_mask_{i_prim}",
                    not self.uv_grids.empty_mask[0, i_prim],
                )
                self.uv_grids.empty_mask[0, i_prim] = not tmp
                if clicked:
                    # TODO: this is dirty! (and unsafe)
                    self.uv_grid_ps_data: List[UvGridPsData] = uv_grid_to_ps_data(
                        self.uv_grids
                    )
                    self.update_ps_structures(self.current_render_mode)
                    uv_grid_modified |= True

                # Disables the rest if it is masked
                psim.BeginDisabled(self.uv_grids.empty_mask[0, i_prim])

                # Select
                psim.TableNextColumn()
                clicked, selected = psim.Checkbox(
                    f"##uv_grid_widget_{self.widget_id}_select_{i_prim}",
                    i_prim in self.selected_structures_indices,
                )
                if clicked:
                    if selected and i_prim not in self.selected_structures_indices:
                        self.selected_structures_indices.add(i_prim)
                    elif not selected and i_prim in self.selected_structures_indices:
                        self.selected_structures_indices.remove(i_prim)
                    self.select()

                # Freeze
                psim.TableNextColumn()
                clicked, self.frozen_prims[i_prim] = psim.Checkbox(
                    f"##uv_grid_widget_{self.widget_id}_frozen_{i_prim:02d}",
                    self.frozen_prims[i_prim],
                )
                if clicked:
                    uv_grid_modified |= True
                    frozen_prims_modified |= True

                psim.EndDisabled()
            psim.EndTable()

        uv_grid_modified |= self.mask_edit_gui()

        selected = len(self.selected_structures_indices) > 0
        return selected, uv_grid_modified, frozen_prims_modified

    # Mask editing is only possible with one selected UV-grid!
    def mask_edit_gui(self) -> bool:

        if len(self.selected_structures_indices) != 1:
            return False

        selected_structure_idx = next(iter(self.selected_structures_indices))

        updated = False
        x_size, y_size = self.uv_grids.grid_mask[0, selected_structure_idx].shape
        for x_grid in range(x_size):
            for y_grid in range(y_size):
                (
                    clicked,
                    self.uv_grids.grid_mask[0, selected_structure_idx, x_grid, y_grid],
                ) = psim.Checkbox(
                    f"##uv_grid_widget_{self.widget_id}_{x_grid}_{y_grid}_mask",
                    self.uv_grids.grid_mask[0, selected_structure_idx, x_grid, y_grid],
                )
                if y_grid < y_size - 1:
                    psim.SameLine()

                updated |= clicked

        if updated:
            self.update_ps_structures(self.current_render_mode)

        return updated

    @staticmethod
    def get_next_save_path() -> str:
        os.makedirs(UV_GRIDS_SAVE_FOLDER, exist_ok=True)
        all_exported_paths = glob.glob(
            os.path.join(UV_GRIDS_SAVE_FOLDER, "exported_*.npz")
        )
        return os.path.join(
            UV_GRIDS_SAVE_FOLDER, f"exported_{len(all_exported_paths):06d}.npz"
        )
