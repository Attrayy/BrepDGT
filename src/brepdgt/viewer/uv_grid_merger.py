from typing import Dict, List
from dataclasses import dataclass, field
import os
import glob

import torch
import numpy as np

import polyscope as ps
import polyscope.imgui as psim

from ps_utils.ui import save_popup

from brepdgt.primitives.uvgrid import UvGrid, stack_uvgrids, concat_uvgrids
from brepdgt.models.tokenizers.base import Tokens
from brepdgt.config import Config
from brepdgt.models.brepdgt import BRepDGT
from brepdgt.viewer.uv_grid_widget import (
    UvGridWidget,
    RenderMode,
    RENDER_MODE_INVMAP,
    RENDER_MODE_MAP,
)


from brepdgt.utils.convert_utils import (
    brep_to_mesh,
    brep_to_uvgrid,
)


class UvGridMergerConfig:
    pass


class UvGridMerger:
    """Simple tool to load an manipulate various UvGrids"""

    def __init__(
        self,
        model: BRepDGT,
        zooc_config: Config,
        config: UvGridMergerConfig = UvGridMergerConfig(),
    ):
        self.config = config

        # We still need the zooc_config to check uv-grid sizes, etc.
        self.zooc_config = zooc_config

        self.render_mode = RenderMode.POINT_CLOUD

        self.widgets: List[UvGridWidget] = []
        self.widget_names: List[str] = []

        self.save_at_text = ""
        self.uv_grids_save_path = UvGridWidget.get_next_save_path()

        # DEBUG
        # self.ps_drop_callback("./uv_grids/00000178.npz")
        # self.ps_drop_callback("./uv_grids/00007655.npz")

    def _get_new_widget_name(self, input_path):
        if input_path not in self.widget_names:
            return input_path
        # Count the number of names starting similarly
        count = len([name for name in self.widget_names if name.startswith(input_path)])
        return f"{input_path}_{count:02d}"

    # =====================================
    # Save/Load utils
    # =====================================

    # Load UV Grid (passed this point, the UV grid file should be valid!)
    # WARNING: uv-grids get modified in place!
    def _load_uv_grids(self, uv_grids: UvGrid, name: str):
        # Create a new UvGridWidget
        new_widget = UvGridWidget(widget_id=len(self.widgets))

        # TODO: Ideally, we don't want this but for compatibility, I have to do this for now :/
        uv_grids.to_tensor("cuda")
        uv_grids = stack_uvgrids([uv_grids])

        new_widget.set_uv_grids(
            uv_grids,
            render_mode=self.render_mode,
        )
        self.widgets.append(new_widget)
        self.widget_names.append(name)

    def ps_drop_callback(self, input: str) -> None:
        extension = os.path.splitext(input)[1]
        if extension == ".npz":
            try:
                # Load UvGrid from .npz file
                uv_grids = UvGrid.load_from_npz_data(np.load(input))
                self._load_uv_grids(uv_grids, self._get_new_widget_name(input))
            except Exception as e:
                # handle the exception
                print("Could not import from:", input)
                print("Error:\n", e)
        elif extension == ".step":
            try:
                from OCC.Extend.DataExchange import read_step_file

                solid = read_step_file(input)
                uv_grids = brep_to_uvgrid(solid)
                self._load_uv_grids(uv_grids, self._get_new_widget_name(input))
            except Exception as e:
                # handle the exception
                print("Could not import from:", input)
                print("Error:\n", e)
        else:
            print("Only .npz and .step files are accepted!")
            return

    def _export_selection(self, save_path: str) -> None:

        all_uv_grids = []
        for widget in self.widgets:
            uv_grids, frozen_prims = widget.get_current()
            filtered_uv_grids = uv_grids.filter(frozen_prims)
            all_uv_grids.append(filtered_uv_grids)

        if len(all_uv_grids) == 0:
            return

        all_uv_grids = concat_uvgrids(all_uv_grids)
        all_uv_grids.export_npz(save_path, 0)

        self.save_at_text = "Saved at: " + str(save_path)

    # =====================================
    # GUI
    # =====================================

    def gui(self) -> None:

        clicked, render_mode_idx = psim.SliderInt(
            "Render Mode",
            RENDER_MODE_MAP[self.render_mode],
            v_min=0,
            v_max=len(RenderMode) - 1,
            format=f"{self.render_mode.value}",
        )

        if clicked:
            self.render_mode = RENDER_MODE_INVMAP[render_mode_idx]
            for widget in self.widgets:
                widget.update_ps_structures(self.render_mode)
            # TODO: update UV Grid rendering
            # self.uv_grid_widget.update_ps_structures(self.render_mode)

        # Export
        clicked, self.uv_grids_save_path = save_popup(
            "uv_grid_merger_save", self.uv_grids_save_path, save_label="Save Frozen"
        )
        if clicked:
            self._export_selection(self.uv_grids_save_path)
            self.uv_grids_save_path = UvGridWidget.get_next_save_path()

        if len(self.save_at_text) > 0:
            psim.Text(self.save_at_text)

        # ================
        # UV Grid Widgets
        # ================

        for i_widget, (widget, widget_name) in enumerate(
            zip(self.widgets, self.widget_names)
        ):
            if psim.TreeNode(widget_name):
                selected, _, _ = widget.gui()
                # If it is selected, make sure to unselect the rest
                # TODO: cleaner state handling here!
                if selected:
                    for j_widget in range(len(self.widgets)):
                        if j_widget == i_widget:
                            continue
                        self.widgets[j_widget].select(True)
                psim.TreePop()
