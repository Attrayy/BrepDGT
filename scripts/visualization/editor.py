import typer
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path
import polyscope as ps
import polyscope.imgui as psim
import time
import os
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from ps_utils.ui import KEY_HANDLER
from ps_utils.structures import create_bbox

from brepdgt.config import load_config
from brepdgt.config import Config
from brepdgt.lit_model import LitModel
from brepdgt.models.brepdgt import BRepDGT
from brepdgt.viewer.denoiser import Denoiser
from brepdgt.viewer.uv_grid_merger import UvGridMerger

app = typer.Typer(pretty_exceptions_enable=False)

COMPONENT_MAP = {
    "denoiser": Denoiser,
    "uv_grids": UvGridMerger,
}


class Viewer:
    def __init__(self, config: Config, model: BRepDGT, component: str = "denoiser"):
        ps.init()
        self.ps_init()

        self.config = config
        self.model = model

        self.component = COMPONENT_MAP[component](zooc_config=config, model=model)

        ps.set_user_callback(self.ps_callback)
        ps.set_drop_callback(self.ps_drop_callback)
        ps.show()

    def ps_init(self) -> None:
        """Initialize Polyscope"""
        ps.set_ground_plane_mode("none")
        ps.set_max_fps(120)
        ps.set_window_size(1920, 1080)
        ps.set_SSAA_factor(4)
        ps.set_automatically_compute_scene_extents(False)
        create_bbox()
        self.last_time = time.time()

    def ps_callback(self) -> None:
        # Update FPS count
        new_time = time.time()
        self.fps = 1.0 / (new_time - self.last_time)
        self.last_time = new_time
        if isinstance(self.component, Denoiser):
            psim.Text(
                f"fps: {self.fps:.4f}; {self.component.i_step} / {self.component.config.n_inference_steps}"
            )
        else:
            psim.Text(f"fps: {self.fps:.4f}")

        self.gui()

        KEY_HANDLER.step()

    def ps_drop_callback(self, input: str) -> None:
        self.component.ps_drop_callback(input)

    def gui(self) -> None:
        self.component.gui()


@app.command()
def main(
    config_path: str = "./configs/brepdgt_deepcad_pc.yaml",
    override: str = "",
    ckpt_path: str = "",
    component: str = "denoiser",
):
    if ckpt_path:
        # use the ckpt's config path
        base_dir = os.path.dirname(os.path.dirname(ckpt_path))
        config_path = os.path.join(base_dir, "config.yaml")

    # Load config
    config = load_config(config_path, override)

    # Load config and model
    model = LitModel(config)
    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        state_dict = ckpt["state_dict"]
        model.load_state_dict(state_dict)
    model.eval()
    model.cuda()

    model = model.model  # Unwrap LitModel

    # Initialize single viewer
    viewer = Viewer(config=config, model=model, component=component)


if __name__ == "__main__":
    app()
