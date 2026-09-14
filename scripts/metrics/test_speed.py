"""
Time diffusion inference to pp
"""

import numpy as np
import typer
import os
import torch
from brepdgt.config import load_config
from brepdgt.lit_model import LitModel
from brepdgt.models.brepdgt import BRepDGTReconstruction, BRepDGT
from brepdgt.primitives.uvgrid import UvGrid, uncat_uvgrids
from brepdgt.postprocessing.postprocessor import BatchPostprocessor, Postprocessor
from collections import defaultdict
from interruptingcow import timeout
import time
from contextlib import nullcontext
from tqdm import tqdm


app = typer.Typer(pretty_exceptions_enable=False)


class Timer:
    def __init__(self):
        self.time_dict = defaultdict(list)
        self.temp_time_dict = defaultdict(list)  # temporary placeholder
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        self.current_name = None

    def __enter__(self):
        self.start_time = time.time()
        return self  # return the Timer instance itself

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed_time = time.time() - self.start_time
        # print(f"{self.name} elapsed time: {self.elapsed_time:.6f} seconds")
        self.temp_time_dict[self.current_name] = elapsed_time
        self.current_name = None
        self.start_time = None

    def add_permanantly(self):
        total_time = 0
        for k, v in self.temp_time_dict.items():
            self.time_dict[k] += [v]
            total_time += v
        self.time_dict["total"] += [total_time]

    def time_block(self, name):
        # Set the name for the current block to be timed
        self.current_name = name
        return self

    def clear(self):
        self.temp_time_dict.clear()

    def print(self):
        # print average stats
        for k, v in self.time_dict.items():
            print(f"{k}: {sum(v)/len(v):.2f}")

    def export(self, out_path: str):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez(out_path, **self.time_dict)


@app.command()
def main(
    ckpt_path: str,
    n_generate: int = 100,
    grid_res: int = 256,
    max_pp_time: int = 45,
    coarse_to_fine: bool = True,  # use 128 res and then 256 for postprocessing
):
    from OCC.Extend.DataExchange import write_step_file
    from brepdgt.utils.brep_checker import check_solid

    base_dir = os.path.dirname(os.path.dirname(ckpt_path))
    config_path = os.path.join(base_dir, "config.yaml")

    override = "test_batch_size=1"

    # Load config first to get default devices if needed
    config = load_config(config_path, override)

    # build model
    lit_model = LitModel.load_from_checkpoint(ckpt_path, config=config)
    model: BRepDGT = lit_model.model
    model.eval()
    model = model.cuda()

    speed_dir = os.path.join(config.log_dir, "speed")
    os.makedirs(speed_dir, exist_ok=True)
    step_dir = os.path.join(speed_dir, "step")
    os.makedirs(step_dir, exist_ok=True)

    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()

    # get train dataloader to sample number of points
    dataloader = lit_model.test_dataloader()
    timer = Timer()
    watertight_cnt = 0
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        batch.uvgrid.to_tensor("cuda")
        with timer.time_block("inference"):
            sample: BRepDGTReconstruction = model.sample(
                batch, return_traj=False, cfg_scale=1.0
            )
        uvgrid = uncat_uvgrids(sample.uvgrids)[0]
        with timer.time_block("postprocessing"):
            try:
                if coarse_to_fine:
                    with timeout(max_pp_time):
                        pp = Postprocessor(uvgrid, grid_res=128)
                        brep = pp.get_brep(ret_none_if_failed=True)
                    step_out_path = os.path.join(
                        step_dir, f"{str(batch_idx).zfill(4)}.step"
                    )
                    write_step_file(brep, step_out_path)
                    is_watertight, n_faces = check_solid(
                        step_path=step_out_path, timeout=20
                    )
                    if not is_watertight:
                        with timeout(max_pp_time):
                            pp = Postprocessor(uvgrid, grid_res=256)
                            brep = pp.get_brep(ret_none_if_failed=True)

                else:
                    with timeout(max_pp_time):
                        pp = Postprocessor(uvgrid, grid_res=grid_res)
                        brep = pp.get_brep(ret_none_if_failed=True)
            except Exception as e:
                print(f"pp failed: {e}")
                brep = None

        if brep is not None:
            step_out_path = os.path.join(step_dir, f"{str(batch_idx).zfill(4)}.step")
            write_step_file(brep, step_out_path)
            is_watertight, n_faces = check_solid(step_path=step_out_path, timeout=20)
            if is_watertight:
                watertight_cnt += 1
                timer.add_permanantly()
        if watertight_cnt >= n_generate:
            break
        timer.clear()

    timer.print()
    timer_path = os.path.join(speed_dir, "speed_ours.npz")
    if coarse_to_fine:
        timer_path.replace(".npz", "_coarse_to_fine.npz")
    timer.export(timer_path)


@app.command()
def dummy():
    pass


if __name__ == "__main__":
    app()
