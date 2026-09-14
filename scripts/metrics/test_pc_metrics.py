from brepdgt.utils.common import save_pointcloud
from brepdgt.metrics.pc_metrics import (
    compute_pc_metrics,
    normalize_pcs_tensor,
    read_ply,
)

from plyfile import PlyData
from pathlib import Path
from glob import glob
from tqdm import tqdm
import numpy as np
import os
import random
from multiprocessing import Pool
from typing import Dict, List
import torch
import typer

app = typer.Typer()

NUM_TRHEADS = 6
N_POINTS = 2000


def downsample_pc(points, n):
    sample_idx = random.sample(list(range(points.shape[0])), n)
    return points[sample_idx]


@app.command()
def eval_pc(
    gen_pc_dir: str = "",
    gen_pc_paths: List[str] = "",
    real_pc_dir: str = None,
    real_pc_npz_path: str = None,
    n_iter: int = 3,
    rand_idx: bool = True,
    verbose: bool = True,
    normalize: str = "longest_axis",
    n_samples: int = None,
):
    print(f"LOADING pcs from reference ply from {real_pc_dir}")
    # Load reference ply
    if real_pc_dir is not None:
        ref_pcs = []
        ply_paths = sorted(glob(real_pc_dir + "/*"))
        print(f"Found {len(ply_paths)} ply files")
        load_iter = Pool(NUM_TRHEADS).imap(read_ply, ply_paths)
        for pc in tqdm(load_iter, total=len(ply_paths)):
            if len(pc) > 0:
                if len(pc) > N_POINTS:
                    pc = downsample_pc(pc, N_POINTS)
                ref_pcs.append(pc)
        ref_pcs = np.stack(ref_pcs, axis=0)
    elif real_pc_npz_path is not None:
        ref_pcs = dict(np.load(real_pc_npz_path))["pts"].astype(np.float32)
        # filter points with infinite
        inf_mask = (~np.isfinite(ref_pcs)).sum(axis=(1, 2)) > 0
        ref_pcs = ref_pcs[~inf_mask]
    else:
        raise ValueError(
            f"Either {real_pc_dir} and {real_pc_npz_path} should not be None"
        )
    print("real point clouds: {}".format(ref_pcs.shape))

    print(f"LOADING pcs from generated ply from {gen_pc_dir}")
    # Load fake ply
    sample_pcs = []
    if gen_pc_dir:
        ply_paths = sorted(glob(gen_pc_dir + "/*.ply"))
    elif gen_pc_paths:
        ply_paths = gen_pc_paths
    else:
        raise ValueError("either give generated ply paths or dir")
    print(f"Found {len(ply_paths)} ply files")
    load_iter = Pool(NUM_TRHEADS).imap(read_ply, ply_paths)
    for pc in tqdm(load_iter, total=len(ply_paths)):
        if len(pc) > 0:
            sample_pcs.append(pc)
    sample_pcs = np.stack(sample_pcs, axis=0)
    print("fake point clouds: {}".format(sample_pcs.shape))

    if n_samples is None:
        n_samples = sample_pcs.shape[0]
    rand_state = np.random.RandomState(seed=0)

    with torch.no_grad():
        ref_pcs_tensor = torch.tensor(ref_pcs).cuda()
        sample_pcs_tensor = torch.tensor(sample_pcs).cuda()

        results_list = []
        print(f"Computing pc metrics by averaging {n_iter} iterations...")
        for i in tqdm(range(n_iter), total=n_iter):
            if verbose:
                print(f"Iteration: {i}")
            if rand_idx:
                # sample random idx
                select_idx = rand_state.permutation(len(ref_pcs_tensor))[:n_samples]
                ref_pcs_tensor_i = ref_pcs_tensor[select_idx]
            else:
                ref_pcs_tensor_i = ref_pcs_tensor[:n_samples]

            if verbose:
                print("Computing metrics")
                print("ref point clouds: {}".format(ref_pcs_tensor_i.shape))
                print("gen point clouds: {}".format(sample_pcs_tensor.shape))
            metrics = compute_pc_metrics(
                sample_pcs_tensor,
                ref_pcs_tensor_i,
                verbose=verbose,
                normalize=normalize,  # Enable normalization
            )

            for k, v in metrics.items():
                metrics[k] = v.cpu().item()

            if verbose:
                print("Point Cloud Metrics:")
                for k, v in metrics.items():
                    print(f"  {k}: {v}")
            results_list.append(metrics)

    avg_result = {}
    for k in results_list[0].keys():
        avg_result.update({"avg-" + k: np.mean([x[k] for x in results_list])})
    if verbose:
        print("Average PC Result:")
        for k, v in avg_result.items():
            print(f"  {k}: {v}")
    return avg_result


if __name__ == "__main__":
    app()
