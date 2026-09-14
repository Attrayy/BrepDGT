import os
from tqdm import tqdm
import typer
import numpy as np
import h5py
from glob import glob
from typing import Optional

from brepdgt.metrics.fid import calculate_fid_given_npz_and_paths, save_fid_stats
from brepdgt.datasets._metadata import ABC_INVALID_IDS
from brepdgt.utils.common import load_txt, load_pkl

app = typer.Typer()


@app.command()
def compute_fid_from_renders(
    render_dir: str = typer.Argument(..., help="Directory containing render images"),
    npz_path: str = typer.Argument(
        "data/abc_processed/fid/fid_cache_10.npz",
        help="Path to cached FID stats npz file",
    ),
    batch_size: int = typer.Option(64, help="Batch size for FID computation"),
    device: str = typer.Option("cuda", help="Device to use (cuda/cpu)"),
    fid_type: str = typer.Option("cad", help="Type of FID to compute"),
    num_workers: int = typer.Option(16, help="Number of workers for data loading"),
    num_samples: int = typer.Option(
        256, help="Number of samples to use for FID computation"
    ),
):
    """Compute FID score between cached stats and a directory of renders"""
    render_paths = glob(os.path.join(render_dir, "*.png"))
    render_paths = render_paths[:num_samples]

    fid = calculate_fid_given_npz_and_paths(
        npz_path=npz_path,
        paths=render_paths,
        batch_size=batch_size,
        device=device,
        fid_type=fid_type,
        num_workers=num_workers,
    )
    print(f"FID score: {fid}")
    return fid


@app.command()
def create_fid_cache(
    h5_path: str = typer.Option(
        "./data/abc_processed/v1_grid8.h5", help="Path to H5 dataset"
    ),
    dataset_name: str = typer.Option("deepcad", help="Dataset name"),
    split: str = typer.Option("val", help="Data split to use for FID cache"),
    max_n_prims: int = typer.Option(10, help="Maximum number of primitives"),
    fid_type: str = typer.Option("cad", help="Type of FID to compute"),
    fid_path: str = typer.Option(
        "data/abc_processed/fid", help="Path to FID cache directory"
    ),
    n_views: int = typer.Option(1, help="Number of views to use for FID cache"),
    device: str = typer.Option("cuda", help="Device to use (cuda/cpu)"),
):
    """Create FID statistics cache from reference dataset"""
    if fid_type == "cad":
        npz_path = os.path.join(
            fid_path, f"fid_cache_{dataset_name}_{max_n_prims}_{n_views}views.npz"
        )
    elif fid_type == "cad_v1":
        npz_path = os.path.join(
            fid_path, f"fid_cache_v1_{dataset_name}_{max_n_prims}_{n_views}views.npz"
        )
    elif fid_type == "vanilla":
        npz_path = os.path.join(
            fid_path,
            f"fid_cache_vanilla_{dataset_name}_{max_n_prims}_{n_views}views.npz",
        )
    else:
        raise ValueError(f"{fid_type} not allowed")
    if split == "test":
        npz_path = npz_path.replace(".npz", "_test.npz")

    print(f"Cache path: {npz_path}")

    # Load dataset-specific data list with format {split}_{dataset}_{max_n_prims}.txt
    data_list_path = os.path.join(
        fid_path, f"fid_{dataset_name}_{max_n_prims}_{split}.txt"
    )
    if not os.path.exists(data_list_path):
        raise FileNotFoundError(
            f"Data list not found at {data_list_path}. Please run create_valid_lists first."
        )

    data_list = load_txt(data_list_path)
    print(f"Using {len(data_list)} samples from {dataset_name} {split} set")

    paths = []
    for uid in data_list:
        for view_idx in range(n_views):
            paths.append(os.path.join(fid_path, uid[:4], uid, f"{view_idx}.png"))

    save_fid_stats(paths, npz_path, batch_size=50, device=device, fid_type=fid_type)

    fid = calculate_fid_given_npz_and_paths(
        npz_path, paths, batch_size=50, device=device, fid_type=fid_type
    )

    print(f"FID: {fid}")


if __name__ == "__main__":
    app()
