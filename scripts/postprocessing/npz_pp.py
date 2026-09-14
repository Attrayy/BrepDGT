"""
A utility tool to visualize postprocesssing.
Given a uv grid npz file, you can visualize the extended partition by running the following:
     python -m scripts.postprocessing.npz_pp npz-pp {path_to_npz_file} {other_options}

Given a directory of npz files, you can postprocess them be running the following:
    python -m scripts.postprocessing.npz_pp save-breps {npz_dir}
"""

import typer
import numpy as np
from brepdgt.primitives.uvgrid import UvGrid
from brepdgt.postprocessing.postprocessor import Postprocessor, BatchPostprocessor
from OCC.Extend.DataExchange import write_stl_file, write_step_file, read_step_file
from brepdgt.utils.brep_checker import check_solid
from glob import glob
import os
import matplotlib.pyplot as plt
from brepdgt.utils.common import load_pkl
from scripts.metrics.evaluate_step_files import (
    test as test_step_files,
    cache_valid_step,
)
from tqdm import tqdm

app = typer.Typer(pretty_exceptions_enable=False)


def plot_pp_face_count_distribution(
    uvgrid_dir: str,
    cached_valid_path: str,
    out_path: str,
    n_test_watertight: int,
    max_n_prims: int,
):
    """Plot distribution of face counts comparing samples vs ground truth"""
    cached_valid = load_pkl(cached_valid_path)
    names, watertight_masks = cached_valid["name"], cached_valid["watertight_mask"]
    watertight_names = names[watertight_masks][:n_test_watertight]
    last_idx = int(watertight_names[-1])

    # Collect counts for sample paths
    gt, pp_success = [], []
    for i in range(last_idx + 1):
        uvgrid_name = str(i).zfill(5)
        npz_path = os.path.join(uvgrid_dir, f"{uvgrid_name}.npz")
        npz_data = np.load(npz_path)
        uvgrid = UvGrid.load_from_npz_data(npz_data)
        n_face = (~uvgrid.empty_mask).sum()
        gt.append(n_face)
        if uvgrid_name in watertight_names:
            pp_success.append(n_face)

    # Get unique face counts and frequencies
    pp_success_unique, pp_success_cnts = np.unique(pp_success, return_counts=True)
    gt_unique, gt_cnts = np.unique(gt, return_counts=True)

    pp_success_freq, gt_freq = np.zeros(max_n_prims + 1), np.zeros(max_n_prims + 1)
    pp_success_freq[pp_success_unique] = 100 * (pp_success_cnts / len(gt))
    gt_freq[gt_unique] = 100 * (gt_cnts / len(gt))

    # Plot side-by-side bar chart
    plt.figure(figsize=(12, 6))
    x = np.arange(max_n_prims + 1)
    width = 0.35

    plt.bar(x - width / 2, pp_success_freq, width, label="PP succeeded", alpha=0.7)
    plt.bar(x + width / 2, gt_freq, width, label="Sample", alpha=0.7)

    plt.xlabel("Number of Faces")
    plt.ylabel("Frequency (%)")
    plt.title("Face Count Distribution: Samples vs PP succeeded")
    # plt.xticks(x, sorted(set(pp_success_unique) | set(gt_unique)))
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(out_path)
    plt.close()


@app.command()
def save_breps(
    out_root_dir: str,
    uvgrid_extension_len: float = 1.0,
    smooth_extension: bool = True,
    process: bool = True,
    grid_res: int = 256,
    min_extend_len: float = 0.0,
    test: bool = True,
    plot: bool = True,
    debug: bool = False,
    max_n_prims: int = 30,
    dataset_name: str = "deepcad",
    n_test_watertight: int = 3000,
    max_pp_time: int = 30,
    skip_if_exists: bool = True,
    multiple_param_extension: bool = False,
    coarse_to_fine: bool = False,
):
    """
    :param out_root_dir: Ex) ./logs/241218-deepcad-30/boom-raw/vis/test/step-000486000/cfg_1.00/
    :param uvgrid_extension_len:
    :param smooth_extension:
    :param process:
    :param grid_res:
    :param test:
    :param plot:
    :param debug:
    :param max_n_prims:
    :param dataset_name:
    :param n_test_watertight:
    :param max_pp_time: timeout for pp (in seconds)
    :param skip_if_exists: skip pp if it already exists
    :param multiple_param_extension: try mulitple param extension if pp fails
    :param coarse_to_fine: try mulitple voxel res if pp fails
    :return:
    """

    if multiple_param_extension:
        assert (
            not multiple_param_extension
        ), f"coarse_to_fine and multiple_param_extension not allowed at the same time"
        out_dir = os.path.join(
            out_root_dir,
            f"multiple_extension-smooth_{str(smooth_extension)}-grid_{grid_res}",
        )
    elif coarse_to_fine:
        out_dir = os.path.join(
            out_root_dir, f"coarse_to_fine-smooth_{str(smooth_extension)}"
        )
    else:
        out_dir = os.path.join(
            out_root_dir,
            f"out_extlen_{str(uvgrid_extension_len)}-smooth_{str(smooth_extension)}-grid_{grid_res}-min_extend_{min_extend_len}",
        )

    if process:
        npz_paths = list(sorted(glob(os.path.join(out_root_dir, "uvgrid", "*.npz"))))
        if debug:
            npz_paths = npz_paths[:5]

        batch_pp = BatchPostprocessor(
            out_dir,
            multiple_param_extension,
            coarse_to_fine,
            uvgrid_extension_len=uvgrid_extension_len,
            smooth_extension=smooth_extension,
            min_extend_len=min_extend_len,
            grid_res=grid_res,
        )
        npz_names = []
        total_processed = 0
        total_watertight = 0

        if skip_if_exists:
            # Restore all step counts
            # Useful cause sometimes the pp dies for no reason
            print("Restoring all step counts...")
            step_paths = list(sorted(glob(os.path.join(out_dir, "step", "*.step"))))
            if len(step_paths) > 1:
                # exclude last one, since it might be defect
                step_paths = step_paths[:-1]
                last_step_name = os.path.basename(step_paths[-1]).replace(".step", "")
                # count
                total_processed = int(last_step_name) + 1
                for step_path in tqdm(step_paths):
                    is_watertight, n_faces = check_solid(step_path)
                    total_watertight += is_watertight
                npz_paths = npz_paths[total_processed:]

        for npz_path in tqdm(npz_paths):
            npz_data = dict(np.load(npz_path))
            uvgrid = UvGrid.load_from_npz_data(npz_data)
            npz_name = os.path.basename(npz_path).replace(".npz", "")
            npz_names.append(npz_name)

            try:
                watertight, n_faces = batch_pp.process_one(
                    uvgrid, npz_name, timeout=max_pp_time
                )
                print(f"Processed item {npz_name}")
                total_processed += 1
                if watertight:
                    total_watertight += 1
                print(
                    f"Current stats - Total processed: {total_processed} \n"
                    f"Watertight: {total_watertight}, Watertight rate: {100 * total_watertight / total_processed:.2f}%\n"
                )

                if (n_test_watertight is not None) and (
                    total_watertight >= n_test_watertight
                ):
                    print(f"Reached target count of {n_test_watertight}")
                    break

            except RuntimeError:
                print(f"Skipped item {npz_name} due to timeout")
            except Exception as e:
                total_processed += 1
                print(f"Pping item {npz_name} failed")

    # check again the watertight meshes for sanity and
    step_dir = os.path.join(out_dir, "step")
    valid_cache_path = os.path.join(os.path.dirname(step_dir), "step_valid.pkl")
    cache_valid_step(
        valid_cache_path=valid_cache_path,
        step_dir=step_dir,
        n_test_watertight=n_test_watertight,
    )

    if test:
        test_step_files(
            step_dir=step_dir,
            n_test_watertight=n_test_watertight,
            use_cached_valid=True,
            dataset_name=dataset_name,
            max_n_prims=max_n_prims,
        )

    if plot:
        plot_pp_face_count_distribution(
            os.path.join(out_root_dir, "uvgrid"),
            cached_valid_path=valid_cache_path,
            out_path=os.path.join(out_dir, "pp_face_count_dist.png"),
            n_test_watertight=n_test_watertight,
            max_n_prims=max_n_prims,
        )


@app.command()
def save_brep(
    npz_path: str,
    grid_res: int = 256,
    grid_range: float = 1.1,
    use_fwn: bool = True,
    psr_occ_thresh: float = 0.5,
    partition_occ_thresh: float = 0.5,
    scaled_rasterization: bool = True,
    uvgrid_extension_len: float = 1.0,
):
    npz_data = dict(np.load(npz_path))
    uvgrid = UvGrid.load_from_npz_data(npz_data)

    pp = Postprocessor(
        uvgrid,
        grid_res,
        grid_range=grid_range,
        use_fwn=use_fwn,
        psr_occ_thresh=psr_occ_thresh,
        partition_occ_thresh=partition_occ_thresh,
        scaled_rasterization=scaled_rasterization,
        uvgrid_extension_len=uvgrid_extension_len,
    )
    brep = pp.get_brep(ret_none_if_failed=True)
    if brep is None:
        print("Brep processing failed")
        return
    step_out_path = npz_path.replace(".npz", ".step")
    write_step_file(brep, step_out_path)


@app.command()
def vis(
    npz_path: str,
    grid_res: int = 256,
    grid_range: float = 1.1,
    use_fwn: bool = True,
    psr_occ_thresh: float = 0.5,
    uvgrid_extension_len: float = 1.0,
    partition_occ_thresh: float = 0.5,
    scaled_rasterization: bool = True,
    smooth_extension: bool = True,
    ret_scaled: bool = True,
):
    """
    Visualize the postposrocessing of a single uvgrid (in npz)
    :param npz_path:
    :param grid_res:
    :param grid_range: range of meshgrid for psr occupancy
    :param ignore_mesh_slicing: when using the slicer, mesh is not affected
    :param use_fwn: use fast winding numbers, if not, use original generalized winding numbers
    :param psr_occ_thresh: poisson surface recon occupancy threshold
    :param partition_occ_thresh: occupancy threshold when voting with winding number partition
    :param scaled_rasterization: scale meshgrid to tightly cover the object bbox
    :param save_brep: export brep as step file
    :param ret_scaled: visualize with uvgrid scaled to fit bbox for better visualization
        and understanding of postprocessing
    :return:
    """
    npz_data = dict(np.load(npz_path))
    uvgrid = UvGrid.load_from_npz_data(npz_data)
    pp = Postprocessor(
        uvgrid,
        grid_res,
        grid_range=grid_range,
        use_fwn=use_fwn,
        psr_occ_thresh=psr_occ_thresh,
        partition_occ_thresh=partition_occ_thresh,
        scaled_rasterization=scaled_rasterization,
        uvgrid_extension_len=uvgrid_extension_len,
        smooth_extension=smooth_extension,
    )
    # scaled visualization is more explicit to visualize
    pp.vis_interactive(ret_scaled=ret_scaled)


if __name__ == "__main__":
    app()
