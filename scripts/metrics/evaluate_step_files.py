import typer
import os
import numpy as np
import trimesh
from glob import glob
from natsort import natsorted
from tqdm import tqdm
from OCC.Extend.DataExchange import write_stl_file, read_step_file
from brepdgt.utils.brep_checker import check_solid
from brepdgt.utils.common import load_pkl, save_pkl
from brepdgt.utils.common import save_pointcloud
from trimesh.sample import sample_surface
from scripts.metrics.test_pc_metrics import eval_pc
from brepdgt.metrics.fid import calculate_fid_given_npz_and_paths
from scripts.visualization.cat_imgs import cat_imgs_with_paths
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
from brepdgt.utils.brep_sampler import (
    get_n_faces_from_step_path,
    sample_points_from_step_path,
)

app = typer.Typer()


@app.command()
def plot_face_count_distribution(
    step_dir: str,
    cached_valid_path: str,
    out_path: str,
    n_test_watertight: int,
    max_n_prims: int,
):
    """Plot distribution of face counts comparing samples vs ground truth"""
    cached_valid = load_pkl(cached_valid_path)
    names, watertight_masks = cached_valid["name"], cached_valid["watertight_mask"]
    watertight_names = names[watertight_masks][:n_test_watertight]

    n_faces = []
    for name in tqdm(watertight_names):
        step_path = os.path.join(step_dir, name + ".step")
        try:
            n_face = get_n_faces_from_step_path(step_path)
        except Exception as e:
            print(f"{name} {e}")
            continue
        n_faces.append(n_face)

    # Get unique face counts and frequencies
    n_faces_unique, n_faces_cnts = np.unique(n_faces, return_counts=True)

    max_n_prims = max(max_n_prims, max(n_faces_unique))
    n_faces_freq = np.zeros(max_n_prims + 1)
    n_faces_freq[n_faces_unique] = n_faces_cnts

    # Plot side-by-side bar chart
    plt.figure(figsize=(12, 6))
    x = np.arange(max_n_prims + 1)
    width = 0.35

    plt.bar(
        x - width / 2,
        n_faces_freq,
        width,
        alpha=0.7,
    )

    plt.xlabel("Number of Faces")
    plt.ylabel("Frequency (%)")
    plt.title("Face Count Distribution of Generated STEP file")
    # plt.xticks(x, sorted(set(pp_success_unique) | set(gt_unique)))
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(out_path)
    plt.close()


def render_blender(mesh_path: str, render_path: str, theta: float, flip: bool = False):
    cmd = f"./blender --background --python scripts/blender/render_for_fid.py -- {mesh_path} {render_path} {theta} normalize z-axis-up > /dev/null"
    if flip:
        cmd += " flip"
    print(cmd)
    os.system(cmd)


def render_view(args):
    """Helper function for multiprocessing"""
    mesh_path, render_path, view_idx, n_views = args
    theta = 2 * np.pi * ((view_idx / n_views) + 1 / 8)
    flip = view_idx % 2 == 1
    render_blender(mesh_path, render_path, theta, flip)


def parallel_render_views(
    mesh_path: str, render_dir: str, name: str, n_views: int, n_processes: int = 4
):
    """Render multiple views in parallel"""
    if n_processes is None:
        n_processes = min(mp.cpu_count(), n_views)

    # Prepare arguments for each view
    render_args = []
    for view_idx in range(n_views):
        render_out_path = os.path.join(render_dir, name + f"_{view_idx}.png")
        if not os.path.exists(render_out_path):
            render_args.append((mesh_path, render_out_path, view_idx, n_views))

    # Only process views that don't exist yet
    if render_args:
        with mp.Pool(n_processes) as pool:
            pool.map(render_view, render_args)


def cache_valid_step(
    step_dir: str,
    valid_cache_path: str,
    n_test_watertight: int,
    n_views: int = 4,
    n_processes: int = 4,
):
    # watertight masks should be renamed to solid masks, but let's keep it for backward's compatibility
    names, watertight_masks, n_faces_list = [], [], []
    watertight_cnt = 0
    step_paths = list(natsorted(glob(os.path.join(step_dir, "*.step"))))
    mesh_dir = os.path.join(os.path.dirname(step_dir), "mesh")
    os.makedirs(mesh_dir, exist_ok=True)
    pc_dir = os.path.join(os.path.dirname(step_dir), "pc")
    os.makedirs(pc_dir, exist_ok=True)
    render_dir = os.path.join(os.path.dirname(step_dir), "render")
    os.makedirs(render_dir, exist_ok=True)
    brep_pc_dir = os.path.join(os.path.dirname(step_dir), "brep_pc")
    os.makedirs(brep_pc_dir, exist_ok=True)

    for step_path in tqdm(step_paths):
        name = os.path.basename(step_path).replace(".step", "")
        try:
            stl_out_path = os.path.join(mesh_dir, name + ".stl")
            is_watertight, n_faces = check_solid(
                step_path=step_path, stl_out_path=stl_out_path, timeout=20
            )

            if is_watertight:
                # Render all views in parallel
                parallel_render_views(
                    stl_out_path,
                    render_dir,
                    name,
                    n_views,
                    n_processes=n_processes,  # Will use all available CPUs by default
                )

                pc_out_path = os.path.join(pc_dir, name + ".ply")
                if not os.path.exists(pc_out_path):
                    mesh = trimesh.load_mesh(stl_out_path)
                    pc, _ = sample_surface(mesh, 2000)
                    save_pointcloud(pc_out_path, pc)

                brep_pc_out_path = os.path.join(brep_pc_dir, name + ".ply")
                if not os.path.exists(brep_pc_out_path):
                    brep_pc = sample_points_from_step_path(step_path, 5000)
                    save_pointcloud(brep_pc_out_path, brep_pc)
                watertight_cnt += 1
            names.append(name)
            watertight_masks.append(is_watertight)
            n_faces_list.append(n_faces)
        except KeyboardInterrupt:
            print("\nInterrupted. Cleaning up...")
            raise
        except Exception as e:
            names.append(name)
            watertight_masks.append(False)
            n_faces_list.append(0)
            continue

        if watertight_cnt >= n_test_watertight:
            print(f"Succesfully found {watertight_cnt} watertight steps!")
            break
        # print(f"{step_name}: valid: {is_valid}, watertight: {is_watertight}")
    if watertight_cnt < n_test_watertight:
        print(
            f"Found {watertight_cnt} watertight steps, expected at least {n_test_watertight}\n"
        )

    names = np.array(names)
    n_generated = int(names[watertight_masks][-1])

    # print watertight stats
    watertight_masks = np.array(watertight_masks)
    watertight_ratio = 100 * watertight_masks.astype(np.float32).sum() / n_generated
    print(f"Watertight cnt: {watertight_masks.sum()}/{n_generated}")
    print(f"Watertight ratio: {watertight_ratio:.2f}%")

    # save stats
    save_dict = {
        "name": names,
        "watertight_mask": watertight_masks,
        "n_faces": n_faces_list,
    }
    save_pkl(save_dict, valid_cache_path)

    txt_path = valid_cache_path.replace(".pkl", ".txt")
    with open(txt_path, "w") as f:
        for i in range(len(names)):
            f.write(
                f"{names[i]} {str(n_faces_list[i]).zfill(2)} {watertight_masks[i]}\n"
            )
    return save_dict


@app.command()
def test(
    step_dir: str,
    n_test_watertight: int = 3000,
    use_cached_valid: bool = True,
    dataset_name: str = "deepcad",
    max_n_prims: int = 30,
    n_views: int = 4,
    n_processes: int = 4,
):
    """
    Given a step directory, test metrics with first {n_test} watertight breps.
    :param step_dir:
    :param n_test_watertight: number of watertight step files to evaluate
    :return:
    """
    print(f"Getting metrics for dir {step_dir}")
    if step_dir[-1] == "/":
        step_dir = step_dir[:-1]
    valid_cache_path = os.path.join(os.path.dirname(step_dir), "step_valid.pkl")
    if use_cached_valid and os.path.exists(valid_cache_path):
        print("Using already cached validness")
        cached_valid = load_pkl(valid_cache_path)
    else:
        print("Checking validness...")
        cached_valid = cache_valid_step(
            step_dir=step_dir,
            valid_cache_path=valid_cache_path,
            n_test_watertight=n_test_watertight,
            n_views=n_views,
            n_processes=n_processes,
        )

    out_dir = os.path.dirname(step_dir)
    ################################
    # Plot face count distribution #
    ################################
    face_cnt_distrib_plot_path = os.path.join(out_dir, "gen_face_count.png")
    plot_face_count_distribution(
        step_dir=step_dir,
        cached_valid_path=valid_cache_path,
        out_path=face_cnt_distrib_plot_path,
        n_test_watertight=n_test_watertight,
        max_n_prims=max_n_prims,
    )

    ####################
    # Watertight ratio #
    ####################
    names, watertight_masks = (
        cached_valid["name"],
        cached_valid["watertight_mask"],
    )
    assert (
        watertight_masks.sum() >= n_test_watertight
    ), f"cache should have at least {n_test_watertight}, got: {watertight_masks.sum()}"

    watertight_names = names[watertight_masks][:n_test_watertight]
    n_generated = int(watertight_names[-1])

    watertight_ratio = 100 * n_test_watertight / n_generated
    metrics = {"watertight_ratio": watertight_ratio}

    ##############
    # PC Metrics #
    ##############
    pc_paths = [os.path.join(out_dir, "pc", f"{name}.ply") for name in watertight_names]

    # PC metrics using references provided by BrepGen
    pc_metrics = eval_pc(
        gen_pc_paths=pc_paths,
        real_pc_dir=f"./data/abc_processed/{dataset_name}_{max_n_prims}_test_pcd",
        n_iter=10,
        verbose=False,
        normalize="points_mean",
        n_samples=1000,
    )
    pc_metrics = {f"brepgen_normalization-{k}": v for k, v in pc_metrics.items()}
    metrics.update(pc_metrics)

    ###################
    # Brep PC Metrics #
    ###################
    brep_pc_paths = [
        os.path.join(out_dir, "brep_pc", f"{name}.ply") for name in watertight_names
    ]
    brep_metrics = eval_pc(
        gen_pc_paths=brep_pc_paths,
        real_pc_npz_path=f"./data/abc_processed/{dataset_name}_{max_n_prims}_test_brep_points.npz",
        n_iter=3,
        verbose=False,
        normalize="longest_axis",
    )
    brep_metrics = {f"brep-{k}": v for k, v in brep_metrics.items()}
    metrics.update(brep_metrics)

    #######
    # FID #
    #######
    render_dir = os.path.join(out_dir, "render")
    render_paths = []
    render_skip_cnt = 0
    for name in watertight_names:
        for view_idx in range(n_views):
            render_path = os.path.join(render_dir, name + f"_{view_idx}.png")
            if not os.path.exists(render_path):
                print(f"{render_path} does not exist! Skipping it for now")
                render_skip_cnt += 1
            else:
                render_paths.append(render_path)
    if render_skip_cnt != 0:
        print(f"WARNING {render_skip_cnt} renderings do not exist for FID")

    fid_vanilla = calculate_fid_given_npz_and_paths(
        npz_path=f"data/abc_processed/fid/fid_cache_vanilla_{dataset_name}_{max_n_prims}_{n_views}views_test.npz",
        paths=render_paths,
        batch_size=256,
        device="cuda",
        fid_type="vanilla",
        num_workers=16,
    )
    metrics["fid_vanilla"] = fid_vanilla

    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    metrics_path = os.path.join(out_dir, f"metrics_{n_test_watertight}.txt")
    with open(metrics_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k} : {v}\n")

    # render first 100 step files
    if len(render_paths[::n_views]) >= 400:
        img_out_path = os.path.join(out_dir, "first_400_samples.png")
        cat_imgs_with_paths(
            render_paths[::n_views], img_out_path=img_out_path, n_imgs=400
        )


@app.command()
def zfill_paths(in_dir: str, postfix: str = ".step"):
    """
    Zfills all the paths for proper sorting
    :param in_dir:
    :return:
    """
    paths = glob(os.path.join(in_dir, "*" + postfix))
    for path in tqdm(paths):
        name = os.path.basename(path).replace(postfix, "")
        zfill_path = os.path.join(in_dir, name.zfill(6) + postfix)
        os.rename(path, zfill_path)


@app.command()
def check_watertight_solid(step_path: str):
    is_valid_solid = check_solid(step_path)
    print(f"{step_path} solid {is_valid_solid}")


if __name__ == "__main__":
    app()
