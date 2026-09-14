import torch
import matplotlib
import subprocess
import signal
import sys

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import io
import PIL.Image
import trimesh
from PIL import Image, ImageDraw
import imageio
from typing import List, Dict, Union
from torchvision.transforms import ToTensor, Resize, ToPILImage
from brepdgt.primitives.uvgrid import UvGrid

cmap = plt.get_cmap("gist_rainbow")
COLORS = [cmap(i / 9) for i in range(10)]

_running_processes = []


def handle_interrupt(signum, frame):
    """Handle interrupt signal by terminating all running processes"""
    print("\nReceived interrupt signal. Terminating all rendering processes...")
    for process in _running_processes:
        try:
            process.terminate()
            process.wait(timeout=1)  # Give process a second to terminate gracefully
        except:
            try:
                process.kill()  # Force kill if not terminated
            except:
                pass  # Process might already be dead
    sys.exit(1)


# Register the interrupt handler
signal.signal(signal.SIGINT, handle_interrupt)


def save_and_render_uvgrid(
    save_dir: str,
    save_name: str,
    uvgrids: UvGrid,
    vis_idx: int,
    render_objects: Union[List[str], None] = None,
    render_blender: bool = True,
    use_cached_if_exists: bool = False,
) -> Union[Image.Image, None]:
    """
    Save npz and renders the uvgrid.
    Renders the 1. coordinates (with spheres) 2. coordinates + normal (with cones) 3. uv mesh
    :param save_dir: directory to save
    :param save_name: name for the .npz and .png files to be saved
    :param uvgrids: UVGrid with elements batched
    :param vis_idx: batch idx of the element to save and visualize
    :param use_cached_if_exists: use cached batch if the rendering with the same name exists
    :return:
        PIL image
    """
    if render_objects is None:
        # render everything
        if uvgrids.normal is not None:
            render_objects = ["coord", "coord_normal", "uv_mesh"]
        else:
            render_objects = ["coord", "uv_mesh"]
    npz_path = os.path.join(save_dir, f"{save_name}.npz")
    imgs = []
    for render_object in render_objects:
        img_path = os.path.join(save_dir, f"{save_name}_{render_object}.png")
        uvgrids.export_npz(file_path=npz_path, vis_idx=vis_idx)
        if render_blender:
            if not (use_cached_if_exists and os.path.exists(img_path)):
                cmd_render_blender(npz_path, img_path, render_object)
            img = PIL.Image.open(img_path)
            imgs.append(img)
    if render_blender:
        res = concat_h_pil(imgs)
        return res
    else:
        return None


def cmd_render_blender(npz_path: str, render_path: str, render_object: str):
    """Execute blender render command with proper process management"""
    try:
        cmd = [
            "./blender",
            "--background",
            "--python",
            "scripts/blender/render_uvgrid.py",
            "--",
            npz_path,
            render_path,
            render_object,
        ]

        process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        _running_processes.append(process)

        # Wait for process to complete
        process.wait()

        # Remove process from tracking list
        _running_processes.remove(process)

        # Check if process completed successfully
        if process.returncode != 0:
            print(f"Warning: Blender render failed for {render_path}")

    except Exception as e:
        print(f"Error in blender rendering: {e}")
        raise


def cmd_render_blender_pc(ply_path: str, render_path: str):
    """Execute blender PC render command with proper process management"""
    try:
        cmd = [
            "./blender",
            "--background",
            "--python",
            "scripts/blender/render_ply_pc.py",
            "--",
            ply_path,
            render_path,
            " > /dev/null",  # make it quite
        ]

        process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        _running_processes.append(process)

        # Wait for process to complete
        process.wait()

        # Remove process from tracking list
        _running_processes.remove(process)

        # Check if process completed successfully
        if process.returncode != 0:
            print(f"Warning: Blender PC render failed for {render_path}")

    except Exception as e:
        print(f"Error in blender PC rendering: {e}")
        raise


def render_pc(
    pc: np.ndarray, pc_out_path: str, render_path: str, use_cached_if_exists: bool
) -> PIL.Image.Image:
    exists = False
    if use_cached_if_exists:
        exists = os.path.exists(render_path)
    if not exists:
        pc = trimesh.PointCloud(pc)
        pc.export(pc_out_path)
        cmd_render_blender_pc(pc_out_path, render_path)
    img = PIL.Image.open(render_path)
    return img


def save_tensor_img(img, path):
    img = ToPILImage()(img)
    img.save(path)


def plt_to_pil(plt, h, w, clear=True):
    buf = io.BytesIO()
    plt.savefig(
        buf,
        dpi=np.mean([h, w]),
        format="png",
        bbox_inches="tight",
    )
    if clear:
        plt.clf()
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img
    img = Resize((h, w))(img)
    return ToTensor()(img)


def concat_h_pil(imgs):
    width_sum = 0
    for img in imgs:
        width_sum += img.width
    dst = PIL.Image.new("RGB", (width_sum, imgs[0].height))
    w = 0
    for img in imgs:
        dst.paste(img, (w, 0))
        w += img.width
    return dst


def concat_v_pil(imgs):
    height_sum = 0
    for img in imgs:
        height_sum += img.height
    dst = PIL.Image.new("RGB", (imgs[0].width, height_sum))
    h = 0
    for img in imgs:
        dst.paste(img, (0, h))
        h += img.height
    return dst


def save_vid_from_img_seq(
    save_path: str, img_seq: List[PIL.Image.Image], fps=10
) -> None:
    """

    :param save_path: path to save video
        Currently supports only mp4 file
    :param img_seq: List of images to cat as video
    :param fps: frame per second
    :return:
    """
    # Get the size of the images (assuming all images have the same size)
    width, height = img_seq[0].size

    img_seq = [np.array(x) for x in img_seq]
    # use image iio cause the codecs using open cv does not work for some reason
    writer = imageio.get_writer(save_path, fps=fps, codec="h264")
    for img in img_seq:
        writer.append_data(img)
    writer.close()


def to8b(img: np.ndarray) -> np.ndarray:
    return (255 * img).astype(np.uint8)


def get_cmap() -> np.ndarray:
    cmap = [
        (0.894, 0.102, 0.110),  # Red
        (0.216, 0.494, 0.722),  # Blue
        (0.302, 0.686, 0.290),  # Green
        (0.596, 0.306, 0.639),  # Purple
        (1.000, 0.498, 0.000),  # Orange
        (1.000, 1.000, 0.200),  # Yellow
        (0.651, 0.337, 0.157),  # Brown
        (0.969, 0.506, 0.749),  # Pink
        (0.600, 0.600, 0.600),  # Grey
        (0.090, 0.745, 0.812),  # Cyan
        (0.980, 0.506, 0.353),  # Salmon
        (0.850, 0.372, 0.007),  # Dark Orange
        (0.550, 0.090, 0.290),  # Burgundy
        (0.294, 0.000, 0.510),  # Indigo
        (0.628, 0.745, 0.245),  # Olive Green
        (0.976, 0.306, 0.906),  # Magenta
        (0.255, 0.412, 0.882),  # Royal Blue
        (0.329, 0.510, 0.208),  # Forest Green
        (0.890, 0.258, 0.447),  # Crimson
        (0.835, 0.150, 0.585),  # Hot Pink
    ]
    return np.array(cmap)
