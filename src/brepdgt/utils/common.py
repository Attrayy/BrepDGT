from typing import Any, List
from dataclasses import fields

import pickle
import json
import numpy as np
import torch

from plyfile import PlyData


def save_txt(obj, path):
    with open(path, "w") as f:
        for l in obj:
            f.write(f"{l}\n")


def load_txt(path):
    with open(path, "r") as f:
        data_list = [line.strip() for line in f.readlines()]
    return data_list


def load_pkl(path):
    """
    Load a .pkl object
    """
    file = open(path, "rb")
    return pickle.load(file)


def save_pkl(obj, path):
    """
    save a dictionary to a pickle file
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_json(path):
    file = open(path, "r")
    return json.load(file)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)


def load_pointcloud(ply):
    plydata = PlyData.read(ply)
    vertices = np.stack(
        [
            plydata["vertex"]["x"],
            plydata["vertex"]["y"],
            plydata["vertex"]["z"],
        ],
        axis=1,
    )
    if "nx" in plydata["vertex"]:
        normals = np.stack(
            [
                plydata["vertex"]["nx"],
                plydata["vertex"]["ny"],
                plydata["vertex"]["nz"],
            ],
            axis=1,
        )
        return vertices, normals
    return vertices


def load_grid(path):
    with open(path, "rb") as f:
        # Read the ASCII part of the header until the binary grid starts
        format_identifier = f.readline().decode("ascii").strip()  # e.g., G3
        data_type = f.readline().decode("ascii").strip()  # e.g., 1 FLOAT
        resolution_line = f.readline().decode("ascii").strip()  # e.g., "256 256 256"

        # Extract the resolution
        res_x, res_y, res_z = map(int, resolution_line.split())

        # Skip the transformation matrix
        T = []
        for _ in range(4):
            T.append(
                f.readline().decode("ascii").strip()
            )  # e.g., "0.01719 0 0 -1.091549"
        T = np.array([list(map(float, T[i].split())) for i in range(4)])

        # Read the binary grid data (float32)
        total_values = res_x * res_y * res_z
        grid_data = np.frombuffer(f.read(total_values * 4), dtype=np.float32)

    return grid_data.reshape((res_x, res_y, res_z)), T


def save_pointcloud(path, points, normals=None):
    # If points have extra dimensions, flatten them
    if len(points.shape) > 2:
        points = points[0]
        if normals is not None:
            normals = normals[0]

    # Convert tensors to numpy arrays if using PyTorch (optional)
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
        if normals is not None:
            normals = normals.detach().cpu().numpy()

    # Prepare the PLY header
    num_points = points.shape[0]
    has_normals = normals is not None
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {num_points}",
        "property float x",
        "property float y",
        "property float z",
    ]

    if has_normals:
        header.extend(["property float nx", "property float ny", "property float nz"])

    header.append("end_header")

    # Write the points and normals (if available) to the PLY file
    with open(path, "w") as ply_file:
        ply_file.write("\n".join(header) + "\n")

        for i in range(num_points):
            point_str = f"{points[i, 0]} {points[i, 1]} {points[i, 2]}"
            if has_normals:
                normal_str = f"{normals[i, 0]} {normals[i, 1]} {normals[i, 2]}"
                ply_file.write(f"{point_str} {normal_str}\n")
            else:
                ply_file.write(f"{point_str}\n")


def save_mesh(path, v, f):
    # Handle case where v or f have extra dimensions
    if len(v.shape) > 2:
        v, f = v[0], f[0]

    # Convert tensors to numpy arrays if using PyTorch
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
        f = f.detach().cpu().numpy()

    # Prepare the PLY header for the mesh
    num_vertices = v.shape[0]
    num_faces = f.shape[0]

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {num_vertices}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {num_faces}",
        "property list uchar int vertex_indices",
        "end_header",
    ]

    # Write the vertices and faces to the PLY file
    with open(path, "w") as ply_file:
        ply_file.write("\n".join(header) + "\n")

        # Write vertices
        for i in range(num_vertices):
            ply_file.write(f"{v[i, 0]} {v[i, 1]} {v[i, 2]}\n")

        # Write faces
        for i in range(num_faces):
            ply_file.write(f"3 {f[i, 0]} {f[i, 1]} {f[i, 2]}\n")


def uv_to_xyzc(coord, empty_mask, grid_mask):
    """
    Converts the uv grid to .xyzc format.

    Parameters:
    - coord: numpy array of shape (M, 8, 8, 3) containing 3D coordinates per face.
    - grid_mask: numpy array of shape (M, 8, 8), a binary mask where 1 means to include the coordinate, and 0 means to skip.

    Returns:
    - xyzc_array: numpy array of shape (N, 4) where N is the number of valid points and columns represent x, y, z, and surface index c.
    """
    coord = coord[~empty_mask]
    grid_mask = grid_mask[~empty_mask]

    num_faces, grid_x, grid_y, _ = coord.shape
    xyzc_list = []

    # Loop over each face (c is the surface index)
    for face_idx in range(num_faces):
        # Loop through the grid points
        for i in range(grid_x):
            for j in range(grid_y):
                # Only include the coordinate if the mask is False
                if grid_mask[face_idx, i, j] == True:
                    x, y, z = coord[face_idx, i, j]
                    xyzc_list.append([x, y, z, face_idx])

    # Convert the list to a NumPy array
    xyzc_array = np.array(xyzc_list)
    return xyzc_array


def save_xyzc(path, xyzc_array):
    """
    Saves the (N, 4) .xyzc array to a file.

    Parameters:
    - path: string, the path to save the .xyzc file.
    - xyzc_array: numpy array of shape (N, 4) where columns represent x, y, z, and surface index c.
    """
    # Save the array to a file in the format: x y z c
    np.savetxt(path, xyzc_array, fmt="%.6f %.6f %.6f %d")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def collate_dataclass_fn(batch: List[Any]) -> Any:
    # convert list of States to State of list
    batch = type(batch[0])(
        **{
            field.name: [getattr(d, field.name) for d in batch]
            for field in fields(batch[0])
        }
    )
    return batch


def downsample(x: torch.tensor, num_sample: int, deterministic=False) -> torch.tensor:
    """
    Args:
        x: torch tensor of N x d
        num_sample: number of samples to downsample to
        deterministic: whether the procedure is deterministic
    Returns:
        downsampled output of torch tensor {sample_num} x d
    """
    if x.shape[0] == 0:
        return torch.zeros(num_sample, 3, device=x.device)

    if x.shape[0] < num_sample:
        multiplier = int(num_sample) // x.shape[0]
        x_multiply = torch.cat((x,) * multiplier, dim=0)
        num_sample -= multiplier * x.shape[0]
        return torch.cat([downsample(x, num_sample, deterministic), x_multiply], dim=0)

    rand_idx = torch.arange(x.shape[0]) if deterministic else torch.randperm(x.shape[0])
    keep_idx = rand_idx[:num_sample]
    return x[keep_idx, :]


def clamp_norm(x: torch.Tensor, min_norm: float):
    """
    Given x of shape B x ... x x_dim, rescale each vector (x_dim) to have at least norm of min_norm
    :param x:
    :param min_norm:
    :return:
    """
    # Don't want numerical problems
    if min_norm < 1e-5:
        return x
    x_norm = torch.norm(x, dim=-1, keepdim=True)
    x_rescale = torch.where(
        x_norm < min_norm,
        min_norm / x_norm,
        torch.ones_like(x_norm),
    )
    return x * x_rescale


import signal


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Function timed out!")


def timeout_wrapper(func):
    def wrapped(*args, timeout=30, **kwargs):
        # Set the timeout handler for the SIGALRM signal
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)  # Start the countdown
        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # Cancel the alarm if the function finishes in time
            return result
        except TimeoutException:
            print(f"Function timed out after {timeout} seconds!")
            return None
        finally:
            signal.alarm(0)  # Ensure the alarm is canceled

    return wrapped
