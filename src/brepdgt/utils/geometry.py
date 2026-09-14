import numpy as np
import torch
import torch.nn.functional as F


def apply_transformation_matrix(x: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    single_point = False
    if len(x.shape) == 1:
        single_point = True
        x = x.reshape(1, -1)
    y = x @ matrix.T
    if single_point:
        y = y.reshape(-1)
    return y


def get_rotate2d_matrix(angle: float) -> np.ndarray:
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    return rotation_matrix.astype(np.float32)


def get_flip_y_matrix() -> np.ndarray:
    flip_matrix = np.array([[-1, 0], [0, 1]])
    return flip_matrix.astype(np.float32)


def compute_normals_from_grids(coords):
    """
    Compute normals per grid point from coordinates for all grid points, including edges.
    :param coords: Tensor of shape B x n_prims x n_grid x n_grid x 3
    :return: normals: Tensor of shape B x n_prims x n_grid x n_grid x 3
    """
    B, n_prims, n_grid, _, _ = coords.shape

    # Initialize du and dv tensors
    du = torch.zeros_like(coords)
    dv = torch.zeros_like(coords)

    # Compute du (derivative along u axis)
    # For interior points, use central differences
    du[:, :, 1:-1, :, :] = (coords[:, :, 2:, :, :] - coords[:, :, :-2, :, :]) / 2
    # For edges, use forward and backward differences
    du[:, :, 0, :, :] = coords[:, :, 1, :, :] - coords[:, :, 0, :, :]
    du[:, :, -1, :, :] = coords[:, :, -1, :, :] - coords[:, :, -2, :, :]

    # Compute dv (derivative along v axis)
    # For interior points, use central differences
    dv[:, :, :, 1:-1, :] = (coords[:, :, :, 2:, :] - coords[:, :, :, :-2, :]) / 2
    # For edges, use forward and backward differences
    dv[:, :, :, 0, :] = coords[:, :, :, 1, :] - coords[:, :, :, 0, :]
    dv[:, :, :, -1, :] = coords[:, :, :, -1, :] - coords[:, :, :, -2, :]

    # Compute normals as cross product of du and dv
    normals = torch.cross(du, dv, dim=-1)

    # Normalize normals to unit vectors
    normals = F.normalize(normals, p=2, dim=-1)

    return normals


def adjust_normals_direction(normals_per_point, generated_normals_at_center):
    """
    Adjust normals per grid point to have consistent direction with the generated normal at the center coordinate
    :param normals_per_point: Tensor of shape B x n_prims x n_grid x n_grid x 3
    :param generated_normals_at_center: Tensor of shape B x n_prims x 3
    :return: normals_per_point: Adjusted normals per grid point
    """
    n_grid = normals_per_point.shape[2]
    n_center = n_grid // 2

    # Extract the computed normal at the center coordinate
    computed_normals_at_center = normals_per_point[
        :, :, n_center, n_center, :
    ]  # Shape: B x n_prims x 3

    # Compute dot product between computed center normal and generated center normal
    dot_product = (computed_normals_at_center * generated_normals_at_center).sum(
        dim=-1, keepdim=True
    )  # Shape: B x n_prims x 1

    # Expand dot_product to match normals_per_point dimensions
    dot_product_expanded = dot_product[
        :, :, None, None, :
    ]  # Shape: B x n_prims x 1 x 1 x 1

    # Flip normals where dot product is negative
    normals_per_point = torch.where(
        dot_product_expanded >= 0, normals_per_point, -normals_per_point
    )

    return normals_per_point
