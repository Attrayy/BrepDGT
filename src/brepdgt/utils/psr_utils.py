import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import griddata


def psr_grid_as_pointcloud(psr_grid, grid_resolution, threshold=0.5):
    """
    Convert PSR grid as a point cloud PLY file for visualization.

    Args:
        psr_grid (np.ndarray): PSR grid (shape: [D, H, W]).
        grid_resolution (tuple): The resolution of the grid (D, H, W).
        threshold (float): Threshold for determining occupancy (e.g., 0.0).

    Returns:
        coords (np.ndarray): The occupied grid coordinates
    """

    # Binarize the PSR grid based on the threshold (occupied if value > threshold)
    psr_grid = psr_grid
    occupancy_mask = psr_grid > threshold

    # Get the indices of the occupied voxels
    occupied_indices = np.argwhere(occupancy_mask)

    # Normalize the occupied indices to get the 3D coordinates
    D, H, W = grid_resolution
    coords = (
        occupied_indices / np.array([D, H, W]) * 2.2 - 1.1
    )  # Normalize to [-1.1, 1.1]
    return coords


def calculate_occupancy_accuracy_from_grids(gt_grid, est_grid, threshold=0.5):
    """
    Calculate detailed occupancy accuracy metrics between ground truth PSR grid and optimized PSR grid.

    Args:
        gt_grid (torch.Tensor): Ground truth PSR grid (shape: [D, H, W]).
        est_grid (torch.Tensor): Estimated PSR grid (shape: [D, H, W]).
        threshold (float): Threshold for occupancy (default is 0.5).

    Returns:
        dict: A dictionary containing occupancy accuracy metrics.
    """

    # Binarize the grids using the threshold
    gt_occupancy = gt_grid > threshold
    opt_occupancy = est_grid > threshold

    # Calculate TP, TN, FP, FN
    true_positives = (
        (gt_occupancy == 1) & (opt_occupancy == 1)
    ).sum()  # Inside in both
    true_negatives = (
        (gt_occupancy == 0) & (opt_occupancy == 0)
    ).sum()  # Outside in both
    false_positives = (
        (gt_occupancy == 0) & (opt_occupancy == 1)
    ).sum()  # Outside in GT, inside in est
    false_negatives = (
        (gt_occupancy == 1) & (opt_occupancy == 0)
    ).sum()  # Inside in GT, outside in est

    # Total number of points in the grid
    total_points = gt_grid.size

    # Calculate metrics
    occupancy_accuracy = (true_positives + true_negatives) / total_points * 100.0
    precision = true_positives / (true_positives + false_positives + 1e-6) * 100.0
    recall = true_positives / (true_positives + false_negatives + 1e-6) * 100.0
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)  # F1 score

    metrics = {
        "occupancy_accuracy": occupancy_accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1_score": f1_score.item(),
        "true_positives": true_positives.item(),
        "true_negatives": true_negatives.item(),
        "false_positives": false_positives.item(),
        "false_negatives": false_negatives.item(),
        "total_points": total_points,
    }

    return metrics


def interpolate_uv_grid(uv_coords, uv_normals, uv_mask, target_size):
    """
    Interpolates UV grid of coordinates, normals, and binary mask to a higher resolution.

    Parameters:
    - uv_coords: numpy array of shape (N, 8, 8, 3), original UV coordinates.
    - uv_normals: numpy array of shape (N, 8, 8, 3), original UV normals.
    - uv_mask: numpy array of shape (N, 8, 8), binary mask representing valid points.
    - target_size: int, desired resolution for the UV grid (e.g., 16 for 16x16 grid).

    Returns:
    - new_coords: numpy array of shape (N, target_size, target_size, 3), interpolated UV coordinates.
    - new_normals: numpy array of shape (N, target_size, target_size, 3), interpolated UV normals.
    - new_mask: numpy array of shape (N, target_size, target_size), interpolated binary mask (0 or 1 values).
    """
    num_faces, orig_size, _, dim = uv_coords.shape

    # Create a normalized 2D grid for the original and target UV grid sizes
    uv_orig_grid = np.linspace(0, 1, orig_size)
    uv_target_grid = np.linspace(0, 1, target_size)
    uv_orig_meshgrid = np.array(np.meshgrid(uv_orig_grid, uv_orig_grid)).T.reshape(
        -1, 2
    )  # shape: (64, 2)
    uv_target_meshgrid = np.array(
        np.meshgrid(uv_target_grid, uv_target_grid)
    ).T.reshape(
        -1, 2
    )  # shape: (target_size**2, 2)

    # Prepare containers for the new coordinates, normals, and mask
    new_coords = np.zeros((num_faces, target_size, target_size, dim))
    new_normals = np.zeros((num_faces, target_size, target_size, dim))
    new_mask = np.zeros((num_faces, target_size, target_size), dtype=int)

    for i in range(num_faces):
        # Flatten the 8x8 grids to apply grid interpolation
        face_coords_flat = uv_coords[i].reshape(-1, dim)  # shape: (64, 3)
        face_normals_flat = uv_normals[i].reshape(-1, dim)  # shape: (64, 3)
        face_mask_flat = uv_mask[i].reshape(-1)  # shape: (64,)

        # Interpolate the coordinates and normals using cubic interpolation
        new_face_coords = griddata(
            uv_orig_meshgrid, face_coords_flat, uv_target_meshgrid, method="cubic"
        )
        new_face_normals = griddata(
            uv_orig_meshgrid, face_normals_flat, uv_target_meshgrid, method="cubic"
        )

        # Interpolate the binary mask using nearest-neighbor interpolation
        new_face_mask = griddata(
            uv_orig_meshgrid, face_mask_flat, uv_target_meshgrid, method="nearest"
        )

        # Reshape back to the target_size x target_size grid
        new_coords[i] = new_face_coords.reshape(target_size, target_size, dim)
        new_normals[i] = new_face_normals.reshape(target_size, target_size, dim)
        new_mask[i] = new_face_mask.reshape(target_size, target_size)

    return new_coords, new_normals, new_mask.astype(bool)


def plot_slices(grid, resolution):
    """
    Function to plot three slices of a 3D grid and display them as a single concatenated figure.

    Parameters:
    - grid: 3D numpy array representing the implicit function
    - resolution: the resolution of the grid (assumed to be the size of the grid along each axis)
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # First slice (XY plane at middle Z)
    axs[0].imshow(grid[int(resolution / 2)], cmap="gray")
    axs[0].set_title("Slice in XY plane (Z mid)")
    fig.colorbar(axs[0].images[0], ax=axs[0])

    # Second slice (XZ plane at middle Y)
    axs[1].imshow(grid[:, int(resolution / 2)], cmap="gray")
    axs[1].set_title("Slice in XZ plane (Y mid)")
    fig.colorbar(axs[1].images[0], ax=axs[1])

    # Third slice (YZ plane at middle X)
    axs[2].imshow(grid[:, :, int(resolution / 2)], cmap="gray")
    axs[2].set_title("Slice in YZ plane (X mid)")
    fig.colorbar(axs[2].images[0], ax=axs[2])

    plt.suptitle("Slices of the Implicit Function")
    plt.show()
