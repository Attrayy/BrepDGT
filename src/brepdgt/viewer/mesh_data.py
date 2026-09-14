from typing import Optional, List
from dataclasses import dataclass


import numpy as np
from tqdm import tqdm

from brepdgt.primitives.uvgrid import UvGrid
from brepdgt.utils.vis import get_cmap


@dataclass
class MeshData:
    """Class to store pre-computed mesh data"""

    vertices: np.ndarray
    faces: np.ndarray
    colors: np.ndarray
    normals: Optional[np.ndarray] = None


def compute_mesh_from_uvgrid(uvgrid: UvGrid) -> MeshData:
    """Pre-compute mesh data from a UvGrid

    Args:
        uvgrid: UvGrid with shape [n_prims x grid_size x grid_size x 3]
               or [batch x n_prims x grid_size x grid_size x 3]
    Returns:
        MeshData containing combined mesh for all primitives
    """
    all_vertices = []
    all_faces = []
    all_colors = []
    vertex_offset = 0

    # Get batch size and n_prims based on coord shape
    is_batched = len(uvgrid.coord.shape) == 5
    batch_size = uvgrid.coord.shape[0] if is_batched else 1
    n_prims = uvgrid.coord.shape[1] if is_batched else uvgrid.coord.shape[0]
    cmap = get_cmap()

    # For each batch (usually just 1)
    for batch_idx in range(batch_size):
        # For each primitive
        for prim_idx in range(n_prims):
            # Check empty mask based on whether batched or not
            is_empty = (
                uvgrid.empty_mask[batch_idx][prim_idx]
                if is_batched
                else uvgrid.empty_mask[prim_idx]
            )

            if not is_empty:
                vertices, faces = uvgrid.meshify(
                    prim_idx,
                    use_grid_mask=False,
                    batch_idx=batch_idx if is_batched else None,
                )
                all_vertices.append(vertices)
                all_faces.append(faces + vertex_offset)
                all_colors.extend([cmap[prim_idx % len(cmap)]] * len(vertices))
                vertex_offset += len(vertices)

    if not all_vertices:  # Return None if no vertices
        return None

    all_vertices = np.vstack(all_vertices)
    all_faces = np.vstack(all_faces)
    all_colors = np.array(all_colors)

    return MeshData(
        vertices=all_vertices,
        faces=all_faces,
        colors=all_colors,
    )


def compute_mesh_trajectory(uvgrids_traj: List[UvGrid]) -> List[Optional[MeshData]]:
    """Pre-compute mesh data for entire trajectory"""
    mesh_traj = []
    for uvgrid in tqdm(uvgrids_traj, desc="Computing meshes"):
        mesh_data = compute_mesh_from_uvgrid(uvgrid)
        mesh_traj.append(mesh_data)
    return mesh_traj
