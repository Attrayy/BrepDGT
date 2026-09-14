from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import torch

import polyscope as ps
from brepdgt.primitives.uvgrid import UvGrid
from brepdgt.utils.vis import get_cmap


# Stores data necessary to render each uv grid
@dataclass
class UvGridPsData:
    # PC
    coords: np.ndarray
    single_color: (
        np.ndarray
    )  # (3,) single color vector to avoid empty array when everything is masked
    normals: Union[np.ndarray, None] = None
    # Mesh
    vertices: Union[np.ndarray, None] = None
    faces: Union[np.ndarray, None] = None


# Creates **N** UvGridPsData (one for each primitive)
def uv_grid_to_ps_data(uvgrids: UvGrid) -> List[UvGridPsData]:
    is_batched = len(uvgrids.coord.shape) == 5
    assert is_batched
    coord = uvgrids.coord[0] if is_batched else uvgrids.coord
    empty_mask = uvgrids.empty_mask[0] if is_batched else uvgrids.empty_mask
    grid_mask = uvgrids.grid_mask[0] if is_batched else uvgrids.grid_mask
    normal = uvgrids.normal[0] if is_batched else uvgrids.normal

    uv_grid_ps_data = []
    for i in range(len(coord)):

        uvgrid_coord = coord[i]
        uvgrid_normal = normal[i] if normal is not None else None
        uvgrid_grid_mask = grid_mask[i] if grid_mask is not None else None

        valid_uvgrid_coord = (
            uvgrid_coord[uvgrid_grid_mask].cpu().numpy()
            if uvgrid_grid_mask is not None
            else uvgrid_coord.reshape(-1, 3).cpu().numpy()
        )
        if uvgrid_normal is not None:
            valid_uvgrid_normal = (
                uvgrid_normal[uvgrid_grid_mask].cpu().numpy()
                if uvgrid_grid_mask is not None
                else uvgrid_normal.reshape(-1, 3).cpu().numpy()
            )
        else:
            valid_uvgrid_normal = None

        cmap = get_cmap()
        color = cmap[i % len(cmap)]

        # Meshes are only computer if it isn't empty
        if not empty_mask[i]:
            # vertices, faces = None, None
            vertices, faces = uvgrids.meshify(i, use_grid_mask=True, batch_idx=0)
        else:
            vertices, faces = None, None

        uv_grid_ps_data.append(
            UvGridPsData(
                coords=valid_uvgrid_coord,
                single_color=color,
                normals=valid_uvgrid_normal,
                vertices=vertices,
                faces=faces,
            )
        )

    return uv_grid_ps_data


def apply_transform(x: torch.Tensor, transform: Union[torch.Tensor, np.ndarray, None]):
    """
    Apply 4x4 transform matrix `transform` to `x`
    """

    if transform is None:
        return x

    if isinstance(transform, torch.Tensor):
        actual_transform = transform
    else:
        actual_transform = torch.tensor(transform).to(x)

    if x.shape[-1] == 3:
        transformed_pos = torch.cat(
            [
                x,
                torch.ones(
                    list(x.shape[:-1]) + [1],
                    device=x.device,
                ),
            ],
            dim=-1,
        )
    else:
        transformed_pos = x

    transformed_pos = torch.einsum("...j, ij->...i", transformed_pos, actual_transform)
    transformed_pos = transformed_pos[..., :3] / transformed_pos[..., 3][..., None]

    return transformed_pos[..., : x.shape[-1]]


ALL_FRAMES = {}

PC_RADIUS = 0.02
PC_NORMALS_RADIUS = 0.01
PC_NORMALS_LENGTH = 0.07


def ps_add_uv_grid(
    data: UvGridPsData,
    name: str,
    render_mesh: bool = False,
    normals_enabled: bool = False,
) -> Union[ps.PointCloud, ps.SurfaceMesh]:
    """Add a frame to the viewer"""

    # Remove existing frame with same name if it exists
    if name in ALL_FRAMES:
        if isinstance(ALL_FRAMES[name], ps.PointCloud) and ps.has_point_cloud(name):
            ps.remove_point_cloud(name)
        elif ps.has_surface_mesh(name):
            ps.remove_surface_mesh(name)

    if render_mesh and data.faces is not None:
        # Mesh visualization
        mesh = ps.register_surface_mesh(name, data.vertices, data.faces)
        mesh.set_color(data.single_color)
        ALL_FRAMES[name] = mesh
        return mesh
    else:
        # Point cloud visualization
        pc = ps.register_point_cloud(name, data.coords, radius=PC_RADIUS)
        pc.set_color(data.single_color)
        if data.normals is not None:
            pc.add_vector_quantity(
                "normals",
                data.normals,
                enabled=normals_enabled,
                color=data.single_color,
                radius=PC_NORMALS_RADIUS,
                length=PC_NORMALS_LENGTH,
            )
        ALL_FRAMES[name] = pc
        return pc
