import numpy as np
import polyscope as ps
from brepdgt.primitives.uvgrid import UvGrid


def visualize_uvgrid_ps(
    name: str,
    uvgrid: UvGrid,
    point_radius: float = 0.007,
    normal_length: float = 0.015,
    normal_radius: float = 0.0015,
):
    """
    Visualize UVGrid points and normals with polyscope.
    Each face will be colored differently.

    Args:
        uvgrid: UVGrid object to visualize
        point_radius: Radius of points in visualization
        normal_length: Length of normal vectors
        normal_radius: Radius of normal vector lines
    """
    # Create color map for different faces
    cmap = np.array(
        [
            [0.12, 0.47, 0.71],  # blue
            [0.20, 0.63, 0.17],  # green
            [0.89, 0.10, 0.11],  # red
            [0.84, 0.49, 0.00],  # orange
            [0.58, 0.40, 0.74],  # purple
            [0.55, 0.34, 0.29],  # brown
            [0.75, 0.75, 0.00],  # yellow
            [0.70, 0.70, 0.70],  # gray
        ]
    )

    # Collect valid points and normals
    uvgrid_coords, uvgrid_normals, uvgrid_coord_colors = [], [], []
    for i in range(len(uvgrid.coord)):
        uvgrid_coord = uvgrid.coord[i][~uvgrid.empty_mask[i]]
        uvgrid_normal = uvgrid.normal[i][~uvgrid.empty_mask[i]]
        uvgrid_grid_mask = uvgrid.grid_mask[i][~uvgrid.empty_mask[i]]
        valid_uvgrid_coord = uvgrid_coord[uvgrid_grid_mask].cpu().numpy()
        valid_uvgrid_normal = uvgrid_normal[uvgrid_grid_mask].cpu().numpy()
        uvgrid_coords.append(valid_uvgrid_coord)
        uvgrid_normals.append(valid_uvgrid_normal)
        color = cmap[i % len(cmap)].reshape(1, -1)
        uvgrid_coord_colors.append(color.repeat(valid_uvgrid_coord.shape[0], axis=0))

    # Concatenate all points and normals
    uvgrid_coords = np.concatenate(uvgrid_coords)
    uvgrid_normals = np.concatenate(uvgrid_normals)
    uvgrid_coord_colors = np.concatenate(uvgrid_coord_colors)

    # Register point cloud with polyscope
    ps_uvgrid_coord = ps.register_point_cloud(
        name,
        uvgrid_coords,
        radius=point_radius,
        point_render_mode="sphere",
    )

    # Add normals and colors
    ps_uvgrid_coord.add_vector_quantity(
        f"{name}_normals",
        uvgrid_normals,
        length=normal_length,
        radius=normal_radius,
        enabled=False,
    )
    ps_uvgrid_coord.add_color_quantity(
        f"{name}_face_idx", uvgrid_coord_colors, enabled=True
    )

    return ps_uvgrid_coord
