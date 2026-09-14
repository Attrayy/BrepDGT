from typing import Tuple, Optional, List, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import torch
import trimesh
import tempfile
import os
from pathlib import Path
from occwl.solid import Solid
from occwl.uvgrid import uvgrid
from OCC.Core.TopoDS import TopoDS_Solid, TopoDS_Compound, TopoDS_CompSolid
from OCC.Extend.DataExchange import write_step_file
from brepdgt.primitives.uvgrid import UvGrid
from brepdgt.postprocessing.occ_wrapper import write_stl_with_timeout


def normalize_to_unit_cube(
    coords: np.ndarray,
    normals: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    """
    Normalize coordinates to fit within a unit cube [-1, 1].

    Args:
        coords: Input coordinates of shape (..., 3)
        normals: Optional normal vectors of shape (..., 3)

    Returns:
        - normalized_coords: Coordinates normalized to [-1, 1]
        - normalized_normals: Normalized normal vectors (if input provided)
        - scale: Scale factor used for normalization
        - center: Center point used for normalization
    """
    # Compute bounding box
    mins = np.min(coords.reshape(-1, 3), axis=0)
    maxs = np.max(coords.reshape(-1, 3), axis=0)

    # Compute center and scale
    center = (mins + maxs) / 2
    scale = np.max(maxs - mins) / 2  # Divide by 2 to map to [-1, 1]

    # Center and scale the coordinates
    normalized_coords = (coords - center) / scale

    # Scale normals if provided (only scale, no translation)
    # Note: we divide by scale to maintain the geometric relationship
    # This ensures normals remain perpendicular to the scaled surface
    normalized_normals = normals / scale if normals is not None else None

    # Re-normalize normal vectors to unit length
    if normalized_normals is not None:
        normalized_normals = normalized_normals / np.linalg.norm(
            normalized_normals, axis=-1, keepdims=True
        )

    return normalized_coords, normalized_normals, scale, center


def parse_solid(solid, num_u: int = 8, num_v: int = 8, normalize=True):
    """
    Parse the surface information needed for UvGrid creation from a CAD solid.

    Args:
    - solid (occwl.solid): A single brep solid in occwl data format.
    - num_u: u resolution (default: 8)
    - num_v: v resolution (default: 8)
    - normalize: normalize to fit in unit bbox

    Returns:
    - data: A dictionary containing coordinates, normals, and masks
    """
    assert isinstance(solid, (Solid, TopoDS_Solid, TopoDS_Compound, TopoDS_CompSolid))

    if isinstance(solid, TopoDS_Solid):
        solid = Solid(solid)
    elif isinstance(solid, (TopoDS_Compound, TopoDS_CompSolid)):
        solid = Solid(solid, allow_compound=True)
    else:
        raise ValueError(f"Unsupported solid type: {type(solid)}")

    # Split closed surface and closed curve to halve
    solid = solid.split_all_closed_faces(num_splits=0)
    solid = solid.split_all_closed_edges(num_splits=0)

    # Sample uv-grid from each face
    face_pnts, face_normals, face_masks = [], [], []
    for face in solid.faces():
        # Sample points and normals
        points = uvgrid(face, method="point", num_u=num_u, num_v=num_v)
        face_pnts.append(points)
        normals = uvgrid(face, method="normal", num_u=num_u, num_v=num_v)
        face_normals.append(normals)

        # Get visibility mask
        visibility_status = uvgrid(
            face, method="visibility_status", num_u=num_u, num_v=num_v
        )
        mask = np.logical_or(
            visibility_status == 0, visibility_status == 2
        )  # 0: Inside, 1: Outside, 2: On boundary
        face_masks.append(mask)

    # Stack all faces
    face_pnts = np.stack(face_pnts)  # N x n_u x n_v x 3
    face_normals = np.stack(face_normals)  # N x n_u x n_v x 3
    face_masks = np.stack(face_masks).squeeze(-1)  # N x n_u x n_v

    if normalize:
        # Normalize coordinates and normals to unit cube
        coords, normals, scale, center = normalize_to_unit_cube(face_pnts, face_normals)
    else:
        coords, normals = face_pnts, face_normals
        scale = np.array(1.0)
        center = np.zeros(3)

    return {
        "coords": coords.astype(np.float32),
        "normals": normals.astype(np.float32),
        "masks": face_masks,
        "scale": scale.astype(np.float32),
        "center": center.astype(np.float32),
    }


def uvgrid_to_brep(
    uvgrid: UvGrid,
    grid_res: int = 256,
    smooth_extension: bool = True,
    uvgrid_extension_len: float = 1.0,
    min_extension_len: float = 0.0,
    psr_occ_thresh: float = 0.5,
    verbose: bool = False,
) -> Tuple[object, bool]:
    """
    UVGrid to Brep conversion

    Args:
        uvgrid: Input UVGrid
        grid_res: Grid resolution for postprocessing (64 or 256)
        smooth_extension: Whether to extend uvgrid smoothly by averaging 3 nearest uvgrid directions
        uvgrid_extension_len: Length of the extended uvgrid (0.5 to 1.5)
        psr_occ_thresh: Occupancy threshold for PSR (0.3 to 0.7)
        verbose: Whether to print verbose output

    Returns:
        - brep: OpenCascade BRep object
        - is_watertight: Whether the BRep is watertight
    """
    from brepdgt.postprocessing.postprocessor import Postprocessor

    pp = Postprocessor(
        uvgrid=uvgrid,
        grid_res=grid_res,
        smooth_extension=smooth_extension,
        uvgrid_extension_len=uvgrid_extension_len,
        min_extend_len=min_extension_len,
        psr_occ_thresh=psr_occ_thresh,
    )

    # Get BRep
    brep = pp.get_brep(verbose=verbose)

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = os.path.join(tmp_dir, "out.step")
        write_step_file(brep, temp_path)

        # Check BRep validity
        from brepdgt.utils.brep_checker import check_solid

        is_watertight, _ = check_solid(temp_path)

    return brep, is_watertight


class PpSuccessState(Enum):
    # Failed
    FAILURE = "FAILURE"
    # Couldn't resolve into Brep => will output the patches
    PATCH_ONLY = "PATCH_ONLY"
    # Got the edges too
    PATCH_WITH_EDGES = "PATCH_WITH_EDGES"
    # Brep but non watertight
    NON_WATERTIGHT = "NON_WATERTIGHT"
    # Valid brep!
    SUCCESS = "SUCCESS"


class PpViewerOutput:
    """Data structure to store the output of pp no matter if it is valid or not!"""

    state: PpSuccessState = PpSuccessState.FAILURE

    patches: trimesh.Trimesh = None
    edges: List[np.ndarray] = None
    brep_occ: Any = None


def uvgrid_to_brep_or_mesh(
    uvgrid: UvGrid,
    grid_res: int = 256,
    smooth_extension: bool = True,
    uvgrid_extension_len: float = 1.0,
    min_extension_len: float = 0.0,
    psr_occ_thresh: float = 0.5,
    verbose: bool = False,
) -> Tuple[object, bool]:
    """
    UVGrid to Brep or Mesh

    Args:
        uvgrid: Input UVGrid
        grid_res: Grid resolution for postprocessing (64 or 256)
        smooth_extension: Whether to extend uvgrid smoothly by averaging 3 nearest uvgrid directions
        uvgrid_extension_len: Length of the extended uvgrid (0.5 to 1.5)

    Returns:
        - pp_viewer_output: PpViewerOutput object
        - is_watertight: Whether the BRep is watertight
    """
    from brepdgt.postprocessing.postprocessor import Postprocessor

    pp = Postprocessor(
        uvgrid=uvgrid,
        grid_res=grid_res,
        smooth_extension=smooth_extension,
        uvgrid_extension_len=uvgrid_extension_len,
        min_extend_len=min_extension_len,
        psr_occ_thresh=psr_occ_thresh,
    )

    # ==================
    # VANILLA PP
    # ==================

    if verbose:
        print("Starting postprocessing...")
    pp_out = pp.postprocess(ret_none_if_failed=False, verbose=verbose)

    # ==================
    # OCC PP
    # ==================

    brep = pp_out.brep
    brep_occ = None
    if brep is not None:

        surf_wcs = brep.faces.cpu().numpy()
        edge_wcs = [x.cpu().numpy() for x in brep.edges]
        face_edge_adj = brep.face_edge_adj
        edge_vertex_adj = brep.edge_vertex_adj.cpu().numpy()

        # requires occ to be installed
        from brepdgt.postprocessing.occ_brep_builder import construct_brep

        if verbose:
            print("Constructing BRep...")
        try:
            brep_occ = construct_brep(
                surf_wcs=surf_wcs,
                edge_wcs=edge_wcs,
                face_edge_adj=face_edge_adj,
                edge_vertex_adj=edge_vertex_adj,
            )
        except Exception as e:
            if verbose:
                print(f"Failed during BRep construction: {str(e)}")

    # ==================
    # WATERTIGHT TEST
    # ==================

    is_watertight = False
    if brep_occ is not None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = os.path.join(tmp_dir, "out.step")
            write_step_file(brep_occ, temp_path)

            # Check BRep validity
            from brepdgt.utils.brep_checker import check_solid

            is_watertight, _ = check_solid(temp_path)

    # ==================
    # RESOLVE OUTPUT
    # ==================

    pp_viewer_output = PpViewerOutput()

    # Elevate states
    if pp_out.patches is not None and pp_out.wrap_patch_mask is not None:
        patches = [
            patch for patch, mask in zip(pp_out.patches, pp_out.wrap_patch_mask) if mask
        ]
        pp_viewer_output.patches = patches
        pp_viewer_output.state = PpSuccessState.PATCH_ONLY

    if pp_out.brep is not None and pp_out.brep.edges is not None:
        pp_viewer_output.edges = [e.cpu().numpy() for e in pp_out.brep.edges]
        pp_viewer_output.state = PpSuccessState.PATCH_WITH_EDGES

    if brep_occ is not None:
        pp_viewer_output.brep_occ = brep_occ
        pp_viewer_output.state = (
            PpSuccessState.SUCCESS if is_watertight else PpSuccessState.NON_WATERTIGHT
        )

    return pp_viewer_output


def brep_to_mesh(
    brep: object, output_path: Optional[str] = None, timeout: int = 20
) -> trimesh.Trimesh:
    """
    Convert BRep to mesh and optionally save as STL

    Args:
        brep: OpenCascade BRep object
        output_path: Optional path to save STL file
        timeout: Timeout in seconds for STL writing

    Returns:
        mesh: Trimesh object of the BRep
    """
    # Create temporary file if no output path specified
    if output_path is None:
        import tempfile

        temp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
        output_path = temp.name
        temp.close()

    # Write STL with timeout
    write_stl_with_timeout(brep, output_path, timeout)

    # Load mesh
    mesh = trimesh.load_mesh(output_path)

    # Clean up temporary file if created
    if output_path == temp.name:
        Path(output_path).unlink()

    return mesh


def brep_to_uvgrid(
    solid: object, grid_size: int = 32, normalize: bool = True
) -> UvGrid:
    """
    Convert BRep to UVGrid by sampling points on each face

    Args:
        solid: OpenCascade BRep object
        grid_size: Size of UV grid (default: 32)

    Returns:
        uvgrid: UVGrid representation of the BRep
    """
    data = parse_solid(solid=solid, normalize=normalize)

    # Convert to tensors
    coords = torch.tensor(data["coords"], dtype=torch.float32)
    normals = torch.tensor(data["normals"], dtype=torch.float32)
    grid_masks = torch.tensor(data["masks"], dtype=torch.bool)
    empty_mask = torch.zeros(len(data["coords"]), dtype=torch.bool)

    # Create UVGrid
    uvgrid = UvGrid.from_raw_values(
        coord=coords,
        grid_mask=grid_masks,
        max_n_prims=len(data["coords"]),
        normal=normals,
    )

    return uvgrid
