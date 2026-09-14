"""
Extracts points from a step file for each uvgrid.
Code from BrepGen https://github.com/samxuxiang/BrepGen

"""

import numpy as np
from occwl.uvgrid import ugrid, uvgrid
from occwl.compound import Compound
from occwl.solid import Solid
from occwl.shell import Shell
from occwl.entity_mapper import EntityMapper
from occwl.io import load_step
import shutup

shutup.mute_warnings()


def face_edge_adj(shape):
    """
    *** COPY AND MODIFIED FROM THE ORIGINAL OCCWL SOURCE CODE ***
    Extract face/edge geometry and create a face-edge adjacency
    graph from the given shape (Solid or Compound)

    Args:
    - shape (Shell, Solid, or Compound): Shape

    Returns:
    - face_dict: Dictionary of occwl faces, with face ID as the key
    - edge_dict: Dictionary of occwl edges, with edge ID as the key
    - edgeFace_IncM: Edge ID as the key, Adjacent faces ID as the value
    """
    assert isinstance(shape, (Shell, Solid, Compound))
    mapper = EntityMapper(shape)

    ### Faces ###
    face_dict = {}
    for face in shape.faces():
        face_idx = mapper.face_index(face)
        face_dict[face_idx] = (face.surface_type(), face)

    ### Edges and IncidenceMat ###
    edgeFace_IncM = {}
    edge_dict = {}
    for edge in shape.edges():
        if not edge.has_curve():
            continue

        connected_faces = list(shape.faces_from_edge(edge))
        if (
            len(connected_faces) == 2
            and not edge.seam(connected_faces[0])
            and not edge.seam(connected_faces[1])
        ):
            left_face, right_face = edge.find_left_and_right_faces(connected_faces)
            if left_face is None or right_face is None:
                continue
            edge_idx = mapper.edge_index(edge)
            edge_dict[edge_idx] = edge
            left_index = mapper.face_index(left_face)
            right_index = mapper.face_index(right_face)

            if edge_idx in edgeFace_IncM:
                edgeFace_IncM[edge_idx] += [left_index, right_index]
            else:
                edgeFace_IncM[edge_idx] = [left_index, right_index]
        else:
            pass  # ignore seam

    return face_dict, edge_dict, edgeFace_IncM


def update_mapping(data_dict):
    """
    Remove unused key index from data dictionary.
    """
    dict_new = {}
    mapping = {}
    max_idx = max(data_dict.keys())
    skipped_indices = np.array(
        sorted(list(set(np.arange(max_idx)) - set(data_dict.keys())))
    )
    for idx, value in data_dict.items():
        skips = (skipped_indices < idx).sum()
        idx_new = idx - skips
        dict_new[idx_new] = value
        mapping[idx] = idx_new


def extract_pts(solid: Solid, num_u: int = 64, num_v: int = 64, n_pts: int = 2000):
    """
    Extract all primitive information from splitted solid

    Args:
    - solid (occwl.Solid): A single b-rep solid in occwl format
    - num_u: u resolution
    - num_v: v resolution
    - n_pts: number of points to sample

    Returns:
    - face_pnts (N x 32 x 32 x 3): Sampled uv-grid points on the bounded surface region (face)
    - edge_pnts (M x 32 x 3): Sampled u-grid points on the boundged curve region (edge)
    - edge_corner_pnts (M x 2 x 3): Start & end vertices per edge
    - edgeFace_IncM (M x 2): Edge-Face incident matrix, every edge is connect to two face IDs
    - faceEdge_IncM: A list of N sublist, where each sublist represents the adjacent edge IDs to a face
    """
    assert isinstance(solid, Solid)

    # Split closed surface and closed curve to halve
    solid = solid.split_all_closed_faces(num_splits=0)
    solid = solid.split_all_closed_edges(num_splits=0)

    # Retrieve face, edge geometry and face-edge adjacency
    face_dict, edge_dict, edgeFace_IncM = face_edge_adj(solid)

    # # Skip unused index key, and update the adj
    # face_dict, face_map = update_mapping(face_dict)
    # edge_dict, edge_map = update_mapping(edge_dict)
    # edgeFace_IncM_update = {}
    # for key, value in edgeFace_IncM.items():
    #     new_face_indices = [face_map[x] for x in value]
    #     edgeFace_IncM_update[edge_map[key]] = new_face_indices
    # edgeFace_IncM = edgeFace_IncM_update

    # # Face-edge adj
    # num_faces = len(face_dict)
    # edgeFace_IncM = np.stack([x for x in edgeFace_IncM.values()])
    # faceEdge_IncM = []
    # for surf_idx in range(num_faces):
    #     surf_edges, _ = np.where(edgeFace_IncM == surf_idx)
    #     faceEdge_IncM.append(surf_edges)

    # Sample uv-grid from surface (32x32)
    face_pts, pts_weights = [], []
    for face_idx, face_feature in face_dict.items():
        _, face = face_feature

        pts = uvgrid(face, method="point", num_u=num_u, num_v=num_v)
        visibility_status = uvgrid(
            face, method="visibility_status", num_u=num_u, num_v=num_v
        )
        mask = np.logical_or(
            visibility_status == 0, visibility_status == 2
        )  # 0: Inside, 1: Outside, 2: On boundary
        if mask.sum() == 0:
            # cannot sample points from the uvgrid
            continue

        pts = pts[mask.reshape(num_u, num_v)]
        face_pts.append(pts)  # n x 3
        pts_weight = np.ones(pts.shape[0]) / pts.shape[0]
        pts_weights.append(pts_weight)

    face_pts = np.concatenate(face_pts)
    pts_weights = np.concatenate(pts_weights) / len(pts_weights)

    sample_idxs = np.random.choice(face_pts.shape[0], size=n_pts, p=pts_weights)
    sample = face_pts[sample_idxs]
    return sample


def sample_points_from_step_path(step_path: str, n_pts: int):
    """
    Returns uniformly sampled points w.r.t. number of faces from a step file
    """
    cad_solid = load_step(step_path)

    if len(cad_solid) != 1:
        raise ValueError("Multi solids not allowed")

    pts = extract_pts(cad_solid[0], n_pts=n_pts)
    return pts


def get_n_faces_from_step_path(step_path: str):
    """
    Returns number of faces from a step file
    """
    solid = load_step(step_path)

    if len(solid) != 1:
        raise ValueError("Multi solids not allowed")
    solid = solid[0]

    # Split closed surface and closed curve to halve
    solid = solid.split_all_closed_faces(num_splits=0)
    solid = solid.split_all_closed_edges(num_splits=0)

    # Retrieve face, edge geometry and face-edge adjacency
    face_dict, edge_dict, edgeFace_IncM = face_edge_adj(solid)
    return len(face_dict)


if __name__ == "__main__":
    from OCC.Extend.DataExchange import write_stl_file, read_step_file
    import point_cloud_utils as pcu
    import trimesh
    from trimesh.sample import sample_surface
    from brepdgt.utils.common import save_pointcloud

    step_path = "./data/steps/00000178_b21aa794b6b64d87a57c245e_step_008.step"
    pts = sample_points_from_step_path(step_path, 5000)
    pcu.save_mesh_v("out_pc_brep.ply", pts)

    brep = read_step_file(step_path)
    stl_out_path = "out_mesh.stl"
    write_stl_file(brep, stl_out_path)
    mesh = trimesh.load_mesh(stl_out_path)
    pc_out, _ = sample_surface(mesh, 5000)
    pc_out_path = "out_pc_mesh.ply"
    save_pointcloud(pc_out_path, pc_out)
