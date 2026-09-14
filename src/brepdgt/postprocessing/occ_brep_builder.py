"""
OCC brep builder
Code from BRepGen:
    https://github.com/samxuxiang/BrepGen/blob/main/utils.py
"""

import numpy as np
from typing import List
from OCC.Core.gp import gp_Pnt, gp_Pnt
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface, GeomAPI_PointsToBSpline
from OCC.Core.GeomAbs import GeomAbs_C2
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeEdge,
)
from OCC.Extend.TopologyUtils import TopologyExplorer, WireExplorer
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.gp import gp_Pnt
from OCC.Core.ShapeFix import ShapeFix_Face, ShapeFix_Wire, ShapeFix_Edge
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Wire
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid


def get_bbox_norm(point_cloud):
    # Find the minimum and maximum coordinates along each axis
    min_x = np.min(point_cloud[:, 0])
    max_x = np.max(point_cloud[:, 0])

    min_y = np.min(point_cloud[:, 1])
    max_y = np.max(point_cloud[:, 1])

    min_z = np.min(point_cloud[:, 2])
    max_z = np.max(point_cloud[:, 2])

    # Create the 3D bounding box using the min and max values
    min_point = np.array([min_x, min_y, min_z])
    max_point = np.array([max_x, max_y, max_z])
    return np.linalg.norm(max_point - min_point)


def add_pcurves_to_edges(face):
    edge_fixer = ShapeFix_Edge()
    top_exp = TopologyExplorer(face)
    for wire in top_exp.wires():
        wire_exp = WireExplorer(wire)
        for edge in wire_exp.ordered_edges():
            edge_fixer.FixAddPCurve(edge, face, False, 0.001)


def fix_wires(face, debug=False):
    top_exp = TopologyExplorer(face)
    for wire in top_exp.wires():
        if debug:
            wire_checker = ShapeAnalysis_Wire(wire, face, 0.01)
            print(f"Check order 3d {wire_checker.CheckOrder()}")
            print(f"Check 3d gaps {wire_checker.CheckGaps3d()}")
            print(f"Check closed {wire_checker.CheckClosed()}")
            print(f"Check connected {wire_checker.CheckConnected()}")
        wire_fixer = ShapeFix_Wire(wire, face, 0.01)

        # wire_fixer.SetClosedWireMode(True)
        # wire_fixer.SetFixConnectedMode(True)
        # wire_fixer.SetFixSeamMode(True)

        assert wire_fixer.IsReady()
        ok = wire_fixer.Perform()
        # assert ok


def fix_face(face):
    fixer = ShapeFix_Face(face)
    fixer.SetPrecision(0.01)
    fixer.SetMaxTolerance(0.1)
    ok = fixer.Perform()
    # assert ok
    fixer.FixOrientation()
    face = fixer.Face()
    return face


def construct_brep(
    surf_wcs: np.ndarray,
    edge_wcs: List[np.ndarray],
    face_edge_adj: List[List],
    edge_vertex_adj: List[List],
):
    """
    Fit parametric surfaces / curves and trim into B-rep
    :param surf_wcs: array of N x grid_res x grid_res x 3 uvgrid coordinates
    :param edge_wcs: list of n x 3  containing edge u coordinates
    :param face_edge_adj: List of size N, each containing list of edge idxs : Face-edge adjacency idx
    :param edge_vertex_adj: List of size M, each containing list of vertex idx (size of 2) : Edge-vertex adjacency idx

    """
    print("Building the B-rep...")
    # Fit surface bspline
    recon_faces = []
    for points in surf_wcs:
        num_u_points, num_v_points = points.shape[0], points.shape[1]
        uv_points_array = TColgp_Array2OfPnt(1, num_u_points, 1, num_v_points)
        for u_index in range(1, num_u_points + 1):
            for v_index in range(1, num_v_points + 1):
                pt = points[u_index - 1, v_index - 1]
                point_3d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
                uv_points_array.SetValue(u_index, v_index, point_3d)
        approx_face = GeomAPI_PointsToBSplineSurface(
            uv_points_array, 3, 8, GeomAbs_C2, 5e-2
        ).Surface()
        recon_faces.append(approx_face)

    recon_edges = []
    for edge_idx, points in enumerate(edge_wcs):
        num_u_points = points.shape[0]
        u_points_array = TColgp_Array1OfPnt(1, num_u_points)
        for u_index in range(1, num_u_points + 1):
            pt = points[u_index - 1]
            point_2d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
            u_points_array.SetValue(u_index, point_2d)
        try:
            approx_edge = GeomAPI_PointsToBSpline(
                u_points_array, 0, 8, GeomAbs_C2, 5e-3
            ).Curve()
        except Exception as e:
            print("high precision failed, trying mid precision...")
            try:
                approx_edge = GeomAPI_PointsToBSpline(
                    u_points_array, 0, 8, GeomAbs_C2, 8e-3
                ).Curve()
            except Exception as e:
                print("mid precision failed, trying low precision...")
                approx_edge = GeomAPI_PointsToBSpline(
                    u_points_array, 0, 8, GeomAbs_C2, 5e-2
                ).Curve()
        recon_edges.append(approx_edge)

    # Create edges from the curve list
    edge_list = []
    for curve in recon_edges:
        edge = BRepBuilderAPI_MakeEdge(curve).Edge()
        edge_list.append(edge)

    # Cut surface by wire
    post_faces = []
    post_edges = []
    for idx, (surface, edge_indices) in enumerate(zip(recon_faces, face_edge_adj)):
        corner_indices = edge_vertex_adj[edge_indices]

        # ordered loop
        loops = []
        ordered = [0]
        seen_corners = [corner_indices[0, 0], corner_indices[0, 1]]
        next_index = corner_indices[0, 1]

        while len(ordered) < len(corner_indices):
            while True:
                next_row = [
                    idx
                    for idx, edge in enumerate(corner_indices)
                    if next_index in edge and idx not in ordered
                ]
                if len(next_row) == 0:
                    break
                ordered += next_row
                next_index = list(set(corner_indices[next_row][0]) - set(seen_corners))
                if len(next_index) == 0:
                    break
                else:
                    next_index = next_index[0]
                seen_corners += [
                    corner_indices[next_row][0][0],
                    corner_indices[next_row][0][1],
                ]

            cur_len = int(
                np.array([len(x) for x in loops]).sum()
            )  # add to inner / outer loops
            loops.append(ordered[cur_len:])

            # Swith to next loop
            next_corner = list(set(np.arange(len(corner_indices))) - set(ordered))
            if len(next_corner) == 0:
                break
            else:
                next_corner = next_corner[0]
            next_index = corner_indices[next_corner][0]
            ordered += [next_corner]
            seen_corners += [
                corner_indices[next_corner][0],
                corner_indices[next_corner][1],
            ]
            next_index = corner_indices[next_corner][1]

        # Determine the outer loop by bounding box length (?)
        bbox_spans = []
        for l in loops:
            bbox_points = np.concatenate([edge_wcs[x] for x in l])
            bbox = get_bbox_norm(bbox_points)
            # bbox = get_bbox_norm(edge_wcs[l].reshape(-1, 3))
            bbox_spans.append(bbox)
        # bbox_spans = [get_bbox_norm(edge_wcs[x].reshape(-1, 3)) for x in loops]

        # Create wire from ordered edges
        _edge_indices_ = [edge_indices[x] for x in ordered]
        edge_post = [edge_list[x] for x in _edge_indices_]
        post_edges += edge_post

        out_idx = np.argmax(np.array(bbox_spans))
        inner_idx = list(set(np.arange(len(loops))) - set([out_idx]))

        # Outer wire
        wire_builder = BRepBuilderAPI_MakeWire()
        for edge_idx in loops[out_idx]:
            wire_builder.Add(edge_list[edge_indices[edge_idx]])
        outer_wire = wire_builder.Wire()

        # Inner wires
        inner_wires = []
        for idx in inner_idx:
            wire_builder = BRepBuilderAPI_MakeWire()
            for edge_idx in loops[idx]:
                wire_builder.Add(edge_list[edge_indices[edge_idx]])
            inner_wires.append(wire_builder.Wire())

        # Cut by wires
        face_builder = BRepBuilderAPI_MakeFace(surface, outer_wire)
        for wire in inner_wires:
            face_builder.Add(wire)
        face_occ = face_builder.Shape()
        fix_wires(face_occ)
        add_pcurves_to_edges(face_occ)
        fix_wires(face_occ)
        face_occ = fix_face(face_occ)
        post_faces.append(face_occ)

    # Sew faces into solid
    sewing = BRepBuilderAPI_Sewing()
    for face in post_faces:
        sewing.Add(face)

    # Perform the sewing operation
    sewing.Perform()
    sewn_shell = sewing.SewedShape()

    # Make a solid from the shell
    maker = BRepBuilderAPI_MakeSolid()
    maker.Add(sewn_shell)
    maker.Build()
    solid = maker.Solid()

    print("BRep built successfully")
    return solid
