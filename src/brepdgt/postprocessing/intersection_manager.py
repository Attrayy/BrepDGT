"""
Some part of code is taken from point2cad:
    https://github.com/YujiaLiu76/point2cad/blob/81e15bfa952aee62cf06cdf4b0897c552fe4fb3a/point2cad/io_utils.py#L37
"""

from typing import List, Union, Tuple
from typing_extensions import Self
from torch_scatter import scatter
from brepdgt.primitives.uvgrid import UvGrid
import numpy as np
import pymesh
import trimesh
import torch
import networkx as nx
from dataclasses import dataclass
from collections import defaultdict


class BrepEdge:
    def __init__(self, edge_idxs: torch.Tensor, uvgrid_idxs: List):
        self.edge_idxs = edge_idxs
        self.uvgrid_idxs = list(sorted(uvgrid_idxs))
        # uvgrid touches brep edges
        # ideally, this should be same as uvgrid_idxs, where each adjacent uvgrid touches edge twice
        self.touched_uvgrid_idxs = []
        self.touchable = True

    def touch(self, uvgrid_idx: int):
        self.touched_uvgrid_idxs.append(uvgrid_idx)

    def is_same(self, oth: Self) -> bool:
        # quick cheap tests
        if self.edge_idxs.shape[0] != oth.edge_idxs.shape[0]:
            return False
        if self.uvgrid_idxs != oth.uvgrid_idxs:
            return False
        # check if all the edge_idxs are same
        uniques, cnts = torch.unique(
            torch.cat([self.edge_idxs, oth.edge_idxs]), return_counts=True
        )
        same = torch.sum(cnts != 2) == 0
        return same

    def touch_cnt(self):
        return len(self.touched_uvgrid_idxs)


class Cycle:
    def __init__(self, uvgrid_idx: int, brep_edges: List[BrepEdge]):
        """
        :param uvgrid_idx: index of the uvgrid that this cycle belongs to
        :param brep_edges: brep edges that form cycle
        """
        self.uvgrid_idx = uvgrid_idx
        self.brep_edges = brep_edges
        # check adjacent uvgrids for the cycle
        adjacent_uvgrid_idxs = []
        for be in brep_edges:
            if self.uvgrid_idx == be.uvgrid_idxs[0]:
                adjacent_uvgrid_idxs.append(be.uvgrid_idxs[1])
            else:
                adjacent_uvgrid_idxs.append(be.uvgrid_idxs[0])
        self.adjacent_uvgrid_idxs = list(set(adjacent_uvgrid_idxs))


@dataclass
class BRep:
    faces: torch.Tensor
    edges: List[torch.Tensor]
    vertices: torch.Tensor
    face_edge_adj: List[List]
    edge_vertex_adj: torch.Tensor


class IntersectionManager:
    def __init__(self, uvgrid: UvGrid):
        """
        :param uvgrid: (extended uvgrid) that partitions the space
        """
        # points in brep edge that have distance below this value will be merged as single point
        self.brep_edge_tol = 0.0
        # duplicate vertices tolerance when merging and segmenting meshes
        self.duplicate_vertices_tol = 1e-6

        self.uvgrid = uvgrid
        self.device = self.uvgrid.coord.device
        self.merged_mesh, self.face2uvgrid = self.merge_uvgrids(
            uvgrid, append_unit_cube=False
        )
        self.face2patch, self.defect_edges = self.get_face2patch(
            self.merged_mesh, self.face2uvgrid
        )
        # face_adj: adjacency matrix between faces
        # face2patch: map face_idx to patch_idx
        self.n_uvgrid = self.uvgrid.coord.shape[0]
        self.n_patches = self.face2patch.max().item() + 1

        # construct patch2uvgrid
        self.patch2uvgrid = -torch.ones(
            self.n_patches, device=self.device, dtype=torch.long
        )
        for i in range(self.n_patches):
            self.patch2uvgrid[i] = self.face2uvgrid[self.face2patch == i][0]

    def resolve_topology(self, wrap_patch_mask: torch.Tensor) -> BRep:
        """
        Resolve topology of the merged_mesh to construct brep
        :param wrap_patch_mask: bool tensor indicating whether each patch wraps around occupancy
        :return:
        """
        v_merged = torch.tensor(self.merged_mesh.vertices, device=self.device)
        f_merged = torch.tensor(self.merged_mesh.faces, device=self.device)
        # select wrap patches for each uvgrid
        uvgrid2wrap_patches = [[] for _ in range(self.n_uvgrid)]
        for patch_idx, mask in enumerate(wrap_patch_mask):
            if not mask:
                continue
            uvgrid_idx = self.face2uvgrid[self.face2patch == patch_idx][0]
            uvgrid2wrap_patches[uvgrid_idx].append(patch_idx)
        unique_edges = torch.tensor(self.merged_mesh.edges_unique, device=self.device)
        faces2unique_edge = torch.tensor(
            self.merged_mesh.faces_unique_edges, device=self.device
        )

        # get all the intersection between uvgrids
        intersection_edge_idxs = self._intersect_uvgrids(
            uvgrid2wrap_patches=uvgrid2wrap_patches,
            unique_edges=unique_edges,
            faces2unique_edge=faces2unique_edge,
        )

        # filter edges that do not form loop
        brep_edge_idxs, brep_face_edge_adj = self._intersection2brep_edges(
            intersection_edge_idxs=intersection_edge_idxs, unique_edges=unique_edges
        )
        # get brep edge
        brep_edge_vertex_adj = self._get_edge_vertex_adj(
            brep_edge_idxs=brep_edge_idxs, unique_edges=unique_edges
        )

        brep_edges = self._get_brep_edges(
            brep_edge_idxs=brep_edge_idxs, unique_edges=unique_edges, v_merged=v_merged
        )

        # vertex coordinates are not needed for occ brep construction,
        # however we obtain it for visualization
        brep_vertices = v_merged[torch.unique(brep_edge_vertex_adj.view(-1))]

        # now clean the indices of brep faces, face_edge_adj, edge_vertex_adj
        brep_faces, brep_face_edge_adj, brep_edge_vertex_adj = self._clean(
            brep_face_edge_adj=brep_face_edge_adj,
            brep_edge_vertex_adj=brep_edge_vertex_adj,
        )

        # visualize all intersection edges (useful for debugging)
        # brep_edges = []
        # for iei_face in intersection_edge_idxs:
        #     for iei in iei_face:
        #         if len(iei) == 0:
        #             continue
        #         v_idxs = torch.unique(unique_edges[iei])
        #         intersection_edge = v_merged[v_idxs]
        #         brep_edges.append(intersection_edge)

        out = BRep(
            faces=brep_faces,
            edges=brep_edges,
            vertices=brep_vertices,
            face_edge_adj=brep_face_edge_adj,
            edge_vertex_adj=brep_edge_vertex_adj,
        )
        return out

    def _clean(
        self,
        brep_face_edge_adj: List[List],
        brep_edge_vertex_adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[List], torch.Tensor]:
        """
        All of the faces, edges, vertices were indices of uvgrid or merged_mesh.
        Here, we clean all the indices so that we have a minimal representation ready for occ export.
        :param brep_face_edge_adj:
        :param brep_edge_vertex_adj:
        :return:
        """
        # remap faces as there may be unused uvgrids in the final brep
        faces, brep_face_edge_adj_remapped = [], []
        for i in range(len(brep_face_edge_adj)):
            # face not used for final brep construction
            if len(brep_face_edge_adj[i]) == 0:
                continue
            faces.append(self.uvgrid.coord[i])
            brep_face_edge_adj_remapped.append(brep_face_edge_adj[i])
        faces = torch.stack(faces)

        # remap vertices
        _, brep_edge_vertex_adj_remapped = torch.unique(
            brep_edge_vertex_adj, return_inverse=True
        )

        return faces, brep_face_edge_adj_remapped, brep_edge_vertex_adj_remapped

    def _get_brep_edges(
        self,
        brep_edge_idxs: List[torch.Tensor],
        unique_edges: torch.Tensor,
        v_merged: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Get list of edges for brep
        :param brep_edge_idxs: List of torch.Tensor containing edges in indices of merged_mesh
        :param unique_edges: unique edges of self.merged_mesh
        :param v_merged:
        :return:
        """

        edges = []
        for i in range(len(brep_edge_idxs)):
            edge_idxs = brep_edge_idxs[i]
            edge = unique_edges[edge_idxs]
            edge = self._unique_vertices(edge)
            edge_coords = v_merged[edge]
            if self.brep_edge_tol > 1e-6:
                edge_coords = self._simplify_edges(edge_coords, self.brep_edge_tol)
            edges.append(edge_coords)
        return edges

    def _simplify_edges(self, edge_coords: torch.Tensor, edge_tol: float):
        """
        Filters edges below this value
        :param edge_coords: N x 3 tensor
        :param edge_tol:
        :return:
        """
        n_edges = edge_coords.shape[0]
        selected_pts = [edge_coords[0]]

        for i in range(1, n_edges - 1):
            curr = edge_coords[i]
            dists = torch.norm(curr - torch.stack(selected_pts), dim=1)

            if torch.all(dists > edge_tol):
                selected_pts.append(curr)
        selected_pts.append(edge_coords[-1])
        return torch.stack(selected_pts)

    def _unique_vertices(self, edge: torch.Tensor):
        """
        Given N x 2 edge, containing vertex indices of edge, obtain unique vertices folloing the edge order.
        """
        if edge.shape[0] == 1:
            return edge.view(-1)
        res = []
        for i in range(edge.shape[0]):
            if i == 0:
                if edge[0, 0] in edge[1]:
                    res.append(edge[0, 1])
                    res.append(edge[0, 0])
                else:
                    res.append(edge[0, 0])
                    res.append(edge[0, 1])
            else:
                if res[-1] == edge[i, 0]:
                    res.append(edge[i, 1])
                else:
                    res.append(edge[i, 0])
        return torch.stack(res)

    def _get_edge_vertex_adj(
        self, brep_edge_idxs: List[torch.Tensor], unique_edges: torch.Tensor
    ) -> torch.Tensor:
        """
        Get edge vertex adjacency
        :param brep_edge_idxs: List of torch.Tensor containing edges in indices of merged_mesh
        :param unique_edges: unique edges of self.merged_mesh
        :return:
            len(brep_edges) x 2 tensor indicating vertex index of each edge
        """
        v_idxs = []
        for be in brep_edge_idxs:
            vs = unique_edges[be]  # n x 2
            if vs.shape[0] == 1:
                v_idxs.append(vs[0])
            else:
                if torch.sum(vs[1] - vs[0, 0] == 0) == 0:
                    start = vs[0, 0]
                else:
                    start = vs[0, 1]
                if torch.sum(vs[-2] - vs[-1, 0] == 0) == 0:
                    end = vs[-1, 0]
                else:
                    end = vs[-1, 1]
                v_idxs.append(torch.stack([start, end]))
        v_idxs = torch.stack(v_idxs)
        return v_idxs

    def _get_cycles_i(
        self,
        i: int,
        intersection_edge_idxs: List[List[Union[None, torch.Tensor]]],
        unique_edges: torch.Tensor,
    ) -> Tuple[List[Cycle], List[Cycle]]:
        """
        :param i: index of current uvgrid to detect cycles
        :param intersection_edge_idxs: squared list of list with size self.n_uvgrids
                Index i, j contains edge indices from unique_edge
        :param unique_edges: unique edges of self.merged_mesh
        """
        edge_idxs2uvgrid_idx = -torch.ones(
            unique_edges.shape[0], device=self.device, dtype=torch.long
        )
        for j in range(self.n_uvgrid):
            if i == j:
                continue
            edge_idxs_ij = intersection_edge_idxs[i][j]
            # no intersecting edges btw i and j
            if len(edge_idxs_ij) == 0:
                continue
            edge_idxs2uvgrid_idx[edge_idxs_ij] = j
        edge_idxs = torch.where(edge_idxs2uvgrid_idx != -1)[0]
        if len(edge_idxs) == 0:
            return [], []
        edges = unique_edges[edge_idxs]
        # map global vertices to local vertices
        local2global_v, edges_local = torch.unique(edges, return_inverse=True)

        # check cycle
        g = nx.Graph()
        g.add_edges_from(edges_local.cpu().numpy())
        cycles = nx.simple_cycles(g)  # list of list of vertices
        cycles = list(cycles)

        # create n_vertex x n_vertex matrix for edge idx retrieval
        n_vs = edges_local.max() + 1
        edge_adj_local = -torch.ones(
            n_vs,
            n_vs,
            dtype=torch.long,
            device=self.device,
        )
        edge_adj_local[tuple(edges_local.T)] = torch.arange(
            edges_local.shape[0], device=self.device
        )
        edges_local_reverse = torch.flip(edges_local, dims=(1,))
        edge_adj_local[tuple(edges_local_reverse.T)] = torch.arange(
            edges_local.shape[0], device=self.device
        )

        ##########################################################################
        # break cycle into edges based on which intersection of uvgrid made edge #
        ##########################################################################
        cycles_brep_edges = []
        for cycle in cycles:
            start = torch.tensor(cycle, dtype=torch.long, device=self.device)
            end = torch.tensor(
                cycle[1:] + cycle[:1], dtype=torch.long, device=self.device
            )
            # get global edge idxs and uvgrid that correspond to edges of each cycle
            edge_idxs_cycle = edge_idxs[edge_adj_local[start, end]]
            uvgrid_idxs_cycle = edge_idxs2uvgrid_idx[edge_idxs_cycle]

            # move start idx of cycle as the first and last edge might stem from same uvgrid
            start_idx = len(uvgrid_idxs_cycle)
            for k in range(len(uvgrid_idxs_cycle) - 1, -1, -1):
                if uvgrid_idxs_cycle[k] == uvgrid_idxs_cycle[0]:
                    start_idx = k
                else:
                    break
            edge_idxs_cycle = torch.cat(
                [edge_idxs_cycle[start_idx:], edge_idxs_cycle[:start_idx]]
            )
            uvgrid_idxs_cycle = torch.cat(
                [uvgrid_idxs_cycle[start_idx:], uvgrid_idxs_cycle[:start_idx]]
            )

            # split edges based on uvgrid changes
            cycle_brep_edge = []
            edge_start = 0
            for k in range(1, len(edge_idxs_cycle)):
                # add edge if the uvgrid changes
                if uvgrid_idxs_cycle[edge_start] != uvgrid_idxs_cycle[k]:
                    brep_edge = BrepEdge(
                        edge_idxs=edge_idxs_cycle[edge_start:k],
                        uvgrid_idxs=[i, uvgrid_idxs_cycle[edge_start].cpu().item()],
                    )
                    cycle_brep_edge.append(brep_edge)
                    edge_start = k
            brep_edge = BrepEdge(
                edge_idxs=edge_idxs_cycle[edge_start : len(edge_idxs_cycle)],
                uvgrid_idxs=[i, uvgrid_idxs_cycle[edge_start].cpu().item()],
            )
            cycle_brep_edge.append(brep_edge)
            cycles_brep_edges.append(Cycle(i, cycle_brep_edge))

        ############################################################
        # Detect cycles that share same edges (overlapping cycles) #
        ############################################################
        # TODO: could optimize this part
        if len(cycles) > 1:
            edges_in_cycles = defaultdict(list)
            for c_idx, c in enumerate(cycles):
                for e in c:
                    edges_in_cycles[e].append(c_idx)
            overlapping_cycle_idxs = set()
            for e, c_idxs in edges_in_cycles.items():
                if len(c_idxs) > 1:
                    overlapping_cycle_idxs.add(frozenset(c_idxs))
            overlapping_cycle_idxs = [
                item for fset in overlapping_cycle_idxs for item in fset
            ]
        else:
            overlapping_cycle_idxs = []

        non_overlapping_cycles, overlapping_cycles = [], []
        for c_idx in range(len(cycles)):
            if c_idx in overlapping_cycle_idxs:
                overlapping_cycles.append(cycles_brep_edges[c_idx])
            else:
                non_overlapping_cycles.append(cycles_brep_edges[c_idx])

        return non_overlapping_cycles, overlapping_cycles

    def _add_brep_edges_from_cycle(
        self,
        uvgrid_idx: int,
        cycle: Cycle,
        brep_edges: List[BrepEdge],
        brep_face_edge_adj: List[List[int]],
    ):
        for new_be in cycle.brep_edges:
            in_existing_brep_edges = False
            for existing_be in brep_edges:
                if new_be.is_same(existing_be):
                    # dont add to brep_edges
                    # only touch the existing brep edge with current uvgrid_idx
                    in_existing_brep_edges = True
                    existing_be.touch(uvgrid_idx)
                    break
            if not in_existing_brep_edges:
                new_be.touch(uvgrid_idx)
                brep_edges.append(new_be)
                brep_face_edge_adj[new_be.uvgrid_idxs[0]].append(len(brep_edges) - 1)
                brep_face_edge_adj[new_be.uvgrid_idxs[1]].append(len(brep_edges) - 1)

    def _resolve_untouched_brep_edges_with_overlapping_cycles(
        self,
        overlapping_cycles: List[Cycle],
        brep_edges: List[BrepEdge],
        brep_face_edge_adj: List[List[int]],
    ) -> int:
        """
        Each edge should be touched (added) twice by two uvgrids
        An untouched edge means that an edge was addded once, which means there does not exist a cycle in the untouching uvgrid that contains the untouched edge.
        This function detects untouched edge and adds a cycle that contains the untouched edge if possible, greedily.
        :param overlapping_cycles:
        :param brep_edges:
        :param brep_face_edge_adj
        :return:
            Integer code whether brep edge was resolved:
            - 0: An edge is resolved
            - 1: Cannot resolve more edges
            - 2: All the edges are resolved :)
        """
        untouched_edge_resolved = 2
        for be_idx in range(len(brep_edges)):
            # detected untouched edge
            brep_edge = brep_edges[be_idx]
            if not brep_edge.touchable:
                untouched_edge_resolved = 1
            elif len(brep_edge.touched_uvgrid_idxs) == 1:
                # there is some problem in the untouched uvgrid
                untouched_uvgrid_idx = set(brep_edges[be_idx].uvgrid_idxs) - set(
                    brep_edges[be_idx].touched_uvgrid_idxs
                )
                untouched_uvgrid_idx = list(untouched_uvgrid_idx)[0]

                # filter cycles that contain the edge of interest
                edge_containing_cycles = []
                for cycle in overlapping_cycles:
                    if cycle.uvgrid_idx != untouched_uvgrid_idx:
                        continue
                    for brep_edge_cycle in cycle.brep_edges:
                        if brep_edge_cycle.is_same(brep_edge):
                            edge_containing_cycles.append(cycle)
                            break

                if len(edge_containing_cycles) == 0:
                    # if unresolvable
                    # mark so don't we don't look for a cycle that contains this edge next time
                    brep_edge.touchable = False
                    untouched_edge_resolved = 1
                    continue
                elif len(edge_containing_cycles) == 1:
                    self._add_brep_edges_from_cycle(
                        uvgrid_idx=untouched_uvgrid_idx,
                        cycle=edge_containing_cycles[0],
                        brep_edges=brep_edges,
                        brep_face_edge_adj=brep_face_edge_adj,
                    )
                else:
                    # choose the best cycle that resolves the situation
                    self._add_brep_edges_from_cycle(
                        uvgrid_idx=untouched_uvgrid_idx,
                        cycle=edge_containing_cycles[0],
                        brep_edges=brep_edges,
                        brep_face_edge_adj=brep_face_edge_adj,
                    )
                    print("Please report this to Dongsu with an example of the uvgrid")
                    # raise NotImplementedError(
                    #     "Please report this to Dongsu with an example of the uvgrid!"
                    # )

                # an edge is resolved, continue
                untouched_edge_resolved = 0
                break
        return untouched_edge_resolved

    def _intersection2brep_edges(
        self,
        intersection_edge_idxs: List[List[Union[None, torch.Tensor]]],
        unique_edges: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[List[int]]]:
        """
        :param intersection_edge_idxs: squared list of list with size self.n_uvgrids
                Index i, j contains edge indices from unique_edge
        :param unique_edges: unique edges of self.merged_mesh
        :return:
            - brep_edge_idxs: List of torch.Tensor containing edges in indices of merged_mesh
            - brep_face_edge_adj: List of list of int
                Contains mapping from a uvgrid to brep_edge_idx
        """
        brep_edges: List[
            BrepEdge
        ] = []  # each element should contain (merged_mesh) edge idx
        # each element should contain face -> brep_edge index mapping
        brep_face_edge_adj = [[] for _ in range(self.n_uvgrid)]

        ##############################################################################################
        # Obtain cycles for by traversing along uvgrid and extract edges from non-overlapping cycles #
        ##############################################################################################
        overlapping_cycles = []
        for i in range(self.n_uvgrid):
            non_overlapping_cycles_i, overlapping_cycles_i = self._get_cycles_i(
                i, intersection_edge_idxs, unique_edges
            )
            overlapping_cycles.extend(overlapping_cycles_i)
            for cycle in non_overlapping_cycles_i:
                # add edges from cycles
                self._add_brep_edges_from_cycle(
                    i, cycle, brep_edges, brep_face_edge_adj
                )
        #####################################################################
        # Deal with brep edges that have been added from only a single face #
        #####################################################################
        while True:
            untouched_edge_resolved = (
                self._resolve_untouched_brep_edges_with_overlapping_cycles(
                    overlapping_cycles, brep_edges, brep_face_edge_adj
                )
            )
            if untouched_edge_resolved != 0:
                # nothing more can be resolved
                if untouched_edge_resolved == 1:
                    print("Topology resolution: an edge with a single face exists")
                elif untouched_edge_resolved == 2:
                    print("Topology successfully resolved")
                break

        # Debug
        # untouched_brep_edges_idxs = [
        #     i for i, x in enumerate(brep_edges) if x.touch_cnt() == 1
        # ]
        # untouched_brep_edges = [brep_edges[i] for i in untouched_brep_edges_idxs]
        # untouched_uvgrid_idxs = [
        #     list(set(x.uvgrid_idxs) - set(x.touched_uvgrid_idxs))[0]
        #     for x in untouched_brep_edges
        # ]
        # unique_untouched_uvgrid_idxs = set(untouched_uvgrid_idxs)

        brep_edge_idxs = [x.edge_idxs for x in brep_edges]
        return brep_edge_idxs, brep_face_edge_adj

    def _add_brep_edge(
        self,
        brep_edges: List[torch.Tensor],
        brep_face_edge_adj_i: List[int],
        edge_idxs: torch.Tensor,
        brep_edge_cnt: int,
    ) -> int:
        """
        Adds a new edge to brep_edges, while handling duplicates in existing edge.
        :param brep_edges:
        :param brep_face_edge_adj:
        :param edge_idxs:
        :param brep_edge_cnt:
        :return:
            brep_edge_cnt
        """
        for i, existing_edge_idxs in enumerate(brep_edges):
            # quick testing with edge length
            if existing_edge_idxs.shape[0] == edge_idxs.shape[0]:
                uniques, cnts = torch.unique(
                    torch.cat([existing_edge_idxs, edge_idxs]), return_counts=True
                )
                same = torch.sum(cnts != 2) == 0
                # do not add to brep_edges since it exist
                if same:
                    brep_face_edge_adj_i.append(i)
                    return brep_edge_cnt

        brep_edges.append(edge_idxs)
        brep_face_edge_adj_i.append(brep_edge_cnt)
        return brep_edge_cnt + 1

    def _intersect_uvgrids(
        self,
        uvgrid2wrap_patches: List[List[int]],
        unique_edges: torch.Tensor,
        faces2unique_edge: torch.Tensor,
    ) -> List[List[Union[torch.Tensor, None]]]:
        """
        # Get edges formed by intersection of patches and a mapping from patches to edges
         :param wrap_patch_idxs: List of patch indexes that wrap voted occupancy
         :param unique_edges: unique edges of self.merged_mesh
         :param faces2unique_edge: mapping from faces to unique edge
         :return:
             - intersection_edge_idxs: squared list of list with size self.n_uvgrids
                Index i, j contains edge indices from unique_edge

        """
        #############################################################
        # Compute all the intersection between all pairs of uvgrids #
        #############################################################
        # cache edge for each uvgrid
        uvgrid2edge_idxs = [
            self._get_patch_edge_idxs(p, faces2unique_edge) for p in uvgrid2wrap_patches
        ]
        # checks intersected edges for all pairs of i and j
        edge_idx_counter = torch.zeros(unique_edges.shape[0], device=self.device)
        intersection_edge_idxs = [
            [None for j in range(self.n_uvgrid)] for i in range(self.n_uvgrid)
        ]
        for i in range(self.n_uvgrid):
            edge_idxs_i = uvgrid2edge_idxs[i]
            for j in range(self.n_uvgrid):
                if (i == j) or (len(edge_idxs_i) == 0):
                    # no edge in i or i == j
                    intersection_edge_idxs[i][j] = torch.tensor([], device=self.device)
                    continue
                if intersection_edge_idxs[i][j] is None:
                    edge_idxs_j = uvgrid2edge_idxs[j]
                    if len(edge_idxs_j) == 0:
                        # no edge in j
                        intersection_edge_idxs[i][j] = torch.tensor(
                            [], device=self.device
                        )
                    else:
                        # compute intersections
                        edge_idx_counter[edge_idxs_i] = 1
                        edge_idx_counter[edge_idxs_j] += 1
                        intersection = torch.where(edge_idx_counter == 2)[0]
                        intersection_edge_idxs[i][j] = intersection
                        # clear counter
                        edge_idx_counter[:] = 0
                else:
                    intersection_edge_idxs[i][j] = intersection_edge_idxs[j][i]
        return intersection_edge_idxs

    def _get_patch_edge_idxs(
        self, patches: List[int], face2unique_edge: torch.Tensor
    ) -> torch.Tensor:
        """
        Get edges contained in list of patches
        :param patches:
        :param face2unique_edge: mapping from faces to unique edge
        :return:
            edges
        """
        edge_idxs = []
        if len(patches) == 0:
            return torch.tensor([], device=self.device)
        for patch in patches:
            edge_idxs.append(face2unique_edge[self.face2patch == patch])
        edge_idxs = torch.cat(edge_idxs)
        edge_idxs = torch.unique(edge_idxs.view(-1))
        return edge_idxs

    def merge_uvgrids(
        self, uvgrid: UvGrid, append_unit_cube: bool
    ) -> Tuple[trimesh.Trimesh, torch.Tensor]:
        """
        Merge all the uvgrid meshes and resolve self-intersection
        :param uvgrid:
        :param append_unit_cube: bool
            Append unit cube mesh when merging uvgrid extended meshes
        :return:
            - merged_mesh: self-intersection solved mesh that merges uv-extended mesh
            - face2uvgrid: mapping from face_idx to uvgrid_idx
        """

        pm_meshes = []
        for f_idx in range(len(uvgrid.coord)):
            v, f = uvgrid.meshify(f_idx, use_grid_mask=False)
            pm_mesh = pymesh.form_mesh(vertices=v, faces=f)
            pm_meshes.append(pm_mesh)

        if append_unit_cube:
            box_mesh_tri = trimesh.primitives.Box(
                bounds=[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]
            )
            box_mesh_pm = pymesh.form_mesh(
                vertices=box_mesh_tri.vertices, faces=box_mesh_tri.faces
            )
            pm_meshes.append(box_mesh_pm)

        pm_merged = pymesh.merge_meshes(pm_meshes)

        face_sources_merged = pm_merged.get_attribute("face_sources").astype(np.int32)
        pm_resolved_ori = pymesh.resolve_self_intersection(pm_merged)
        # removing duplicated vertices raise problems
        # pm_resolved, info_dict = pymesh.remove_duplicated_vertices(
        #     pm_resolved_ori, tol=self.duplicate_vertices_tol, importance=None
        # )

        face_sources_resolved_ori = pm_resolved_ori.get_attribute(
            "face_sources"
        ).astype(np.int32)
        face2uvgrid = face_sources_merged[
            face_sources_resolved_ori
        ]  # merged_face_idx -> pm_mesh idx
        tri_resolved = trimesh.Trimesh(
            vertices=pm_resolved_ori.vertices, faces=pm_resolved_ori.faces
        )
        face2uvgrid = torch.from_numpy(face2uvgrid).to(self.device)
        return tri_resolved, face2uvgrid

    def get_face2patch(
        self,
        mesh: trimesh.Trimesh,
        face2uvgrid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run connected components to get labels for each faces.
        Patches are fixed to remove self-intersection caused bugs in pymesh
        :return:
        """
        face_adjacency = mesh.face_adjacency
        face2patch = trimesh.graph.connected_component_labels(
            edges=face_adjacency, node_count=len(mesh.faces)
        )  # array of F, indicating labels of each faces

        ##############################
        # Fix self intersection bugs #
        ##############################
        # Ideally, patches should be partitioned by uvgrid meshes from other partitions.
        # However, there is a bug when solving the self-intersection
        # that patches are partitioned without the existence of other uvgrids.

        n_edges = mesh.edges_unique.shape[0]
        # We first obtain edges for each patch
        n_patches = face2patch.max() + 1
        edge2patch = np.zeros((n_edges, n_patches), dtype=np.bool)
        for patch_idx in range(n_patches):
            face_in_patch = np.arange(mesh.faces.shape[0])[face2patch == patch_idx]
            edges_in_patch = mesh.faces_unique_edges[face_in_patch]
            edge2patch[edges_in_patch, patch_idx] = True

        n_uvgrids = face2uvgrid.max() + 1
        edge2uvgrid = np.zeros((n_edges, n_uvgrids), dtype=np.bool)
        for uvgrid_idx in range(n_uvgrids):
            face_in_uvgrid = np.arange(mesh.faces.shape[0])[
                face2uvgrid.cpu().numpy() == uvgrid_idx
            ]
            edges_in_uvgrid = mesh.faces_unique_edges[face_in_uvgrid]
            edge2uvgrid[edges_in_uvgrid, uvgrid_idx] = True

        # find edges that belong to multiple patches but only one uvgrid
        defect_edge_masks = (edge2patch.sum(axis=1) > 1) & (
            edge2uvgrid.sum(axis=1) == 1
        )

        # {# of same patches} x n_edges, row indicates column index of True belongs to same patch
        same_patch_masks = edge2patch[defect_edge_masks]

        # build adjacency matrix of same patches and run connected components
        adj_matrix = np.zeros((n_patches, n_patches), dtype=bool)
        for same_patch_mask in same_patch_masks:
            same_patch_idxs = np.where(same_patch_mask)[0]
            adjacencies = np.stack(
                np.meshgrid(same_patch_idxs, same_patch_idxs), axis=-1
            ).reshape(-1, 2)
            adj_matrix[adjacencies[:, 0], adjacencies[:, 1]] = True

        G = nx.from_numpy_array(adj_matrix)
        connected_components = list(nx.connected_components(G))

        # create a mapping from prev patch to updated patch
        patch2new_patch = -np.ones(n_patches, dtype=np.int32)
        for i, cc in enumerate(connected_components):
            patch2new_patch[np.array(list(cc))] = i
        face2patch = patch2new_patch[face2patch]

        # obtain defect edges for debugging
        # {n_defect_edges} x 2
        defect_edges = mesh.edges_unique[np.where(defect_edge_masks)[0]]
        defect_edges = torch.tensor(defect_edges, device=self.device)
        if defect_edges.shape[0] != 0:
            print(f"Found {defect_edges.shape[0]} defect edges")

        face2patch = torch.tensor(face2patch, device=self.device)

        return face2patch, defect_edges

    def _sample_points_on_faces(self, mesh, face_indices, num_points):
        """
        Samples points on specified faces of a mesh.

        Parameters:
            mesh (trimesh.Trimesh): The mesh object.
            face_indices (list or array): List of face indices to sample from.
            num_points (int): Number of points to sample across the specified faces.

        Returns:
            numpy.ndarray: An array of sampled points of shape (num_points, 3).
        """
        # Get the areas of the specified faces
        areas = mesh.area_faces[face_indices]
        areas_sum = np.sum(areas)
        # if area is close to zero
        if areas_sum < 1e-6:
            return np.empty((0, 3)), np.empty((0,), dtype=np.int64)

        # Calculate the probability of sampling each face based on its area
        face_probs = areas / areas_sum

        # Randomly choose faces based on their area
        chosen_faces = np.random.choice(face_indices, size=num_points, p=face_probs)

        # Sample points on the chosen faces
        sampled_points = []
        for face_id in chosen_faces:
            # Get the vertices of the face
            vertices = mesh.vertices[mesh.faces[face_id]]

            # Sample a random point within the triangle
            u, v = np.random.rand(2)
            if u + v > 1:
                u, v = 1 - u, 1 - v
            point = (1 - u - v) * vertices[0] + u * vertices[1] + v * vertices[2]
            sampled_points.append(point)

        return np.array(sampled_points), chosen_faces

    def segment_wrap_patch(
        self,
        occ_grid: torch.Tensor,
        grid_min_range: float,
        grid_max_range: float,
        patch_n_sample: int,
        min_sample_per_patch: int,
        patch_normal_offset: float,
        min_wrap_patch_rate: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given occupancy grid, segment whether the patches wraps (divides the occupancy) around occupancy.
        :param occ_grid: Tensor of grid_res x grid_res x grid_res
        :param grid_min_range:
        :param grid_max_range:
        :param patch_normal_offset: offset to detect occupancy from each face to normal direction
        :param patch_n_sample: number of samples for detecting occupancy
        :param min_sample_per_patch: additional minimum number of samples per patch to ensure each patch has at least some amount of points
        :param patch_normal_offset:
        :param min_wrap_patch_rate:
        :return:
            patch_wrap_mask: bool torch.Tensor of len(self.face_labels)
            query_coord: N x 3 (query_coord and occupancy) for visualization
        """
        grid_res = occ_grid.shape[0]
        assert (
            (len(occ_grid.shape) == 3)
            and (grid_res == occ_grid.shape[1])
            and (grid_res == occ_grid.shape[2])
        ), f"occ_grid should be of size res x res x res, got shape: {occ_grid.shape}"
        device = occ_grid.device

        # sample points and corresponding face idx from the merged mesh
        sample_coords, sample_f_idxs = [], []
        # uniformly sample from the merged mesh
        sample_coord_uniform, sample_f_idx_uniform = self.merged_mesh.sample(
            patch_n_sample, return_index=True
        )
        sample_coords.append(sample_coord_uniform)
        sample_f_idxs.append(sample_f_idx_uniform)

        # for each patch, sample faces to ensure little patches are well segmented
        for patch_idx in range(self.n_patches):
            patch_faces = torch.where(self.face2patch == patch_idx)[0]
            sample_coord_patch, sample_f_idx_patch = self._sample_points_on_faces(
                self.merged_mesh, patch_faces.cpu().numpy(), min_sample_per_patch
            )
            sample_coords.append(sample_coord_patch)
            sample_f_idxs.append(sample_f_idx_patch)

        sample_coords = torch.tensor(
            np.concatenate(sample_coords), device=device, dtype=torch.float32
        )
        sample_f_idxs = torch.tensor(np.concatenate(sample_f_idxs), device=device)

        # transform to occupancy grid space coordinates
        sample_coord_grid = (sample_coords - grid_min_range) / (
            grid_max_range - grid_min_range
        )
        sample_coord_grid = (grid_res - 1) * sample_coord_grid

        # query occupancy near the points sampled from the mesh
        face_normal = torch.tensor(
            self.merged_mesh.face_normals, device=device, dtype=torch.float32
        )
        query_normal = face_normal[sample_f_idxs]
        pos_query_coord = sample_coord_grid + patch_normal_offset * query_normal
        neg_query_coord = sample_coord_grid - patch_normal_offset * query_normal

        # query occupancy for poistive and negative samples
        # if one is outside and the other is inside, patch wraps the query points
        pos_occ = self._query_occ(occ_grid, pos_query_coord)
        neg_occ = self._query_occ(occ_grid, neg_query_coord)
        wraps_patch_query = torch.bitwise_xor(pos_occ, neg_occ)  # Tensor of N

        # aggregate patch-wise
        query_patch_idx = self.face2patch[sample_f_idxs]
        scatter_src = torch.cat(
            [
                wraps_patch_query.float(),
                torch.zeros(self.n_patches, device=device, dtype=torch.float32),
            ]
        )
        scatter_idx = torch.cat(
            [
                query_patch_idx.long(),
                torch.arange(self.n_patches, device=device, dtype=torch.int64),
            ]
        )
        patch_wrap_rate = scatter(scatter_src, scatter_idx, reduce="mean")
        patch_wrap_mask = patch_wrap_rate > min_wrap_patch_rate

        # transform query coord to patch world space coordinate
        query_coord = torch.cat([pos_query_coord, neg_query_coord], dim=0)
        query_coord = (query_coord / (grid_res - 1)) * (
            grid_max_range - grid_min_range
        ) + grid_min_range
        # append occupancy as well
        query_coord = torch.cat(
            [query_coord, torch.cat([pos_occ, neg_occ]).float().unsqueeze(1)], dim=1
        )  # N x 4
        return patch_wrap_mask, query_coord

    @staticmethod
    def _query_occ(occ_grid: torch.Tensor, query_coord: torch.Tensor) -> torch.Tensor:
        """
        Mark occupancy of the query coord
        :param occ_grid: Bool tensor of grid_res x grid_res x grid_res
        :param query_coord: Float tensor of N x 3 in grid coordinate system
        :return:
            query_occ: Bool tensor of N where if query_coord lies in the occupied cell
        """
        query_coord = torch.round(query_coord).int()
        inside_grid_mask = torch.bitwise_and(
            torch.sum(query_coord >= 0, dim=1) == 3,
            torch.sum(query_coord < occ_grid.shape[0], dim=1) == 3,
        )

        query_occ = torch.zeros(
            query_coord.shape[0], dtype=torch.bool, device=occ_grid.device
        )
        query_occ[inside_grid_mask] = occ_grid[tuple(query_coord[inside_grid_mask].T)]
        return query_occ

    def get_patches(self) -> List[trimesh.Trimesh]:
        """
        Obtain patches made by running connected components on merged meshes
        :return:
        """
        face2patch = self.face2patch.cpu().numpy()
        submeshes = []
        for patch_idx in range(self.n_patches):
            # # DEBUG
            # # exclude certain patches in uvgrid
            # note that this does not affect patch_idxs
            # if self.patch2uvgrid[patch_idx] in [0]:
            #     continue
            submesh = trimesh.Trimesh(
                vertices=np.array(self.merged_mesh.vertices),
                faces=np.array(self.merged_mesh.faces)[
                    np.where(face2patch == patch_idx)
                ],
            )
            submeshes.append(submesh)
        return submeshes
