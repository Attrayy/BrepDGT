import numpy as np
import glob

from ps_utils.structures import create_bbox

from brepdgt.primitives.uvgrid import UvGrid
from brepdgt.utils.common import (
    load_pointcloud,
    load_grid,
    save_pointcloud,
)
from brepdgt.utils.psr_utils import interpolate_uv_grid
from brepdgt.postprocessing.intersection_manager import (
    IntersectionManager,
    BRep,
)
from brepdgt.utils.vis import get_cmap
from brepdgt.utils.brep_checker import check_solid
from typing import Dict, Tuple, List, Union
from dataclasses import dataclass
from torch_scatter import scatter
from multiprocessing import Pool
from skimage import measure
import torch
import math
import os
import tempfile
import igl
import trimesh
from brepdgt.utils.common import timeout_wrapper


def render_blender(mesh_path: str, render_path: str, theta: float, quite: bool = True):
    cmd = f"./blender --background --python scripts/blender/render_for_fid.py -- {mesh_path} {render_path} {theta} normalize z-axis-up"
    if quite:
        cmd += " > /dev/null"
    else:
        print(cmd)
    os.system(cmd)


@dataclass(frozen=True)
class PostprocessorOutput:
    """
    Output of BRepConstructor containing occupancy and partition information.
    """

    uvgrid: UvGrid
    # Occupancy grid from PSR
    grid_coord: torch.Tensor
    occupancy: torch.Tensor
    # Extended partition grid by winding numbers
    partition: torch.Tensor
    n_partition: int
    # Partition voting results
    partition_occ_bit: torch.Tensor
    partition_occ: torch.Tensor
    partition_occ_rate: torch.Tensor
    partition_mesh: trimesh.base.Trimesh
    # extended uvgrid
    extended_uvgrid: UvGrid
    patches: List[trimesh.Geometry]
    wrap_patch_mask: torch.Tensor
    wrap_patch_query: torch.Tensor
    # topology fixed brep
    brep: BRep

    # debugging
    merged_mesh: trimesh.Trimesh
    defect_edges: torch.Tensor


class Postprocessor:
    def __init__(
        self,
        uvgrid: UvGrid,
        grid_res: int = 256,
        grid_range: float = 1.1,
        psr_uvgrid_res: Union[int, None] = 32,
        use_fwn: bool = True,
        psr_occ_thresh: float = 0.5,
        partition_occ_thresh: float = 0.5,
        scaled_rasterization: bool = True,
        scaled_stretch_thresh: float = 1.0,
        uvgrid_extension_len: float = 1.0,
        patch_n_sample: int = 1000000,
        min_patch_per_sample: int = 100,
        patch_normal_offset: float = 1.0,
        min_wrap_patch_rate: float = 0.7,
        smooth_extension: bool = True,
        min_extend_len: float = 0.0,
        device: str = "cuda",
        n_workers: int = 0,
    ):
        """
        Class for post-processing.
        Output includes brep and other internal by-products made during post-processing
        :param uvgrid: UVGrid
        :param grid_res: grid resolution for psr and winding numbers
        :param psr_uvgrid_res:
            Increase the resolution of uv grid for poisson surface reconstruction.
        :param use_fwn:
            Use fast winding numbers instead of original winding numbers.
            Is faster, but creates approximations
        :param psr_occ_thresh:
            Occupancy thresholding used in poisson surface reconstruction.
            Ideally, should be 0.5 but we use 0.5 for conservative occupancy prediction.
        :param partition_occ_thresh:
            Partition occupancy rate threshold used in occupancy voting.
            A partition with occupancy rate ({# of occupied pts} / {whole pts}) greater than this threshold will be considered occupied.
            Ideally, should be 0.5 but we use 0.5 for conservative occupancy voting.
        :param scaled_rasterization
            Rasterize in a scaled world-axis so that the object fits tightly in a unit bbox.
        :param uvgrid_extension_len
            Length of the extended uvgrid
        :param patch_n_sample: int
            Number of points to sample for patch when checking whether a patch divdies the voted occupancy.
        :param min_patch_per_sample: int
            Additional minimum number of points to sample for each patch when checking whether a patch divides the voted occupancy.
        :param patch_normal_offset
            Offset used for checking whether a patch (from the intersection of meshes) divides the voted occupancy.
        :param min_wrap_patch_rate
            Minimum patch wrap rate used when determining whether the patch wraps the occupied area.
        :param smooth_extension
            Extend uvgrid smoothly by averaging 3 nearest uvgrid directions near the boundary.
            Else, only coniders the nearest uvgrid to extend linearly.
        :param min_extend_len
            Minimum extend length for each uvgrid.
            Note that the length is applied in the scaled uvgrid dimension.
        :param device: str
            Device for acceleration. Use "cuda"
        :param n_workers: int
            Use multiprocessing for winding numbers (currently multiprocessing is slower :( )
        """
        uvgrid.compactify()
        self.uvgrid = uvgrid
        self.grid_res = grid_res
        self.grid_range = grid_range
        self.psr_interpolate_res = psr_uvgrid_res
        self.use_fwn = use_fwn
        self.psr_occ_thresh = psr_occ_thresh
        self.partition_occ_thresh = partition_occ_thresh
        self.scaled_rasterization = scaled_rasterization
        self.scaled_stretch_thresh = scaled_stretch_thresh
        self.uvgrid_extend_len = uvgrid_extension_len
        self.patch_n_sample = patch_n_sample
        self.min_patch_per_sample = min_patch_per_sample
        self.patch_normal_offset = patch_normal_offset
        self.min_wrap_patch_rate = min_wrap_patch_rate
        self.smooth_extension = smooth_extension
        self.device = device
        self.n_workers = n_workers

        # Move UVGrid attributes to the specified device
        self.uvgrid.to_tensor(self.device)

        # Normalize UVGrid coordinates to fit within a unit cube [-1, 1]^3
        if self.scaled_rasterization:
            (
                self.scaled_uvgrid,
                self.axis_scale,
                self.axis_offset,
            ) = self.uvgrid.stretch_axis_under_threshold(
                range_min=-self.scaled_stretch_thresh / 2,
                range_max=self.scaled_stretch_thresh / 2,
                threshold=self.scaled_stretch_thresh,
            )
        else:
            self.scaled_uvgrid = self.uvgrid
            self.axis_scale = torch.ones(3, device=device)
            self.axis_offset = torch.zeros(3, device=device)
        self.scaled_uvgrid_extend = self.scaled_uvgrid.extend(
            self.uvgrid_extend_len,
            smooth_extension=self.smooth_extension,
            min_extend_len=min_extend_len,
        )

        # Meshgrid to use for rasterization operations
        # approximately ranges [-1, 1] ^ 3
        # the min and max is not exactly -1 and 1 since PSR requires a little margin (grid_range)
        #   and we want to meshgrid to be exactly same as PSR
        self.grid_min_range = -self.grid_range + self.grid_range / (self.grid_res + 1)
        self.grid_max_range = self.grid_range - self.grid_range / (self.grid_res + 1)
        self.scaled_meshgrid = self.create_meshgrid(
            self.grid_min_range, self.grid_max_range
        )

    def postprocess(
        self, ret_scaled: bool = False, ret_none_if_failed=False, verbose: bool = False
    ) -> Union[PostprocessorOutput, None]:
        """
        Computes the B-Rep (Boundary Representation) from the UV grid.
        Each stage is wrapped with try-except to handle failures, and None is filled for failed ones.
        :param ret_scaled
            Return scaled output for debugging
        :param ret_none_if_failed
            Return None if failed.
        :param verbose
            Whether to print verbose output
        """
        if verbose:
            print("Starting postprocessing stages...")

        # Try each stage and catch errors, filling with None in case of failure
        try:
            if verbose:
                print("Computing occupancy from PSR...")
            # Get occupancy from PSR
            occupancy = self.get_occupancy()
        except Exception as e:
            occupancy = None
            print(f"Failed to get occupancy: {e}")
            if ret_none_if_failed:
                return None

        try:
            if verbose:
                print("Computing extended partition grid using winding numbers...")
            # Get extended partition grid using winding numbers
            partition, n_partition = self.get_extended_partition_grid()
        except Exception as e:
            partition, n_partition = None, None
            print(f"Failed to get extended partition grid: {e}")
            if ret_none_if_failed:
                return None

        try:
            if verbose:
                print("Performing occupancy voting based on partitions...")
            # Perform occupancy voting based on partitions
            partition_occ_bit, partition_occ, partition_occ_rate = self.vote_occupancy(
                occupancy=occupancy,
                partition=partition,
                n_partition=n_partition,
            )
        except Exception as e:
            partition_occ_bit, partition_occ, partition_occ_rate = None, None, None
            print(f"Failed to perform occupancy voting: {e}")
            if ret_none_if_failed:
                return None

        try:
            if verbose:
                print("Creating mesh from partition occupancy...")
            # Perform marching cubes on partition_occ
            partition_mesh = self.create_mesh_from_partition_occ(partition_occ)
        except Exception as e:
            partition_mesh = None
            print(f"Failed to perform marching cubes: {e}")
            if ret_none_if_failed:
                return None

        try:
            if verbose:
                print("Creating intersection manager...")
            # Intersect all the uvgrid extended meshes
            intersection_manager = IntersectionManager(self.scaled_uvgrid_extend)
        except Exception as e:
            intersection_manager = None
            print(f"Failed to create IntersectionManager: {e}")
            if ret_none_if_failed:
                return None

        try:
            if verbose:
                print("Segmenting patches that wrap occupied region...")
            # Segment patches that wrap occupied region
            wrap_patch_mask, wrap_patch_query = intersection_manager.segment_wrap_patch(
                occ_grid=partition_occ,
                grid_min_range=self.grid_min_range,
                grid_max_range=self.grid_max_range,
                patch_n_sample=self.patch_n_sample,
                min_sample_per_patch=self.min_patch_per_sample,
                patch_normal_offset=self.patch_normal_offset,
                min_wrap_patch_rate=self.min_wrap_patch_rate,
            )
        except Exception as e:
            wrap_patch_mask, wrap_patch_query = None, None
            print(f"Failed to segment wrap patches: {e}")
            if ret_none_if_failed:
                return None

        try:
            if verbose:
                print("Getting patches from intersection manager...")
            # Get patches from intersection manager
            patches = intersection_manager.get_patches()
        except Exception as e:
            patches = None
            print(f"Failed to get patches: {e}")
            if ret_none_if_failed:
                return None

        try:
            if verbose:
                print("Constructing BRep by resolving topology...")
            # Construct brep
            brep = intersection_manager.resolve_topology(wrap_patch_mask)
            if not ret_scaled:
                brep_faces_shape = brep.faces.shape
                brep.faces = self._unscale(brep.faces.view(-1, 3)).view(
                    *brep_faces_shape
                )
                for i in range(len(brep.edges)):
                    brep.edges[i] = self._unscale(brep.edges[i])
                brep.vertices = self._unscale(brep.vertices)
        except Exception as e:
            brep = None
            print(f"Failed to resolve topology: {e}")
            if ret_none_if_failed:
                return None

        merged_mesh = intersection_manager.merged_mesh
        defect_edges = intersection_manager.defect_edges

        # Unscale meshgrid for visualization
        if ret_scaled:
            # visualizing scaled output is better for debugging
            meshgrid = self.scaled_meshgrid
            uvgrid = self.scaled_uvgrid
        else:
            if verbose:
                print("Unscaling meshgrid and meshes...")
            meshgrid = self._unscale(self.scaled_meshgrid)
            if partition_mesh is not None:
                partition_mesh.vertices = self._unscale(
                    np.array(partition_mesh.vertices)
                )
            for patch in patches:
                patch.vertices = self._unscale(np.array(patch.vertices))
            merged_mesh.vertices = self._unscale(merged_mesh.vertices)
            uvgrid = self.uvgrid

        if verbose:
            print("Creating PostprocessorOutput...")
        # Prepare the output
        pp_out = PostprocessorOutput(
            uvgrid=uvgrid,
            grid_coord=meshgrid,
            occupancy=occupancy,
            partition=partition,
            n_partition=n_partition,
            partition_occ_bit=partition_occ_bit,
            partition_occ=partition_occ,
            partition_occ_rate=partition_occ_rate,
            partition_mesh=partition_mesh,
            extended_uvgrid=self.scaled_uvgrid_extend,
            patches=patches,
            wrap_patch_mask=wrap_patch_mask,
            wrap_patch_query=wrap_patch_query,
            brep=brep,
            merged_mesh=merged_mesh,
            defect_edges=defect_edges,
        )
        return pp_out

    def get_occupancy(self) -> torch.Tensor:
        """
        Computes the occupancy grid using Poisson Surface Reconstruction (PSR).
        """
        depth = int(math.log2(self.grid_res))
        assert (
            2**depth == self.grid_res
        ), f"grid_res must be a power of 2, got {self.grid_res}"

        # Extract data from UVGrid
        coord = self.scaled_uvgrid.coord.cpu().numpy()
        grid_mask = self.scaled_uvgrid.grid_mask.cpu().numpy()
        empty_mask = self.scaled_uvgrid.empty_mask.cpu().numpy()
        normal = self.scaled_uvgrid.normal.cpu().numpy()

        # Filter out empty faces
        coord = coord[~empty_mask]
        normal = normal[~empty_mask]
        grid_mask = grid_mask[~empty_mask]

        # Optionally interpolate UV grid for higher PSR resolution
        if self.psr_interpolate_res is not None:
            coord, normal, grid_mask = interpolate_uv_grid(
                coord, normal, grid_mask, self.psr_interpolate_res
            )

        # Keep only valid surface points
        coord = coord[grid_mask]
        normal = normal[grid_mask]

        # PSR: https://github.com/mkazhdan/PoissonRecon
        with tempfile.TemporaryDirectory() as tmpdir:
            save_pointcloud(f"{tmpdir}/pc.ply", coord, normal)
            os.system(
                f"./psr --in {tmpdir}/pc.ply "
                f"--grid {tmpdir}/psr.grid --depth {depth} --scale {self.grid_range}"
            )
            occ_grid_value, T_matrix = load_grid(f"{tmpdir}/psr.grid")

        # PSR uses different primary axis (swap X-axis and Z-axis)
        occ_grid_value = occ_grid_value.transpose(2, 1, 0)
        # Create binary occupancy grid (1 for occupied, 0 for free)
        occ_grid = (occ_grid_value > self.psr_occ_thresh).reshape(-1)
        occupancy = torch.from_numpy(occ_grid).to(self.device)

        # Get the actual range of the normalized and stretched coordinates
        coord_min, coord_max = self.scaled_uvgrid.calculate_bounding_box()

        # Adjust occupancy values for points outside the valid range
        for axis in range(3):
            outside_mask = (self.scaled_meshgrid[:, axis] < coord_min[axis]) | (
                self.scaled_meshgrid[:, axis] > coord_max[axis]
            )
            occupancy[outside_mask] = False

        return occupancy

    def vote_occupancy(
        self,
        occupancy: torch.Tensor,
        partition: torch.Tensor,
        n_partition: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs occupancy voting based on partitions.
        """
        # Flatten partition grid to align with voxel indices
        partition_flat = partition.flatten()

        # Get partition indices for each voxel
        partition_indices = partition_flat

        # Scatter for occupancy voting
        scatter_src = torch.cat(
            [
                occupancy.float(),
                torch.zeros(n_partition, device=self.device, dtype=torch.float32),
            ]
        )
        scatter_idx = torch.cat(
            [
                partition_indices.long(),
                torch.arange(n_partition, device=self.device, dtype=torch.int64),
            ]
        )

        # Compute occupancy rate per partition
        occ_rate = scatter(scatter_src, scatter_idx, reduce="mean")
        partition_occ_bit = occ_rate > self.partition_occ_thresh

        # Map partitions back to occupancy grid
        partition_occ = partition_occ_bit[partition_flat].view(
            self.grid_res, self.grid_res, self.grid_res
        )
        partition_occ_rate = occ_rate[partition_flat].view(
            self.grid_res, self.grid_res, self.grid_res
        )

        return partition_occ_bit, partition_occ, partition_occ_rate

    def _uvgrid2wn_single_face(self, f_idx: int) -> torch.Tensor:
        """
        Computes the generalized winding numbers for a single face.
        """
        meshgrid_np = self.scaled_meshgrid.cpu().numpy()

        # Generate mesh for the face
        v, f = self.scaled_uvgrid_extend.meshify(face_idx=f_idx, use_grid_mask=False)

        # Compute winding numbers
        if self.use_fwn:
            wn = igl.fast_winding_number_for_meshes(v, f, meshgrid_np)
        else:
            wn = igl.winding_number(v, f, meshgrid_np)

        wn = torch.from_numpy(wn).to(device=self.device)
        return wn

    @staticmethod
    def _wn(args: Tuple[np.ndarray, np.ndarray, np.ndarray, bool]) -> np.ndarray:
        """
        Computes the generalized winding numbers for a single face.
        :param query_coord: N x 3 query coordinates
        :param v: V x 3 mesh vertices
        :param f: F x 3 mesh faces
        :param use_fwn: use fast-winding numbers
            If false, use vanilla winding numbers which is slower but exact
        """
        # Compute winding numbers
        query_coord, v, f, use_fwn = args

        if use_fwn:
            wn = igl.fast_winding_number_for_meshes(v, f, query_coord)
        else:
            wn = igl.winding_number(v, f, query_coord)
        return wn

    def get_extended_partition_grid(self) -> Tuple[torch.Tensor, int]:
        """
        Computes the extended partition grid using winding numbers from all faces.
        """
        dims = (self.grid_res,) * 3

        assert (
            len(self.scaled_uvgrid.coord) < 64
        ), f"Number of faces exceeds 64, got {len(self.scaled_uvgrid.coord)}"
        partition_buffer = torch.zeros(
            self.grid_res**3, dtype=torch.int64, device=self.device
        )

        # partition with winding numbers
        if self.n_workers == 0:
            for i in range(len(self.scaled_uvgrid.empty_mask)):
                if not self.scaled_uvgrid.empty_mask[i]:
                    # Compute winding numbers for the face
                    wn_i = self._uvgrid2wn_single_face(i)
                    # Create a bitmask for the partition
                    partition_bit = (2**i) * (wn_i > 0).type(torch.int64)
                    # Accumulate partition bits
                    partition_buffer += partition_bit
        else:
            # multiprocessing
            # currently is slower than single processing
            meshgrid_np = self.scaled_meshgrid.cpu().numpy()
            wn_input = []
            for i in range(len(self.scaled_uvgrid.empty_mask)):
                if not self.scaled_uvgrid.empty_mask[i]:
                    v, f = self.scaled_uvgrid.meshify(face_idx=i, use_grid_mask=False)
                    wn_input.append((meshgrid_np, v, f, self.use_fwn))

            wn_iter = Pool(4).imap(self._wn, wn_input)
            for i, wn_i in enumerate(wn_iter):
                wn_mask = torch.Tensor(wn_i > 0).to(self.device).type(torch.int64)
                # Create a bitmask for the partition
                partition_bit = (2**i) * wn_mask
                # Accumulate partition bits
                partition_buffer += partition_bit

        # Get unique partition labels and remap them to sequential indices
        unique, invmap = torch.unique(partition_buffer, return_inverse=True)
        new_indices = torch.arange(len(unique), device=self.device, dtype=torch.int64)
        new_partition_buffer = new_indices[invmap]

        # Reshape partition buffer to grid dimensions
        partition_grid = new_partition_buffer.view(*dims)
        return partition_grid, len(unique)

    def create_mesh_from_partition_occ(
        self,
        partition_occ: torch.Tensor,
    ) -> trimesh.Trimesh:
        """
        Creates a mesh from the partition occupancy grid using the marching cubes algorithm.

        Args:
            partition_occ: Tensor of shape (grid_res, grid_res, grid_res) containing occupancy
                        (1 for occupied, 0 for free space).

        Returns:
            mesh: A trimesh.Trimesh object representing the mesh extracted from the occupancy grid.
        """
        # Convert partition_occ to a numpy array
        partition_occ_np = partition_occ.cpu().numpy()

        min, max = (
            self.scaled_meshgrid.min(0)[0].cpu().numpy(),
            self.scaled_meshgrid.max(0)[0].cpu().numpy(),
        )
        spacing = (max - min) / (self.grid_res - 1)

        # Run the marching cubes algorithm
        verts, faces, normals, values = measure.marching_cubes(
            volume=partition_occ_np,
            level=0.5,
            spacing=spacing,
        )
        verts += min

        # Create the mesh using trimesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

        return mesh

    def create_meshgrid(self, grid_min_range, grid_max_range) -> torch.Tensor:
        """
        Creates a meshgrid of 3D points in world coordinates covering the specified grid range.
        """
        # Create a 1D grid along each axis within the specified range
        # PoissonRecon returns grid in below range
        one_dim_grid = torch.linspace(
            grid_min_range,
            grid_max_range,
            steps=self.grid_res,
            device=self.device,
        )
        # Create 3D meshgrid coordinates
        grid_x, grid_y, grid_z = torch.meshgrid(
            one_dim_grid, one_dim_grid, one_dim_grid, indexing="ij"
        )
        meshgrid = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(-1, 3)
        return meshgrid

    def _unscale(self, x: Union[torch.Tensor, np.ndarray]):
        """
        Unscale to original coordinates
        :param x: np or torch of N x 3
        :return:
        """
        axis_offset = self.axis_offset.unsqueeze(0)
        axis_scale = self.axis_scale.unsqueeze(0)
        if isinstance(x, np.ndarray):
            axis_offset = axis_offset.cpu().numpy()
            axis_scale = axis_scale.cpu().numpy()
        x_unscaled = (x - axis_offset) / axis_scale
        return x_unscaled

    def get_brep(self, ret_none_if_failed: bool = False, verbose: bool = False):
        """
        :param ret_none_if_failed:
            Returns none if failed.
        :param verbose:
            Whether to print verbose output
        :return:
        """
        if verbose:
            print("Starting postprocessing...")
        pp_out = self.postprocess(
            ret_none_if_failed=ret_none_if_failed, verbose=verbose
        )
        if ret_none_if_failed and (pp_out is None):
            if verbose:
                print("Failed during postprocessing")
            return None

        if verbose:
            print("Postprocessing completed successfully")
            print("Extracting BRep data...")

        brep = pp_out.brep
        if brep is None:
            if verbose:
                print("Failed: BRep is None after postprocessing")
            return None

        surf_wcs = brep.faces.cpu().numpy()
        edge_wcs = [x.cpu().numpy() for x in brep.edges]
        face_edge_adj = brep.face_edge_adj
        edge_vertex_adj = brep.edge_vertex_adj.cpu().numpy()

        # requires occ to be installed
        from brepdgt.postprocessing.occ_brep_builder import construct_brep

        if verbose:
            print("Constructing BRep...")
        try:
            brep = construct_brep(
                surf_wcs=surf_wcs,
                edge_wcs=edge_wcs,
                face_edge_adj=face_edge_adj,
                edge_vertex_adj=edge_vertex_adj,
            )
        except Exception as e:
            if verbose:
                print(f"Failed during BRep construction: {str(e)}")
            if ret_none_if_failed:
                return None
            raise

        if verbose:
            print("BRep construction completed successfully")

        return brep

    @staticmethod
    def _merge_meshes(vs: List[np.ndarray], fs: List[np.ndarray]):
        """
        Merge list of vertices and faces
        :param vs: list of vertices (array N_i x 3)
        :param fs: list of faces (array M_i x 3)
        :return:
        """
        v_out, f_out = [], []
        v_cnt = 0
        for i in range(len(vs)):
            v_out.append(vs[i])
            f = fs[i] + v_cnt
            f_out.append(f)
            v_cnt += vs[i].shape[0]
        v_out = np.concatenate(v_out)
        f_out = np.concatenate(f_out)
        return v_out, f_out

    def vis_interactive(self, ret_scaled=False):
        """
        Interactively visualize debugging process with polyscope.
        Requires polyscope to be installed
        :return:
        """
        import polyscope as ps
        import time

        start_time = time.time()
        pp_out = self.postprocess(ret_scaled=ret_scaled, ret_none_if_failed=False)
        partition = pp_out.partition.cpu().numpy()
        print(f"time took: {time.time() - start_time: .3f} seconds")
        cmap = get_cmap()

        ps.init()
        create_bbox(enabled=False)
        if pp_out.grid_coord is not None:
            #############################
            # Uvgrid extended partition #
            #############################
            meshgrid = pp_out.grid_coord.cpu().numpy()
            ps_partition = ps.register_point_cloud(
                "partition",
                meshgrid,
                radius=0.5 / self.grid_res,
                point_render_mode="sphere",
            )
            ps_partition.add_color_quantity(
                "label",
                cmap[partition.reshape(-1) % len(cmap)][:, :3],
                enabled=False,
            )
            # occupancy rate
            partition_occ_rate = pp_out.partition_occ_rate.cpu().flatten().numpy()
            ps_partition.add_scalar_quantity("occ_rate", partition_occ_rate)
            ps_partition.set_enabled(False)

            #######################
            # Partition occupancy #
            #######################
            partition_occ = pp_out.partition_occ.flatten().bool().cpu()
            ps_partition_occ = ps.register_point_cloud(
                "partition_occ",
                meshgrid[partition_occ],
                radius=0.5 / self.grid_res,
                point_render_mode="sphere",
                enabled=False,
            )

            if pp_out.partition_mesh is not None:
                ##################
                # Partition Mesh #
                ##################
                partition_mesh = pp_out.partition_mesh
                ps_partition_occ_mesh = ps.register_surface_mesh(
                    "partition_occ_mesh",
                    partition_mesh.vertices,
                    partition_mesh.faces,
                    enabled=False,
                )

            if pp_out.occupancy is not None:
                #################
                # Raw occupancy #
                #################
                occupancy = pp_out.occupancy.cpu().numpy()
                raw_occ_coord = meshgrid[occupancy]
                ps_raw_occ = ps.register_point_cloud(
                    "raw_occupancy",
                    raw_occ_coord,
                    radius=0.5 / self.grid_res,
                    point_render_mode="sphere",
                    enabled=False,
                )

        if pp_out.uvgrid is not None:
            ###########################
            # (Scaled) Uv grid points #
            ###########################
            uvgrid_out = pp_out.uvgrid
            uvgrid_coords, uvgrid_coord_colors, uvgrid_normals = [], [], []
            for i in range(len(uvgrid_out.coord)):
                uvgrid_coord = uvgrid_out.coord[i][~uvgrid_out.empty_mask[i]]
                uvgrid_grid_mask = uvgrid_out.grid_mask[i][~uvgrid_out.empty_mask[i]]
                valid_uvgrid_coord = uvgrid_coord[uvgrid_grid_mask].cpu().numpy()
                uvgrid_coords.append(valid_uvgrid_coord)
                color = cmap[i % len(cmap)].reshape(1, -1)
                uvgrid_coord_colors.append(
                    color.repeat(valid_uvgrid_coord.shape[0], axis=0)
                )
                uvgrid_normal = uvgrid_out.normal[i][~uvgrid_out.empty_mask[i]]
                uvgrid_normals.append(uvgrid_normal[uvgrid_grid_mask].cpu().numpy())
            uvgrid_coords = np.concatenate(uvgrid_coords)
            ps_uvgrid_coord = ps.register_point_cloud(
                "uvgrid_points",
                uvgrid_coords,
                radius=1.0 / self.grid_res,
                point_render_mode="sphere",
                enabled=False,
            )
            uvgrid_coord_colors = np.concatenate(uvgrid_coord_colors)
            ps_uvgrid_coord.add_color_quantity(
                "face_idx", uvgrid_coord_colors, enabled=True
            )
            uvgrid_normals = np.concatenate(uvgrid_normals)
            ps_uvgrid_coord.add_vector_quantity("normal", uvgrid_normals, enabled=True)

            if self.uvgrid.empty_mask is not None:
                ##################
                # Uv grid meshes #
                ##################
                v, f = [], []
                for i in range(len(self.uvgrid.empty_mask)):
                    v_i, f_i = self.uvgrid.meshify(i, use_grid_mask=False)
                    v.append(v_i)
                    f.append(f_i)

                v, f = self._merge_meshes(v, f)

                # register uv grid mesh
                ps_mesh = ps.register_surface_mesh("uvgrid_mesh", v, f, enabled=False)

        if pp_out.extended_uvgrid is not None:
            ###################
            # Extended uvgrid #
            ###################
            uvgrid_extend = pp_out.extended_uvgrid
            v_extend, f_extend = [], []
            colors_uvgrid_extended = []
            for i in range(len(uvgrid_extend.coord)):
                v_i, f_i = uvgrid_extend.meshify(i, use_grid_mask=False)
                v_extend.append(v_i)
                f_extend.append(f_i)
                color = cmap[i % len(cmap)].reshape(1, -1)
                colors_uvgrid_extended.append(color.repeat(f_i.shape[0], axis=0))
            v_extend, f_extend = self._merge_meshes(v_extend, f_extend)

            # register uv grid mesh
            ps_extended_uvgrid_mesh = ps.register_surface_mesh(
                "extended_uvgrid_mesh", v_extend, f_extend, enabled=False
            )
            colors_uvgrid_extended = np.concatenate(colors_uvgrid_extended, axis=0)
            ps_extended_uvgrid_mesh.add_color_quantity(
                "uvgrid_idx", colors_uvgrid_extended, defined_on="faces", enabled=True
            )

        if pp_out.patches is not None:
            ##########################
            # Patches & Wrap patches #
            ##########################
            patches = pp_out.patches
            v_patch, f_patch, colors_patch = [], [], []
            v_wrap_patch, f_wrap_patch, colors_wrap_patch = [], [], []
            v_patch_cnt, v_wrap_patch_cnt = 0, 0
            for i, patch in enumerate(patches):
                v_patch.append(np.array(patch.vertices))
                f_patch.append(np.array(patch.faces) + v_patch_cnt)
                v_patch_cnt += patch.vertices.shape[0]
                color = cmap[i % len(cmap)].reshape(1, -1)
                colors_patch.append(color.repeat(patch.faces.shape[0], axis=0))

                if pp_out.wrap_patch_mask[i]:
                    v_wrap_patch.append(np.array(patch.vertices))
                    f_wrap_patch.append(np.array(patch.faces) + v_wrap_patch_cnt)
                    v_wrap_patch_cnt += patch.vertices.shape[0]
                    colors_wrap_patch.append(color.repeat(patch.faces.shape[0], axis=0))

            v_patch = np.concatenate(v_patch)
            f_patch = np.concatenate(f_patch)
            ps_patch_mesh = ps.register_surface_mesh(
                "patch", v_patch, f_patch, enabled=False
            )
            colors_patch = np.concatenate(colors_patch, axis=0)
            ps_patch_mesh.add_color_quantity(
                "patch", colors_patch, defined_on="faces", enabled=True
            )

            v_wrap_patch = np.concatenate(v_wrap_patch)
            f_wrap_patch = np.concatenate(f_wrap_patch)
            ps_wrap_patch_mesh = ps.register_surface_mesh(
                "wrap_patch", v_wrap_patch, f_wrap_patch
            )
            colors_wrap_patch = np.concatenate(colors_wrap_patch, axis=0)
            ps_wrap_patch_mesh.add_color_quantity(
                "wrap_patch", colors_wrap_patch, defined_on="faces", enabled=True
            )

        if pp_out.wrap_patch_query is not None:
            ####################
            # Wrap patch query #
            ####################
            wrap_patch_query = pp_out.wrap_patch_query.cpu().numpy()
            wrap_patch_query_occ = wrap_patch_query[:, 3]
            wrap_patch_query_coord = wrap_patch_query[:, :3]
            ps_uvgrid_coord = ps.register_point_cloud(
                "patch_query_coord",
                wrap_patch_query_coord,
                radius=0.5 / self.grid_res,
                point_render_mode="sphere",
                enabled=False,
            )
            ps_uvgrid_coord.add_scalar_quantity(
                "occ", wrap_patch_query_occ, enabled=True
            )

        if pp_out.brep is not None:
            ##############
            # Brep Faces #
            ##############
            brep = pp_out.brep
            face_coords, face_colors = [], []
            for i, face_coord in enumerate(brep.faces):
                face_coord = face_coord.cpu().view(-1, 3).numpy()
                face_coords.append(face_coord)
                color = cmap[i % len(cmap)].reshape(1, -1)
                face_colors.append(color.repeat(face_coord.shape[0], axis=0))
            face_coords = np.concatenate(face_coords)
            face_colors = np.concatenate(face_colors)
            ps_brep_faces = ps.register_point_cloud(
                "brep_faces",
                face_coords,
                radius=0.5 / self.grid_res,
                point_render_mode="sphere",
                enabled=False,
            )
            ps_brep_faces.add_color_quantity("edge", face_colors, enabled=True)

            ##############
            # Brep Edges #
            ##############
            edge_coords, edge_colors = [], []
            for i, edge_coord in enumerate(brep.edges):
                edge_coords.append(edge_coord.cpu().numpy())
                color = cmap[i % len(cmap)].reshape(1, -1)
                edge_colors.append(color.repeat(edge_coord.shape[0], axis=0))
            edge_coords = np.concatenate(edge_coords)
            edge_colors = np.concatenate(edge_colors)
            ps_brep_edges = ps.register_point_cloud(
                "brep_edges",
                edge_coords,
                radius=0.5 / self.grid_res,
                point_render_mode="sphere",
                enabled=True,
            )
            ps_brep_edges.add_color_quantity("edge", edge_colors, enabled=True)

            #################
            # Brep Vertices #
            #################
            ps_brep_vertices = ps.register_point_cloud(
                "brep_vertices",
                brep.vertices.cpu().numpy(),
                radius=0.5 / self.grid_res,
                point_render_mode="sphere",
                enabled=True,
            )

        if pp_out.defect_edges is not None:
            #################################################
            # defect edges induced by self-intersection bug #
            #################################################
            nodes = pp_out.merged_mesh.vertices
            edges = pp_out.defect_edges.cpu().numpy()
            ps_defect_edges = ps.register_curve_network(
                "defect edges", nodes, edges, enabled=False
            )

        # slice plane
        ps_plane = ps.add_scene_slice_plane()
        ps_plane.set_draw_plane(False)  # render the semi-transparent gridded plane
        ps_plane.set_draw_widget(True)

        ps.show()


class BatchPostprocessor:
    """
    Wrapper for postprocessor with additional features as saving step or stl files
        + aggregates statistics regarding validity of postprocessing
    """

    def __init__(
        self,
        out_dir: str,
        multiple_param_extension: bool,
        coarse_to_fine: bool,
        **kwargs,
    ):
        """

        :param out_dir:
        :param multiple_param_extension:
            Try param extension with both 1 and 0.5
        :param kwargs: keyword arguments for postprocessor
        """
        self.out_dir = out_dir
        self.multiple_param_extension = multiple_param_extension
        self.coarse_to_fine = coarse_to_fine
        assert not (
            self.multiple_param_extension and self.coarse_to_fine
        ), "both options on are not allowed"
        self.kwargs = kwargs

        self.step_dir = os.path.join(self.out_dir, "step")
        os.makedirs(self.step_dir, mode=0o777, exist_ok=True)
        self.names = []
        self.watertight_masks = []
        self.n_faces = []

    def process_one(
        self, uvgrid: UvGrid, name: str, timeout: int = 30
    ) -> Tuple[bool, int]:
        """
        Creates brep from uvgrid and save step and stl files
        :param: uvgrid
            Uvgrid to process
        :param: name
            Name to save
        :return:
        """
        from OCC.Extend.DataExchange import write_step_file

        if self.multiple_param_extension:
            self.kwargs["uvgrid_extension_len"] = 1.0
        if self.coarse_to_fine:
            # try low-res first
            self.kwargs["grid_res"] = 128

        is_watertight, n_faces = False, 0

        try:
            pp = Postprocessor(uvgrid, **self.kwargs)
            timeout_pp = timeout_wrapper(pp.get_brep)
            brep = timeout_pp(ret_none_if_failed=True, timeout=timeout)
            if brep is None:
                raise RuntimeError()

            step_out_path = os.path.join(self.step_dir, name + ".step")
            write_step_file(brep, step_out_path)
            is_watertight, n_faces = check_solid(step_path=step_out_path, timeout=20)
        except Exception as e:
            print(e)

        if (not is_watertight) and (
            self.multiple_param_extension or self.coarse_to_fine
        ):
            # try again
            if self.multiple_param_extension:
                self.kwargs["uvgrid_extension_len"] = 0.5
            elif self.coarse_to_fine:
                self.kwargs["grid_res"] = 256
            try:
                pp = Postprocessor(uvgrid, **self.kwargs)
                timeout_pp = timeout_wrapper(pp.get_brep)
                brep = timeout_pp(ret_none_if_failed=True, timeout=timeout)
                if brep is None:
                    raise RuntimeError()

                step_out_path = os.path.join(self.step_dir, name + ".step")
                write_step_file(brep, step_out_path)
                is_watertight, n_faces = check_solid(
                    step_path=step_out_path, timeout=20
                )
            except Exception as e:
                print(e)

        self.names.append(name)
        self.watertight_masks.append(is_watertight)
        self.n_faces.append(n_faces)

        return is_watertight, n_faces
