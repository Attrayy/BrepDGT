from __future__ import annotations

import numpy as np
import torch
import dgl
from typing import List, Dict, Tuple, Union, Optional, Any
from typing_extensions import Self
from brepdgt.utils.common import downsample, clamp_norm

KEYS_TO_SERIALIZE = [
    "coord",
    "grid_mask",
    "empty_mask",
    "normal",
    "prim_type",
    "face_adj",
    "coord_logits",
]


class UvGrid:
    """
    A class for storing uv grids associcated with each face (3D)
    """

    eps = 1e-2  # to set coordinates near zero be masked

    def __init__(
        self,
        coord: Union[np.ndarray, torch.Tensor],
        grid_mask: Union[np.ndarray, torch.Tensor, None],
        empty_mask: Union[np.ndarray, torch.Tensor, None],
        normal: Union[np.ndarray, torch.Tensor, None] = None,
        prim_type: Union[np.ndarray, torch.Tensor, None] = None,
        face_adj: Union[dgl.DGLGraph, List[dgl.DGLGraph]] = None,
        coord_logits: Union[np.ndarray, torch.Tensor] = None,
    ):
        """
        :param coord: float array of n_prims x n_grid x n_grid x 3 (or B x n_prims x n_grid x n_grid x 3)
            coordinates of each uv grid
        :param grid_mask: bool array of n_prims x n_grid x n_grid x 3 (or B x n_prims x n_grid x n_grid x 3)
            whether each uv grid lies on surface. True indicates that grid lies on surface
        :param empty_mask: bool array of n_prims (or B x n_prims)
            whether each face is empty or not. True indicates face is empty
        :param normal: float array of n_prims x n_grid x n_grid x 3 (or B x n_prims x n_grid x n_grid x 3)
            normal of each uv grid
        :param prim_type: long array of n_prims (or B x n_prims)
            primitive type of each face

        """
        self.coord = coord
        self.grid_mask = grid_mask  # True indicates that coord lies on surface
        self.empty_mask = empty_mask  # True indicates empty!
        self.normal = normal
        self.prim_type = prim_type
        self.face_adj = face_adj
        self.coord_logits = coord_logits

    def tensorize(self, normalize=False):
        """
        Stack and tensorize attributes of uvgrid (coords, grid_masks, normals).
        :param normalize: map values to [-1, 1] (such as grid masks)
        :return:
        """
        grid_vals = [self.coord]
        if self.grid_mask is not None:
            grid_mask = self.grid_mask[:, :, :, :, np.newaxis]
            if isinstance(self.grid_mask, np.ndarray):
                grid_mask = grid_mask.astype(np.float32)
            else:
                grid_mask = grid_mask.float()
            if normalize:
                grid_mask = 2 * grid_mask - 1
            grid_vals.append(grid_mask)
        if self.normal is not None:
            grid_vals.append(self.normal)

        if isinstance(self.coord, np.ndarray):
            grid_vals = np.concatenate(grid_vals, axis=-1)
        elif isinstance(self.coord, torch.Tensor):
            grid_vals = torch.cat(grid_vals, dim=-1)
        else:
            raise ValueError(f"{type(self.coord)} not allowed")
        return grid_vals

    @staticmethod
    def _to_npy(x: Union[torch.Tensor, np.ndarray, None]):
        if x is None:
            return x
        if isinstance(x, np.ndarray):
            return x
        elif isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        else:
            raise ValueError(f"type: {type(x)} not allowed")

    @staticmethod
    def _to_tensor(x: Union[torch.Tensor, np.ndarray, None], device: str):
        if x is None:
            return x
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x.to(device)

    def to_tensor(self, device) -> None:
        """
        Converts attributes to tensor and puts on given device
        :param self:
        :param device:
        :return:
        """
        self.coord = self._to_tensor(self.coord, device)
        self.grid_mask = self._to_tensor(self.grid_mask, device)
        self.empty_mask = self._to_tensor(self.empty_mask, device)
        self.normal = self._to_tensor(self.normal, device)
        self.prim_type = self._to_tensor(self.prim_type, device)
        self.face_adj = self._to_tensor(self.face_adj, device)

    @staticmethod
    def _pad_face_adj(
        face_adj,
        max_n_prims,
    ):
        n = face_adj.shape[0]
        if n > max_n_prims:
            raise ValueError("face_adj size is larger than max_n_prims.")

        is_tensor = isinstance(face_adj, torch.Tensor)

        if is_tensor:
            padded_adj = torch.zeros(
                (max_n_prims, max_n_prims), dtype=face_adj.dtype, device=face_adj.device
            )
        else:
            padded_adj = np.zeros((max_n_prims, max_n_prims), dtype=face_adj.dtype)

        padded_adj[:n, :n] = face_adj

        return padded_adj

    @classmethod
    def from_raw_values(
        cls,
        coord: Union[np.ndarray, torch.Tensor],
        grid_mask: Union[np.ndarray, torch.Tensor, None],
        max_n_prims: int,
        normal: Union[np.ndarray, torch.Tensor, None] = None,
        prim_type: Union[np.ndarray, torch.Tensor, None] = None,
    ) -> Self:
        """
        Creates uvgrid from raw, unpadded values
        :param coord: float array of n_prims x n_grid x n_grid x 3 (or B x n_prims x n_grid x n_grid x 3)
            coordinates of each uv grid
        :param grid_mask: bool array of n_prims x n_grid x n_grid x 3 (or B x n_prims x n_grid x n_grid x 3)
            whether each uv grid lies on surface
        :param max_n_prims: int
            maximum number of allowed primitives. All the arrays are padded up to this value
        :param normal: float array of n_prims x n_grid x n_grid x 3 (or B x n_prims x n_grid x n_grid x 3)
            normal of each uv grid
        :param prim_type: long array of n_prims (or B x n_prims)
            primitive type of each face
        :return:
        """
        if len(coord.shape) == 4:
            pad_dim = 0
        else:
            raise ValueError(f"coord shape: {coord.shape} not allowed")
        empty_mask = UvGrid._get_empty_mask(coord, pad_dim, max_n_prims)

        # sample random indices for padding with duplicates
        n_prims = coord.shape[0]
        # uniformly fill pad_idxs
        pad_idxs = np.arange(0, max_n_prims) % n_prims
        n_remainder = max_n_prims - (max_n_prims // n_prims) * n_prims
        # for remainder, sample randomly
        if n_remainder > 0:
            pad_idxs[-n_remainder:] = np.random.permutation(np.arange(n_prims))[
                :n_remainder
            ]

        coord = UvGrid._pad(coord, max_n_prims, pad_idxs)
        grid_mask = UvGrid._pad(grid_mask, max_n_prims, pad_idxs)
        normal = UvGrid._pad(normal, max_n_prims, pad_idxs)
        prim_type = UvGrid._pad(prim_type, max_n_prims, pad_idxs)

        uvgrid = UvGrid(
            coord=coord,
            grid_mask=grid_mask,
            empty_mask=empty_mask,
            normal=normal,
            prim_type=prim_type,
        )
        return uvgrid

    @staticmethod
    def _get_empty_mask(
        coord: Union[np.ndarray, torch.Tensor], pad_dim: int, max_n_prims: int
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Pads given array as empty (zero array) so that the n_prims match max_n_prims
        :param coord:
        :param pad_dim: int
            dimension to apply padding
        :param max_n_prims: int
            maximum number of primitives
        :return:
        """
        empty_mask = np.zeros(max_n_prims, dtype=bool)
        empty_mask[coord.shape[pad_dim] :] = True
        if isinstance(coord, torch.Tensor):
            empty_mask = torch.from_numpy(empty_mask)
        return empty_mask

    @staticmethod
    def _pad(
        x: Union[np.ndarray, torch.Tensor, None],
        max_n_prims: int,
        pad_idxs: Union[np.ndarray, torch.Tensor, None] = None,
    ) -> Union[np.ndarray, torch.Tensor, None]:
        """
        Pads given array as empty (zero array) so that the n_prims match max_n_prims
        :param x: array to pad, should be shape of n_prims x *
        :param pad_dim: int
            dimension to apply padding
        :param max_n_prims: int
            maximum number of primitives
        :param pad_idxs: how to pad when randomly padding
        :return:
        """
        if x is None:
            return x
        n_prims = x.shape[0]
        n_pad = max_n_prims - n_prims
        pad_shape = list(x.shape)
        pad_shape[0] = n_pad
        if isinstance(x, np.ndarray):
            pad = np.zeros(pad_shape, dtype=x.dtype)
            x = np.concatenate([x, pad])
        elif isinstance(x, torch.Tensor):
            pad = torch.zeros(pad_shape, dtype=x.dtype)
            x = torch.cat([x, pad])
        else:
            raise ValueError(f"type {type(x)} not allowed")
        return x

    def export_npz(self, file_path: str, vis_idx: Union[int, None] = None):
        """
        Export the whole class as npz file
        :param file_path:
        :param vis_idx:
            idx to save
        :return:
        """

        coord = self.coord
        grid_mask = self.grid_mask
        empty_mask = self.empty_mask
        normal = self.normal
        prim_type = self.prim_type

        if isinstance(self.coord, torch.Tensor):
            coord = self._to_npy(coord)
            grid_mask = self._to_npy(grid_mask)
            empty_mask = self._to_npy(empty_mask)
            normal = self._to_npy(normal)
            prim_type = self._to_npy(prim_type)

        save_dict = {}
        if vis_idx is not None:
            assert len(coord.shape) == 5, f"got coord shape of {coord.shape}"
            save_dict["coord"] = coord[vis_idx]
            if grid_mask is not None:
                save_dict["grid_mask"] = grid_mask[vis_idx]
            if empty_mask is not None:
                save_dict["empty_mask"] = empty_mask[vis_idx]
            if normal is not None:
                save_dict["normal"] = normal[vis_idx]
            if prim_type is not None:
                save_dict["prim_type"] = prim_type[vis_idx]
        else:
            save_dict["coord"] = coord
            if grid_mask is not None:
                save_dict["grid_mask"] = grid_mask
            if empty_mask is not None:
                save_dict["empty_mask"] = empty_mask
            if normal is not None:
                save_dict["normal"] = normal
            if prim_type is not None:
                save_dict["prim_type"] = prim_type

        np.savez(file_path, **save_dict)

    @classmethod
    def load_from_npz_data(
        cls, npz_data: Dict, max_prims: Union[int, None] = None
    ) -> Self:
        """
        Loads uvgrid from npz_path
        :param npz_data: Dict loaded from np.load()
        :return:
        """
        if max_prims is not None:
            uvgrid = UvGrid(
                coord=npz_data["coord"][:max_prims],
                grid_mask=npz_data.get("grid_mask")[:max_prims],
                empty_mask=npz_data["empty_mask"][:max_prims],
                normal=npz_data.get("normal")[:max_prims],
                prim_type=(
                    npz_data.get("prim_type")[:max_prims]
                    if "prim_type" in npz_data
                    else None
                ),
            )
        else:
            uvgrid = UvGrid(
                coord=npz_data["coord"],
                grid_mask=npz_data.get("grid_mask"),
                empty_mask=npz_data["empty_mask"],
                normal=npz_data.get("normal"),
                prim_type=(
                    npz_data.get("prim_type") if "prim_type" in npz_data else None
                ),
            )
        return uvgrid

    def meshify(
        self, face_idx: int, use_grid_mask: bool = True, batch_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Creates uv grid mesh for a face

        Args:
            face_idx: Index of face/primitive to meshify
            use_grid_mask: Whether to use grid mask to filter vertices
            batch_idx: Which batch item to meshify. If None, assumes unbatched data
        Returns:
            - vertices: array of V x 3
            - faces: array of idx of F x 3
        """
        # Handle batched case by selecting correct batch
        is_batched = len(self.coord.shape) == 5
        if is_batched:
            if batch_idx is None:
                raise ValueError("batch_idx must be provided for batched UvGrid")
            coord = self.coord[batch_idx]
            grid_mask = (
                self.grid_mask[batch_idx] if self.grid_mask is not None else None
            )
        else:
            coord = self.coord
            grid_mask = self.grid_mask

        if use_grid_mask and (grid_mask is not None):
            grid_mask = self._to_npy(grid_mask)
            grid_mask = grid_mask[face_idx]
        else:
            grid_mask = np.ones(coord[face_idx].shape[:2], dtype=bool)

        coord = self._to_npy(coord)
        coord = coord[face_idx]

        # create valid vertices from grid_mask and mapping from uv to vertex idx
        vertices, faces = [], []
        uv2vertex_idx = -1 * np.ones(coord.shape[:2], dtype=np.int32)
        for i in range(coord.shape[0]):
            for j in range(coord.shape[1]):
                if not grid_mask[i, j]:
                    continue
                uv2vertex_idx[i, j] = len(vertices)
                vertices.append(coord[i, j])

        # create faces
        for i in range(coord.shape[0] - 1):
            for j in range(coord.shape[1] - 1):
                if not grid_mask[i, j]:
                    continue
                if not grid_mask[i + 1, j + 1]:
                    continue
                if grid_mask[i + 1, j]:
                    faces.append(
                        (
                            uv2vertex_idx[i, j],
                            uv2vertex_idx[i + 1, j],
                            uv2vertex_idx[i + 1, j + 1],
                        )
                    )
                if grid_mask[i, j + 1]:
                    faces.append(
                        (
                            uv2vertex_idx[i, j],
                            uv2vertex_idx[i + 1, j + 1],
                            uv2vertex_idx[i, j + 1],
                        )
                    )

        if len(faces) == 0:
            vertices = np.zeros((3, 3))
            faces = np.array([[0, 1, 2]]).astype(np.int64)
        else:
            vertices = np.stack(vertices, axis=0)
            faces = np.stack(faces, axis=0)

        return vertices, faces

    def sample_pts(self, num_sample: int) -> torch.Tensor:
        """
        Samples pts from the uvgrid.
        Number of samples match the {num_sample} value
        :param num_sample:
        :return:
            Tensor of B x num_sample x 3
        """
        valid_coords = []
        for batch_idx in range(len(self.coord)):
            empty_mask = self.empty_mask[batch_idx]
            coord = self.coord[batch_idx][~empty_mask]
            if self.grid_mask is None:
                grid_mask = torch.mean(torch.abs(coord), dim=3) > self.eps
            else:
                grid_mask = self.grid_mask[batch_idx][~empty_mask]
            valid_coord = coord[grid_mask]

            # downsample coords to match num_sample
            valid_coord = downsample(valid_coord, num_sample)
            valid_coords.append(valid_coord)
        valid_coords = torch.stack(valid_coords, dim=0)
        return valid_coords

    def calculate_bounding_box(
        self, nonempty_only: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the axis-aligned bounding box for the UV grid coordinates.

        :param nonempty_only: Specifies whether to consider nonempty uv grids
                        in the computations
        :return: Tuple of (min_coords, max_coords) with shape [3] representing the bounding box
        """
        considered_coords = self.coord
        if nonempty_only:
            considered_coords = considered_coords[~self.empty_mask]
        # Min and max are taken over all primitives and grid points
        coord_min = torch.min(considered_coords.view(-1, 3), dim=0)[0]  # [3]
        coord_max = torch.max(considered_coords.view(-1, 3), dim=0)[0]  # [3]
        return coord_min, coord_max

    def normalize_to_cube(
        self, nonempty_only: bool = False
    ) -> Tuple[Self, torch.Tensor, torch.Tensor]:
        """
        Normalizes the UV grid coordinates by:
        1. Centering at origin
        2. Scaling so the longest axis fits in [-1,1]

        :param nonempty_only: Specifies whether to consider nonempty uv grids
                              in the computations

        :return: Tuple of (normalized_uvgrid, scale, offset)
                normalized_uvgrid: New UvGrid with normalized coordinates
                scale: Tensor of shape (3,), uniform scaling factor for all axes
                offset: Tensor of shape (3,), translation to origin
        """
        # Get the bounding box
        coord_min, coord_max = self.calculate_bounding_box(nonempty_only=nonempty_only)

        # Center the coordinates
        center = (coord_max + coord_min) / 2
        offset = -center

        # Scale based on longest axis
        axis_ranges = coord_max - coord_min
        max_range = axis_ranges.max()
        if max_range > 0:
            scale = torch.ones_like(axis_ranges) * (2.0 / max_range)
        else:
            scale = torch.ones_like(axis_ranges)

        # Apply normalization
        coord_normalized = (self.coord + offset) * scale

        # Scale the normals (if they exist)
        if self.normal is not None:
            normal_scaled = self.normal * scale
            normal_scaled = torch.nn.functional.normalize(normal_scaled, p=2, dim=-1)
        else:
            normal_scaled = None

        # Create a new UvGrid instance with the normalized coordinates
        uvgrid_normalized = UvGrid(
            coord=coord_normalized,
            grid_mask=self.grid_mask,
            empty_mask=self.empty_mask,
            normal=normal_scaled,
            prim_type=self.prim_type,
        )

        return uvgrid_normalized, scale, offset

    def stretch_axis_under_threshold(
        self,
        range_min=-0.5,
        range_max=0.5,
        threshold: float = 1.0,
        normalize: bool = True,
    ) -> Tuple[Self, torch.Tensor, torch.Tensor]:
        """
        Rescales the UV grid coordinates so that each axis is scaled separately to fit into a cubic shape.
        Scales an axis only if its range is below a specified threshold.

        :param range_min: Lower bound of the target cubic range (default: -0.5)
        :param range_max: Upper bound of the target cubic range (default: 0.5)
        :param threshold: Threshold for the axis range below which scaling is applied
        :param normalize: If True, first normalize coordinates to [-1,1] based on longest axis
        :return: Tuple of (new_uvgrid, axis_scale, axis_offset)
                new_uvgrid: New UvGrid with normalized coordinates
                axis_scale: Tensor of shape (3,), scaling factors for each axis
                axis_offset: Tensor of shape (3,), offset for each axis
        """
        if normalize:
            # First normalize to [-1,1] cube
            uvgrid_norm, norm_scale, norm_offset = self.normalize_to_cube()

            # Get the bounding box of normalized coordinates
            coord_min, coord_max = uvgrid_norm.calculate_bounding_box()
            axis_ranges = coord_max - coord_min

            # Initialize stretch transforms
            stretch_scale = torch.ones_like(axis_ranges)
            stretch_offset = torch.zeros_like(axis_ranges)

            target_range = range_max - range_min

            # Apply stretching only to axes below threshold
            for i in range(3):
                if threshold is not None and axis_ranges[i] < threshold:
                    stretch_scale[i] = target_range / axis_ranges[i]
                    stretch_offset[i] = -stretch_scale[i] * coord_min[i] + range_min

            # Apply stretching to normalized coordinates
            coord_scaled = uvgrid_norm.coord * stretch_scale + stretch_offset

            # Combine transformations
            # For a point p: p_final = ((p + norm_offset) * norm_scale) * stretch_scale + stretch_offset
            # So final scale and offset are:
            axis_scale = norm_scale * stretch_scale
            axis_offset = norm_offset * norm_scale * stretch_scale + stretch_offset

            if uvgrid_norm.normal is not None:
                normal_scaled = uvgrid_norm.normal * stretch_scale
                normal_scaled = torch.nn.functional.normalize(
                    normal_scaled, p=2, dim=-1
                )
            else:
                normal_scaled = None

            uvgrid_scaled = UvGrid(
                coord=coord_scaled,
                grid_mask=self.grid_mask,
                empty_mask=self.empty_mask,
                normal=normal_scaled,
                prim_type=self.prim_type,
            )

            return uvgrid_scaled, axis_scale, axis_offset

        coord_min, coord_max = self.calculate_bounding_box()
        axis_ranges = coord_max - coord_min

        axis_scale = torch.ones_like(axis_ranges)
        axis_offset = torch.zeros_like(axis_ranges)

        target_range = range_max - range_min

        for i in range(3):
            if threshold is not None and axis_ranges[i] < threshold:
                axis_scale[i] = target_range / axis_ranges[i]
                axis_offset[i] = -axis_scale[i] * coord_min[i] + range_min

        coord_scaled = self.coord * axis_scale + axis_offset

        if self.normal is not None:
            normal_scaled = self.normal * axis_scale
            normal_scaled = torch.nn.functional.normalize(normal_scaled, p=2, dim=-1)
        else:
            normal_scaled = None

        uvgrid_scaled = UvGrid(
            coord=coord_scaled,
            grid_mask=self.grid_mask,
            empty_mask=self.empty_mask,
            normal=normal_scaled,
            prim_type=self.prim_type,
        )

        return uvgrid_scaled, axis_scale, axis_offset

    def extend(
        self,
        extend_len: float,
        smooth_extension: bool = False,
        min_extend_len: float = 0.0,
    ) -> Self:
        """
        Extend the current uvgrid by appending one more uvgrid at each ends
        :param extend_len: Extension length
        :param smooth_extension: bool
            Extends
            When extending uvgrid, consider extension using neighbors from the edges
        :param min_extend_len: float
            Extend so that the minimum extension len is above this value
        :return:
        """

        coord = self.coord[~self.empty_mask]
        cat_fn = torch.cat if isinstance(coord, torch.Tensor) else np.concatenate
        x_res, y_res = coord.shape[1:3]

        if smooth_extension:
            x_src_start = coord[:, 1 : x_res // 2].mean(axis=1)
            x_src_end = coord[:, -x_res // 2 : -2].mean(axis=1)
        else:
            x_src_start = coord[:, 1]
            x_src_end = coord[:, -2]

        x_start_extend = coord[:, 0] - clamp_norm(
            extend_len * (x_src_start - coord[:, 0]), min_extend_len
        )
        x_end_extend = coord[:, -1] - clamp_norm(
            extend_len * (x_src_end - coord[:, -1]), min_extend_len
        )

        coord = cat_fn(
            [x_start_extend.unsqueeze(1), coord, x_end_extend.unsqueeze(1)], axis=1
        )

        if smooth_extension:
            y_src_start = coord[:, :, 1 : y_res // 2].mean(axis=2)
            y_src_end = coord[:, :, -y_res // 2 : -2].mean(axis=2)
        else:
            y_src_start = coord[:, :, 1]
            y_src_end = coord[:, :, -2]

        y_start_extend = coord[:, :, 0] - clamp_norm(
            extend_len * (y_src_start - coord[:, :, 0]), min_extend_len
        )
        y_end_extend = coord[:, :, -1] - clamp_norm(
            extend_len * (y_src_end - coord[:, :, -1]), min_extend_len
        )
        coord = cat_fn(
            [y_start_extend.unsqueeze(2), coord, y_end_extend.unsqueeze(2)],
            axis=2,
        )

        if isinstance(coord, torch.Tensor):
            empty_mask = torch.ones(
                coord.shape[0], device=coord.device, dtype=torch.bool
            )
        else:
            empty_mask = np.ones(coord.shape[0], dtype=bool)

        uvgrid_extend = UvGrid(coord=coord, empty_mask=empty_mask, grid_mask=None)
        return uvgrid_extend

    def sort(self) -> None:
        """
        Sorts the current uvgrid based on center coordinate
        """
        assert (
            len(self.coord.shape) == 4
        ), f"ordering can not be applied in batch, got coord shape of {self.coord.shape}"
        assert (
            self.coord.shape[1] % 2 == 1
        ), f"uvgrid should be odd for ordering, got {self.coord.shape[0]}"
        center_idx = self.coord.shape[1] // 2
        center_coords = self.coord[~self.empty_mask][:, center_idx, center_idx]
        # scale all values to [0, 1]
        min_coord, max_coord = center_coords.min(), center_coords.max()
        # if the diff is too small, then just set it arbitrarily, it does not matter
        if max_coord - min_coord < 1e-3:
            max_coord = min_coord + 10
        center_coords = (center_coords - min_coord) / (max_coord - min_coord)
        center_val = (
            100 * center_coords[:, 0] + 10 * center_coords[:, 1] + center_coords[:, 2]
        )
        if isinstance(center_val, torch.Tensor):
            sort_idxs = torch.argsort(center_val)
        elif isinstance(center_val, np.ndarray):
            sort_idxs = np.argsort(center_val)
        else:
            raise ValueError(f"type {type(center_val)} not allowed")

        # replace exising grids
        self.coord[~self.empty_mask] = self.coord[sort_idxs]
        if self.grid_mask is not None:
            self.grid_mask[~self.empty_mask] = self.grid_mask[sort_idxs]
        if self.empty_mask is not None:
            self.empty_mask[~self.empty_mask] = self.empty_mask[sort_idxs]
        if self.normal is not None:
            self.normal[~self.empty_mask] = self.normal[sort_idxs]
        if self.prim_type is not None:
            self.prim_type[~self.empty_mask] = self.prim_type[sort_idxs]

    def compactify(self) -> None:
        """
        Remove faces with empty_mask so that the uvgrid is compact.
        :return:
        """
        self.coord = self.coord[~self.empty_mask]
        if self.grid_mask is not None:
            self.grid_mask = self.grid_mask[~self.empty_mask]
        if self.normal is not None:
            self.normal = self.normal[~self.empty_mask]
        if self.prim_type is not None:
            self.prim_type = self.prim_type[~self.empty_mask]
        self.empty_mask = self.empty_mask[~self.empty_mask]

    def to_cpu(self) -> None:
        """Move all tensors to CPU"""
        self.coord = (
            self.coord.cpu() if isinstance(self.coord, torch.Tensor) else self.coord
        )
        if self.grid_mask is not None:
            self.grid_mask = (
                self.grid_mask.cpu()
                if isinstance(self.grid_mask, torch.Tensor)
                else self.grid_mask
            )
        if self.empty_mask is not None:
            self.empty_mask = (
                self.empty_mask.cpu()
                if isinstance(self.empty_mask, torch.Tensor)
                else self.empty_mask
            )
        if self.normal is not None:
            self.normal = (
                self.normal.cpu()
                if isinstance(self.normal, torch.Tensor)
                else self.normal
            )
        if self.prim_type is not None:
            self.prim_type = (
                self.prim_type.cpu()
                if isinstance(self.prim_type, torch.Tensor)
                else self.prim_type
            )

    def filter(self, filter_mask: List[bool]) -> UvGrid:
        """
        Returns a uv-grid filtered according to filter_mask.
        WARNING: only works in batched form
        :return:
        """
        assert isinstance(self.coord, torch.Tensor)
        assert len(filter_mask) == self.coord.shape[1]
        coord = self.coord[:, filter_mask]
        grid_mask = (
            self.grid_mask[:, filter_mask] if self.grid_mask is not None else None
        )
        empty_mask = (
            self.empty_mask[:, filter_mask] if self.empty_mask is not None else None
        )
        normal = self.normal[:, filter_mask] if self.normal is not None else None
        prim_type = (
            self.prim_type[:, filter_mask] if self.prim_type is not None else None
        )
        face_adj = self.face_adj[:, filter_mask] if self.face_adj is not None else None
        return UvGrid(
            coord=coord,
            grid_mask=grid_mask,
            empty_mask=empty_mask,
            normal=normal,
            prim_type=prim_type,
            face_adj=face_adj,
        )

    def pad(
        self,
        max_n_prims: int,
        field_names: List[str] = [
            "coord",
            "grid_mask",
            "empty_mask",
            "normal",
            "prim_type",
            "face_adj",
        ],
    ) -> None:
        """
        Pad uv_grids to max_n_prims IN PLACE!
        The extra uv-grids are initialized to zero or random duplicates
        WARNING: only works in batched form
        WARNING: this is different from UvGrid._pad (although inspired from it)
        :return:
        """
        assert isinstance(self.coord, torch.Tensor)
        if self.coord is None:
            return
        n_prims = self.coord.shape[1]  # WARNING! Batched here!
        n_to_pad = max_n_prims - n_prims
        if n_to_pad == 0:
            return

        for field_name in field_names:
            padded_field: torch.Tensor = getattr(self, field_name)
            if padded_field is not None:
                assert isinstance(padded_field, torch.Tensor)
                pad_shape = list(padded_field.shape)
                pad_shape[1] = n_to_pad
                padded_field = torch.cat(
                    [
                        padded_field,
                        torch.zeros(
                            pad_shape,
                            dtype=padded_field.dtype,
                            device=padded_field.device,
                        ),
                    ],
                    dim=1,
                )
                setattr(self, field_name, padded_field)

    @torch.no_grad()
    def serialize(self) -> Dict[str, Any]:
        data = {}
        for k in KEYS_TO_SERIALIZE:
            if hasattr(self, k) and getattr(self, k) is not None:
                val = getattr(self, k)
                if isinstance(val, torch.Tensor):
                    val = val.cpu().numpy()
                elif isinstance(val, np.ndarray):
                    pass
                else:
                    raise ValueError()
                data[k] = val

        return data

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> UvGrid:
        # Easy: just instanciate with everything in data!
        return UvGrid(**data)

    @torch.no_grad()
    def clone(self):
        data = {}
        for k in KEYS_TO_SERIALIZE:
            if hasattr(self, k) and getattr(self, k) is not None:
                val = getattr(self, k)
                data[k] = val.clone()
        return UvGrid(**data)

    @torch.no_grad()
    def extract(self, i_batch: int | List[int] | np.ndarray) -> UvGrid:
        """Extracts the i_batch UvGrid along batch dimension and keeps it in batched
        format."""
        data = {}
        for k in KEYS_TO_SERIALIZE:
            if hasattr(self, k) and getattr(self, k) is not None:
                val = getattr(self, k)
                if isinstance(i_batch, int):
                    data[k] = val[i_batch : i_batch + 1].clone()
                elif isinstance(i_batch, list) or isinstance(i_batch, np.ndarray):
                    data[k] = val[i_batch].clone()
                else:
                    raise ValueError()
        return UvGrid(**data)

    @torch.no_grad()
    def repeat(self, n_repeat: int) -> UvGrid:
        """Repeats along the batch_dimension"""
        assert len(self.coord.shape) == 5
        # assert self.coord.shape[0] == 1
        data = {}
        for k in KEYS_TO_SERIALIZE:
            if hasattr(self, k) and getattr(self, k) is not None:
                val = getattr(self, k)
                if isinstance(val, torch.Tensor):
                    data[k] = val.clone().repeat(
                        [n_repeat] + [1] * (len(val.shape) - 1)
                    )
                elif isinstance(val, np.ndarray):
                    data[k] = val.clone().repeat(n_repeat, axis=0)
                else:
                    raise ValueError()
        return UvGrid(**data)

    @torch.no_grad()
    def unmask_N_first(self, N: int) -> UvGrid:
        """Unmasks N first primitives IN-PLACE"""
        self.empty_mask[:, 0:N] = False
        self.empty_mask[:, N:] = True

    def copy(self) -> UvGrid:
        """Create a deep copy of the UvGrid."""
        return UvGrid(
            coord=self.coord.clone(),
            grid_mask=self.grid_mask.clone() if self.grid_mask is not None else None,
            empty_mask=self.empty_mask.clone(),
            normal=self.normal.clone() if self.normal is not None else None,
            prim_type=self.prim_type.clone() if self.prim_type is not None else None,
        )


def concat_uvgrids(uvgrids: List[UvGrid], cat_dim: int = 1) -> UvGrid:
    """
    Given list of uvgrids, stack the parameters and create a single uvgrid
    :param uvgrids:
    :param cat_dim: dimension over which to concat the uv grids (can be either batch_wise or prim_wise)
    :return:
    """
    if len(uvgrids) == 0:
        raise ValueError("We need at least 1 uv grid to concatenate!")

    if isinstance(uvgrids[0].coord, np.ndarray):
        raise ValueError("Concatenation isn't supported for np.ndarray UvGrids!")

    if len(uvgrids[0].coord.shape) != 5:
        raise ValueError(
            "Concatenation is only supported for batched uv grids for now!"
        )

    if cat_dim > 1 or cat_dim < 0:
        raise ValueError(
            "Concatenation is only supported along batch or n_prim dimension!"
        )

    # Just map all the attributes to the same device before!
    device = uvgrids[0].coord.device

    coord = torch.cat([x.coord.to(device) for x in uvgrids], dim=cat_dim)
    grid_mask = (
        torch.cat([x.grid_mask.to(device) for x in uvgrids], dim=cat_dim)
        if uvgrids[0].grid_mask is not None
        else None
    )
    empty_mask = (
        torch.cat([x.empty_mask.to(device) for x in uvgrids], dim=cat_dim)
        if uvgrids[0].empty_mask is not None
        else None
    )
    normal = (
        torch.cat([x.normal.to(device) for x in uvgrids], dim=cat_dim)
        if uvgrids[0].normal is not None
        else None
    )
    prim_type = (
        torch.cat([x.prim_type.to(device) for x in uvgrids], dim=cat_dim)
        if uvgrids[0].prim_type is not None
        else None
    )
    face_adj = (
        torch.cat([x.face_adj.to(device) for x in uvgrids], dim=cat_dim)
        if uvgrids[0].face_adj is not None
        else None
    )

    ret = UvGrid(
        coord=coord,
        grid_mask=grid_mask,
        empty_mask=empty_mask,
        normal=normal,
        prim_type=prim_type,
        face_adj=face_adj,
    )
    return ret


def uncat_uvgrids(uvgrids: UvGrid, cat_dim: int = 0, stride: int = 1) -> List[UvGrid]:
    """
    Given a single batch of uvgrids, uncat the parameters and create a list of batched uvgrids
    :param uvgrids:
    :param cat_dim: dimension over which to concat the uv grids (can be either batch_wise or prim_wise)
    :return:
    """
    if len(uvgrids.coord.shape) != 5:
        raise ValueError(
            "Un-Concatenation is only supported for batched uv grids for now!"
        )

    if cat_dim != 0:
        raise ValueError(
            "Un-Concatenation is only supported along batch or n_prim dimension!"
        )

    return [
        UvGrid(
            coord=uvgrids.coord[i : i + stride] if uvgrids.coord is not None else None,
            grid_mask=(
                uvgrids.grid_mask[i : i + stride]
                if uvgrids.grid_mask is not None
                else None
            ),
            empty_mask=(
                uvgrids.empty_mask[i : i + stride]
                if uvgrids.empty_mask is not None
                else None
            ),
            normal=(
                uvgrids.normal[i : i + stride] if uvgrids.normal is not None else None
            ),
            prim_type=(
                uvgrids.prim_type[i : i + stride]
                if uvgrids.prim_type is not None
                else None
            ),
            face_adj=(
                uvgrids.face_adj[i : i + stride]
                if uvgrids.face_adj is not None
                else None
            ),
        )
        for i in range(0, len(uvgrids.coord), stride)
    ]


def stack_uvgrids(uvgrids: List[UvGrid]) -> UvGrid:
    """
    Given list of uvgrids, stack the parameters and create a single uvgrid
    :param uvgrids:
    :return:
    """
    # list of array -> tensor is slow and raises warning
    # disable warning by converting list of array -> array -> tensor
    if isinstance(uvgrids[0].coord, np.ndarray):
        coord = torch.tensor(np.array([x.coord for x in uvgrids]))
        grid_mask = (
            torch.tensor(np.array([x.grid_mask for x in uvgrids]))
            if uvgrids[0].grid_mask is not None
            else None
        )
        empty_mask = (
            torch.tensor(np.array([x.empty_mask for x in uvgrids]))
            if uvgrids[0].empty_mask is not None
            else None
        )
        normal = (
            torch.tensor(np.array([x.normal for x in uvgrids]))
            if uvgrids[0].normal is not None
            else None
        )
        prim_type = (
            torch.tensor(np.array([x.prim_type for x in uvgrids]))
            if uvgrids[0].prim_type is not None
            else None
        )
        face_adj = (
            torch.tensor(np.array([x.face_adj for x in uvgrids]))
            if uvgrids[0].face_adj is not None
            else None
        )
    else:
        coord = torch.stack([x.coord for x in uvgrids])
        grid_mask = (
            torch.stack([x.grid_mask for x in uvgrids])
            if uvgrids[0].grid_mask is not None
            else None
        )
        empty_mask = (
            torch.stack([x.empty_mask for x in uvgrids])
            if uvgrids[0].empty_mask is not None
            else None
        )
        normal = (
            torch.stack([x.normal for x in uvgrids])
            if uvgrids[0].normal is not None
            else None
        )
        prim_type = (
            torch.stack([x.prim_type for x in uvgrids])
            if uvgrids[0].prim_type is not None
            else None
        )
        face_adj = (
            torch.stack([x.face_adj for x in uvgrids])
            if uvgrids[0].face_adj is not None
            else None
        )

    ret = UvGrid(
        coord=coord,
        grid_mask=grid_mask,
        empty_mask=empty_mask,
        normal=normal,
        prim_type=prim_type,
        face_adj=face_adj,
    )
    return ret


def stack_batched_uvgrids(uvgrids: List[UvGrid]) -> UvGrid:
    """
    Given list of batched uvgrids, stack the parameters and create a single uvgrid
    :param uvgrids:
        with each uvgrid having coords of B x n_prims x n_grid x n_grid x 3
    :return:
        - uvgrid with B * len(uvgrids) x n_prims x n_grid x n_grid x 3
    """
    assert isinstance(uvgrids[0].coord, torch.Tensor)
    # list of array -> tensor is slow and raises warning
    # disable warning by converting list of array -> array -> tensor
    list2npy = lambda x: torch.vstack(x)
    coord = list2npy([x.coord for x in uvgrids])
    grid_mask = (
        list2npy([x.grid_mask for x in uvgrids])
        if uvgrids[0].grid_mask is not None
        else None
    )
    empty_mask = (
        list2npy([x.empty_mask for x in uvgrids])
        if uvgrids[0].empty_mask is not None
        else None
    )
    normal = (
        list2npy([x.normal for x in uvgrids]) if uvgrids[0].normal is not None else None
    )
    prim_type = (
        list2npy([x.prim_type for x in uvgrids])
        if uvgrids[0].prim_type is not None
        else None
    )
    face_adj = (
        (list2npy([x.face_adj for x in uvgrids]))
        if uvgrids[0].face_adj is not None
        else None
    )
    ret = UvGrid(
        coord=coord,
        grid_mask=grid_mask,
        empty_mask=empty_mask,
        normal=normal,
        prim_type=prim_type,
        face_adj=face_adj,
    )
    return ret


def uv_coord2normal(
    coord: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """

    :param coord: tensor or numpy of B x n_grid x n_grid x 3
    :return:
        normal: tensor or numpy of B x n_grid x n_grid x 3
    """
    b, n, _, _ = coord.shape

    is_np = False
    if isinstance(coord, np.ndarray):
        is_np = True
        coord = torch.from_numpy(coord)

    # Prepare tangent tensors
    tangent_u = torch.zeros_like(coord)
    tangent_v = torch.zeros_like(coord)

    # Compute tangents using central differences for interior
    tangent_u[:, 1:-1] = coord[:, 2:] - coord[:, :-2]
    tangent_v[:, :, 1:-1] = coord[:, :, 2:] - coord[:, :, :-2]

    # Borders (forward/backward differences)
    tangent_u[:, 0] = coord[:, 1] - coord[:, 0]
    tangent_u[:, -1] = coord[:, -1] - coord[:, -2]
    tangent_v[:, :, 0] = coord[:, :, 1] - coord[:, :, 0]
    tangent_v[:, :, -1] = coord[:, :, -1] - coord[:, :, -2]

    # Compute cross product manually for each batch
    normals_x = (
        tangent_u[..., 1] * tangent_v[..., 2] - tangent_u[..., 2] * tangent_v[..., 1]
    )
    normals_y = (
        tangent_u[..., 2] * tangent_v[..., 0] - tangent_u[..., 0] * tangent_v[..., 2]
    )
    normals_z = (
        tangent_u[..., 0] * tangent_v[..., 1] - tangent_u[..., 1] * tangent_v[..., 0]
    )

    # Stack the components to form normals
    normals = torch.stack((normals_x, normals_y, normals_z), dim=-1)

    # Normalize the normals
    norm = torch.norm(normals, dim=-1, keepdim=True)
    normalized_normals = normals / torch.clamp(norm, min=1e-8)  # Avoid division by zero
    if is_np:
        normalized_normals = normalized_normals.numpy(force=True)
    return normalized_normals
