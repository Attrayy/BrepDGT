from collections import defaultdict

import torch
import os
import numpy as np
from tqdm import tqdm
from typing import List, Union
from dataclasses import dataclass
import h5py

from brepdgt.config import Config
from brepdgt.utils.common import (
    load_pkl,
    collate_dataclass_fn,
)
from brepdgt.primitives.uvgrid import UvGrid, stack_uvgrids
from brepdgt.datasets.base_dataset import BaseDataset
from brepdgt.utils.transform_3d import (
    apply_affine_transform,
    Rot90Flip,
    ScaleTranslate,
)


@dataclass
class ABCDatasetOutput:
    name: Union[str, List[str]]
    uvgrid: Union[UvGrid, List[UvGrid]]


class ABCDataset(BaseDataset):
    name = "abc"

    def __init__(self, config: Config, split="train"):
        overfit_val = False
        if config.overfit and split != "train":
            split = "train"
            overfit_val = True

        # for testing, we use training set to sample number of faces distribution
        train_for_test = False
        if split == "train_for_test":
            split = "train"
            train_for_test = True

        super().__init__(config, split)

        self.h5 = h5py.File(self.config.h5_path, "r")
        self.data_list_path = self.get_data_list_path()
        self.invalid_step_list_path = self.get_invalid_step_list_path()

        self.aa_rot_flip = Rot90Flip()
        self.scale_translate = ScaleTranslate(
            scale_min=config.scale_min,
            scale_max=config.scale_max,
            translate_min=config.translate_min,
            translate_max=config.translate_max,
        )

        print("Getting Data List...")
        self.data_list = self.get_data_list()

        # Only use partial subset of dataset when testing for speed
        if (self.split == "val") and self.config.test_in_loop:
            print(f"Using {self.config.val_num_sample} samples for {split}")
            self.data_list = self.data_list[: self.config.val_num_sample]

        if train_for_test:
            print(
                f"Using {self.config.test_num_sample} samples for training dataset for testing"
            )
            self.data_list = self.data_list[: self.config.test_num_sample]
            if self.config.test_chunk != "":
                chunk_idx, total_chunk = self.config.test_chunk.split("/")
                chunk_idx, total_chunk = int(chunk_idx) - 1, int(total_chunk)
                data_list = np.array_split(self.data_list, total_chunk)[chunk_idx]
                print(f"Using chunk of {self.config.test_chunk} for dataset")
                print(f"dataset reduced to {len(data_list)}/{len(self.data_list)}")
                self.data_list = list(data_list)

        if overfit_val:
            self.data_list = self.data_list[: self.config.val_num_sample]
        print(f"Using {len(self.data_list)} {self.split} data in {self.config.h5_path}")
        print(f"Filtered with {self.config.max_n_prims} prims")
        print(f"Augmentation mode: {self.config.random_augmentations}")

    def __getitem__(self, idx) -> ABCDatasetOutput:
        uid = self.data_list[idx]
        data = self.h5["data"][uid]

        # Copy data from h5
        label = data["types"][:]
        # We don't care about labels :)
        label = label + 1  # let Empty be 0

        coord = data["coords"][:]  # n_prims x n_grid x n_grid x 3
        normal = data["normals"][:]  # n_prims x n_grid x n_grid x 3
        grid_mask = data["masks"][:]  # n_prims x n_grid x n_grid

        # Apply augmentations during training
        if self.split == "train":
            if self.config.random_augmentations == "scale_translate":
                tx = self.scale_translate.get_rand_tx()
                coord = (
                    apply_affine_transform(coord.reshape(-1, 3), tx)
                    .reshape(coord.shape)
                    .astype(np.float32)
                )
            elif self.config.random_augmentations == "scale_translate_rotation":
                tx = self.aa_rot_flip.get_rand_tx(
                    enable_rotation=True,
                    enable_flip=False,
                )
                coord = (
                    apply_affine_transform(coord.reshape(-1, 3), tx)
                    .reshape(coord.shape)
                    .astype(np.float32)
                )
                normal = (
                    apply_affine_transform(normal.reshape(-1, 3), tx)
                    .reshape(normal.shape)
                    .astype(np.float32)
                )
                tx = self.scale_translate.get_rand_tx()
                coord = (
                    apply_affine_transform(coord.reshape(-1, 3), tx)
                    .reshape(coord.shape)
                    .astype(np.float32)
                )

        uvgrid = UvGrid.from_raw_values(
            coord=coord,
            grid_mask=grid_mask,
            max_n_prims=self.config.max_n_prims,
            normal=normal,
            prim_type=label,
        )

        uid = uid.decode("utf-8")
        ret = ABCDatasetOutput(name=uid, uvgrid=uvgrid)
        return ret

    def collate_fn(self, batch: List[ABCDatasetOutput]) -> ABCDatasetOutput:
        # convert list of dataclasses to dataclasses of list
        batch: ABCDatasetOutput = collate_dataclass_fn(batch)
        if not batch.uvgrid[0]:
            batch.uvgrid = None
        if batch.uvgrid is not None:
            batch.uvgrid = stack_uvgrids(batch.uvgrid)
        return batch

    def get_dim(self) -> int:
        return 3

    def get_data_list_path(self) -> str:
        data_list_path = os.path.join(
            os.path.dirname(self.config.h5_path), f"abc_50_{self.split}.txt"
        )
        return data_list_path

    def get_invalid_step_list_path(self) -> str:
        invalid_path = os.path.join(
            os.path.dirname(self.config.h5_path), f"abc_50_pkl_absence.txt"
        )
        return invalid_path

    def get_data_list(self) -> List:
        with open(self.data_list_path, "r") as f:
            data_list = f.readlines()
        data_list = [x.replace("\n", "").encode("utf-8") for x in data_list]

        with open(self.invalid_step_list_path, "r") as f:
            invalid_step_list = f.readlines()
        invalid_step_list = [
            x.replace("\n", "").encode("utf-8") for x in invalid_step_list
        ]
        # pkls that have been processed wrong
        data_list = sorted(list(set(data_list) - set(invalid_step_list)))

        data_list_new, data_list_removed = [], []
        for uid in tqdm(data_list):
            if self.h5["data"][uid]["types"].shape[0] > self.config.max_n_prims:
                continue
            data_list_new.append(uid)
            if self.config.overfit:
                print(f"Overfit on data id: {uid}")
                if len(data_list_new) >= self.config.overfit_data_size:
                    break
            if self.config.debug:
                if len(data_list_new) == self.config.debug_data_size:
                    print(f"Debug with {self.config.debug_data_size} data")
                    break
        if len(data_list_removed) != 0:
            print(f"Warning!! {len(data_list_removed)} was removed")

        if self.config.overfit:
            data_list_new = data_list_new * self.config.overfit_data_repetition
        data_list = data_list_new
        return data_list


class ABC50(ABCDataset):
    name = "abc_50"
    """
    ABC with 50 faces.
    Use this for comparing with baselines, cause this datalist will be fixed.
    """

    def __init__(self, config: Config, split: str):
        super().__init__(config, split)
        assert (
            self.config.max_n_prims == 50
        ), f"max_n_prims should be 50, got {self.config.max_n_prims}"

    def get_data_list(self) -> List:
        with open(self.data_list_path, "r") as f:
            data_list = f.readlines()
        data_list = [x.replace("\n", "").encode("utf-8") for x in data_list]

        with open(self.invalid_step_list_path, "r") as f:
            invalid_step_list = f.readlines()
        invalid_step_list = [
            x.replace("\n", "").encode("utf-8") for x in invalid_step_list
        ]
        data_list = sorted(list(set(data_list) - set(invalid_step_list)))
        if self.config.debug:
            data_list = data_list[: self.config.debug_data_size]
        return data_list


class DeepCadDataset(ABCDataset):
    name = "deepcad"

    def get_data_list_path(self) -> str:
        data_list_path = os.path.join(
            os.path.dirname(self.config.h5_path), f"deepcad_30_{self.split}.txt"
        )
        return data_list_path

    def get_invalid_step_list_path(self) -> str:
        invalid_path = os.path.join(
            os.path.dirname(self.config.h5_path), f"deepcad_30_pkl_absence.txt"
        )
        return invalid_path


class DeepCad30Dataset(DeepCadDataset):
    name = "deepcad_30"
    """
    DeepCAD with 30 faces.
    Use this for comparing with baselines, cause this datalist will be fixed.
    """

    def __init__(self, config: Config, split: str):
        super().__init__(config, split)
        assert (
            self.config.max_n_prims == 30
        ), f"max_n_prims should be 30, got {self.config.max_n_prims}"

    def get_data_list(self) -> List:
        with open(self.data_list_path, "r") as f:
            data_list = f.readlines()
        data_list = [x.replace("\n", "").encode("utf-8") for x in data_list]

        with open(self.invalid_step_list_path, "r") as f:
            invalid_step_list = f.readlines()
        invalid_step_list = [
            x.replace("\n", "").encode("utf-8") for x in invalid_step_list
        ]
        # pkls that have been wrongly processed
        data_list = sorted(
            list(set(data_list) - set(invalid_step_list))
        )  # there should be no interseciton btw both invalid list, but I'm keeping it to play safe
        if self.config.debug:
            data_list = data_list[: self.config.debug_data_size]
        if self.config.overfit:
            data_list = data_list[: self.config.overfit_data_size]
            data_list = self.config.overfit_data_repetition * data_list
        return data_list
