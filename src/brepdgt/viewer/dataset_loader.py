from __future__ import annotations
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import os
from enum import Enum
from copy import copy, deepcopy
import clip

import torch
import numpy as np
import trimesh
import faiss
from tqdm import tqdm

import polyscope as ps
import polyscope.imgui as psim

from brepdgt.config import Config
from brepdgt.models.brepdgt import BRepDGT
from brepdgt.datasets import DATASETS
from brepdgt.datasets.abc_dataset import DeepCad30Dataset, ABCDatasetOutput
from brepdgt.primitives.uvgrid import stack_uvgrids, UvGrid

TMP_FOLDER = "tmp/"
CLIP_EMBEDDINGS_FOLDER = "data/clip"

ALLOWED_SPLITS = ["train", "test", "val"]
ALLOWED_SPLITS_MAP = {x: i for i, x in enumerate(ALLOWED_SPLITS)}
ALLOWED_SPLITS_INVMAP = {i: x for i, x in enumerate(ALLOWED_SPLITS)}


class DataLoadMode(Enum):
    GLOBAL = "global"
    KNN = "knn"
    CLIP = "clip"


DATALOAD_MODE_KEYS = [x.value for x in DataLoadMode]
DATALOAD_MODE_MAP = {x: i for i, x in enumerate(DataLoadMode)}
DATALOAD_MODE_INVMAP = {i: x for i, x in enumerate(DataLoadMode)}


@dataclass
class DatasetLoaderConfig:
    split: str = "test"
    k: int = 1000


class DatasetLoader:
    def __init__(
        self,
        zooc_config: Config,
        model: BRepDGT,
        config: DatasetLoaderConfig = DatasetLoaderConfig(),
    ):
        self.config = config
        self.model = model

        self.clip_model = None
        self.clip_prompt: str = "star"

        # Make sure to disable random augmentations!
        self.custom_zooc_config = deepcopy(zooc_config)
        self.custom_zooc_config.random_augmentations = "none"

        # Reset dataset
        self._reset_dataset()

    def _reset_dataset(self):
        print("loading split", self.config.split)
        self.dataset: DeepCad30Dataset = DATASETS[self.custom_zooc_config.dataset](
            self.custom_zooc_config, split=self.config.split
        )
        self.data_load_mode = DataLoadMode.GLOBAL
        self.current_idx: int = 0
        self.knn_list = []
        self.clip_list = []

    def get_current(self):
        if self.data_load_mode == DataLoadMode.GLOBAL:
            return stack_uvgrids([self.dataset[self.current_idx].uvgrid])
        elif self.data_load_mode == DataLoadMode.KNN:
            return stack_uvgrids([self.dataset[self.knn_list[self.current_idx]].uvgrid])
        elif self.data_load_mode == DataLoadMode.CLIP:
            return (
                stack_uvgrids([self.dataset[self.clip_list[self.current_idx]].uvgrid])
                if self.current_idx < len(self.clip_list)
                else stack_uvgrids([self.dataset[self.current_idx].uvgrid])
            )
        else:
            raise ValueError()

    # ==========================
    # KNN
    # ==========================

    def find_closest(self):
        uv_grid = self.get_current()
        target_n_prims = self._get_n_prims(uv_grid)
        self.create_index(target_n_prims)
        self.query_index(uv_grid, k=self.config.k)

    def _get_n_prims(self, uv_grid: UvGrid):
        return (
            torch.count_nonzero(~uv_grid.empty_mask)
            if isinstance(uv_grid, torch.Tensor)
            else np.count_nonzero(~uv_grid.empty_mask)
        )

    def __get_index_paths(self, target_n_prims: int, max_n: int = 100000):
        target_folder = os.path.join(TMP_FOLDER, self.model.config.dataset)
        os.makedirs(target_folder, exist_ok=True)
        return os.path.join(
            target_folder, f"{self.config.split}_{target_n_prims}_{max_n}.index"
        ), os.path.join(
            target_folder, f"{self.config.split}_{target_n_prims}_{max_n}.npy"
        )

    def create_index(self, target_n_prims: int, max_n: int = 100000):
        index_path, i_to_g_path = self.__get_index_paths(target_n_prims, max_n)
        if os.path.exists(index_path) and os.path.exists(i_to_g_path):
            self.index = faiss.read_index(index_path)
            self.index_to_global = np.load(i_to_g_path)
        else:
            self.index_to_global = []
            self.index = None
            for i_data, data in tqdm(enumerate(self.dataset)):
                data: ABCDatasetOutput = data
                if i_data >= max_n:
                    break

                n_prims = self._get_n_prims(data.uvgrid)
                if n_prims != target_n_prims:
                    continue

                feats = data.uvgrid.coord[~data.uvgrid.empty_mask].flatten()[None, :]

                if self.index is None:
                    self.index = faiss.IndexFlatL2(feats.shape[-1])

                self.index.add(feats)
                self.index_to_global.append(i_data)

            faiss.write_index(self.index, index_path)
            np.save(i_to_g_path, np.array(self.index_to_global))

        self.data_load_mode = DataLoadMode.KNN
        self.current_idx = 0

    def query_index(self, uv_grid: UvGrid, k: int) -> None:
        feats = uv_grid.coord[~uv_grid.empty_mask].flatten()[None, :]
        if isinstance(feats, torch.Tensor):
            feats = feats.cpu().numpy()
        _, indices = self.index.search(feats, k)
        indices = indices[0]
        global_indices = [self.index_to_global[i] for i in indices]
        self.knn_list = global_indices

        # print(f"Created index with {}")

    # ==========================
    # KNN
    # ==========================

    def _load_clip(self):
        if self.clip_model is not None:
            return
        self.clip_model, _ = clip.load("ViT-B/32", device="cuda")

    def _get_clip_index_path(self):
        target_folder = os.path.join(TMP_FOLDER, self.model.config.dataset)
        os.makedirs(target_folder, exist_ok=True)
        return os.path.join(
            target_folder, f"clip_{self.config.split}.index"
        ), os.path.join(target_folder, f"clip_{self.config.split}.npy")

    def index_of_a_in_b(self, a, b):
        b_indices = np.where(np.isin(b, a))[0]
        b_values = b[b_indices]
        return b_indices[b_values.argsort()[a.argsort().argsort()]]

    @torch.no_grad()
    def _create_clip_index(self):
        clip_embed_path = os.path.join(
            CLIP_EMBEDDINGS_FOLDER, self.config.split + ".npz"
        )
        if not os.path.exists(clip_embed_path):
            print(
                f"Couldn't find CLIP embeddings for split '{self.config.split}' at: {clip_embed_path}"
            )
            return
        data = np.load(clip_embed_path)
        embeds = data["clip_emb"].astype(np.float32)
        uid = data["uid"].astype(np.int64)

        # First, we need to query all the data_uids in the dataset to map them to the ones in the embeddings!
        all_data_uids = []
        for entry in tqdm(self.dataset):
            all_data_uids.append(int(entry.name))
        all_data_uids = np.array(all_data_uids).astype(np.uint64)

        # Filter with what is in the data uids
        inside_mask = np.isin(uid, all_data_uids)
        embeds = embeds[inside_mask]
        uid = uid[inside_mask]
        # Then, create the map from the filtered embeddings to global
        self.clip_index_to_global = self.index_of_a_in_b(
            uid,
            all_data_uids,
        )
        # Then, create the index
        self.clip_index = faiss.IndexFlatIP(embeds.shape[-1])
        self.clip_index.add(embeds)

        index_path, i_to_g_path = self._get_clip_index_path()

        faiss.write_index(self.clip_index, index_path)
        np.save(i_to_g_path, np.array(self.clip_index_to_global))

    @torch.no_grad()
    def _load_clip_index(self):
        # First try to load the index
        clip_index_path, clip_i_to_g_path = self._get_clip_index_path()
        if os.path.exists(clip_index_path) and os.path.exists(clip_i_to_g_path):
            self.clip_index = faiss.read_index(clip_index_path)
            self.clip_index_to_global = np.load(clip_i_to_g_path)
        else:
            self._create_clip_index()

    @torch.no_grad()
    def query_clip(self, k: int):
        tokens = clip.tokenize(self.clip_prompt)
        text_embeds = self.clip_model.encode_text(tokens.cuda())
        _, indices = self.clip_index.search(
            text_embeds.cpu().numpy().astype(np.float32), k
        )
        indices = indices[0]
        global_indices = [self.clip_index_to_global[i] for i in indices]
        self.clip_list = global_indices

    @torch.no_grad()
    def query_clip_by_embed(self, embed: np.ndarray, k: int):
        if len(embed.shape) == 1:
            embed = embed[None, :]
        _, indices = self.clip_index.search(embed.astype(np.float32), k)
        indices = indices[0]
        global_indices = [self.clip_index_to_global[i] for i in indices]
        self.clip_list = global_indices

    @torch.no_grad()
    def gui(self):
        update = False

        # Initialize new_mode with current mode
        new_mode = self.data_load_mode

        # ===============================
        # CHOOSE MODE
        # ===============================

        clicked, dataload_mode_idx = psim.Combo(
            "mode##dataset_loader",
            DATALOAD_MODE_MAP[self.data_load_mode],
            DATALOAD_MODE_KEYS,
        )
        update |= clicked
        if clicked:
            new_mode = DATALOAD_MODE_INVMAP[dataload_mode_idx]
            if new_mode == DataLoadMode.KNN and len(self.knn_list) == 0:
                pass
            else:
                if new_mode == DataLoadMode.CLIP:
                    self._load_clip()
                    self._load_clip_index()

                # Make sure to reset indices
                self.current_idx = 0
                self.data_load_mode = new_mode

        # ===============================
        # CHOOSE SPLIT
        # ===============================

        clicked, split_idx = psim.Combo(
            "split##dataset_loader",
            ALLOWED_SPLITS_MAP[self.config.split],
            ALLOWED_SPLITS,
        )
        if clicked:
            self.config.split = ALLOWED_SPLITS_INVMAP[split_idx]
            self._reset_dataset()

            # Reload indices here!
            if new_mode == DataLoadMode.CLIP:
                self._load_clip_index()

            update |= True

        # ===============================
        # MODE INFO
        # ===============================

        if self.data_load_mode == DataLoadMode.GLOBAL:
            psim.Text(f"GLOBAL MODE (dataset size: {len(self.dataset)})")
            # Display current uid
            current_uid = self.dataset.data_list[self.current_idx]
            psim.Text(f"Current uid: {current_uid}")

            clicked, self.current_idx = psim.InputInt(
                "idx##dataset_loader", self.current_idx, step=1
            )
            self.current_idx = max(min(self.current_idx, len(self.dataset) - 1), 0)

            if psim.Button("Find closest##dataset_loader"):
                self.find_closest()
                update |= True

        elif self.data_load_mode == DataLoadMode.KNN:
            psim.Text("KNN (go to 0 to see reference shape!):")
            # Display current uid
            current_uid = self.dataset.data_list[self.knn_list[self.current_idx]]
            psim.Text(f"Current uid: {current_uid}")

            clicked, self.current_idx = psim.InputInt(
                "idx##dataset_loader", self.current_idx, step=1
            )
            self.current_idx = max(min(self.current_idx, len(self.knn_list) - 1), 0)
        elif self.data_load_mode == DataLoadMode.CLIP:
            _, self.clip_prompt = psim.InputText(
                "Prompt##dataset_loader", self.clip_prompt
            )
            if psim.Button("Search prompt##dataset_loader"):
                self.query_clip(self.config.k)
                update |= True

            clicked, self.current_idx = psim.InputInt(
                "idx##dataset_loader", self.current_idx, step=1
            )
            self.current_idx = max(min(self.current_idx, len(self.clip_list) - 1), 0)
        else:
            raise ValueError()
        update |= clicked

        # clicked, self.config.k = psim.SliderInt(
        #     "k##dataset_loader", self.config.k, v_min=1, v_max=1000
        # )

        return update
