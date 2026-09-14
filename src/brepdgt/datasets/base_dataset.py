from torch.utils.data import Dataset, DataLoader
from abc import ABC
from typing import List, Dict

from brepdgt.config import Config


class BaseDataset(Dataset, ABC):
    name = "base"

    def __init__(self, config: Config, split: str = "train"):
        self.config: Config = config
        self.split = split
        self.dim = self.get_dim()
        self.data_list = []

    def __getitem__(self, idx) -> Dict:
        # sparse tensor and tensor should have equal size
        raise NotImplemented

    def collate_fn(self, batch: List) -> dict:
        # convert list of dict to dict of list
        batch = {k: [d[k] for d in batch] for k in batch[0]}
        return batch

    def __len__(self):
        return len(self.data_list)

    def get_dim(self) -> int:
        """
        :return: dimension of the dataset
        """
        raise NotImplementedError()

    def get_data_list(self) -> List:
        """
        :return: data list of current data
        """
        raise NotImplementedError()

    def get_data_name(self, idx) -> str:
        """
        :param idx: idx of datalist
        :return: data name of the index
        """
        raise NotImplementedError()
