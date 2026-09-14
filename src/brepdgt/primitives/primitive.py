from __future__ import annotations
from abc import ABC
from typing import Tuple
import numpy as np


class Primitive(ABC):
    name = ""

    def __init__(self, *args, **kwargs):
        pass

    def get_grid_coord(self, n_grid):
        raise NotImplementedError()

    def get_normal(self, pts):
        raise NotImplementedError()

    def get_slicer(self):
        raise NotImplementedError()

    @classmethod
    def get_slicer_type(cls):
        raise NotImplementedError()

    @classmethod
    def is_slicer(cls) -> bool:
        raise NotImplementedError()

    @classmethod
    def n_param(cls) -> int:
        raise NotImplementedError()

    def get_param(self) -> np.ndarray:
        raise NotImplementedError()

    @classmethod
    def get_param_range(cls) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    @classmethod
    def get_activations(cls):
        raise NotImplementedError()

    def to_arr(self) -> np.array:
        raise NotImplementedError()

    @classmethod
    def from_arr(cls, arr):
        raise NotImplementedError()

    def apply_transformation(self, transformation):
        raise NotImplementedError()

    @classmethod
    def intersect(
        cls, other: Primitive, epsilon_threshold: float = 0.0
    ) -> tuple[np.ndarray]:
        raise NotImplementedError()
