from torch import nn as nn
from abc import ABC, abstractmethod

from brepdgt.utils.acc_logger import AccLogger
from brepdgt.config import Config

# ==========
# Model ABCs
# ==========


class BaseModule(nn.Module, ABC):
    def __init__(self, config: Config, acc_logger: AccLogger):
        super().__init__()
        self.config = config
        self.acc_logger = acc_logger


class BaseModel(BaseModule):
    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError()
