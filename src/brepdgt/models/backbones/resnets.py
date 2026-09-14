# Adapted from:
#
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
#

import torch
from torch import nn
from torchvision import models


class BaseFeatureExtractor(nn.Module):
    """Hold some common functions among all feature extractor networks."""

    @property
    def feature_size(self):
        return self._feature_size

    def forward(self, X):
        return self._feature_extractor(X)


class ResNet(BaseFeatureExtractor):
    """
    Build a feature extractor using the pretrained ResNet architecture specified by `name` for
    image based inputs.
    """

    def __init__(self, input_channels: int, feature_size: int, name: str):
        super(ResNet, self).__init__()
        self._feature_size = feature_size

        self._feature_extractor = models.get_model(name=name, pretrained=False)

        # Override input and output to output single dimensional latent
        self._feature_extractor.conv1 = torch.nn.Conv2d(
            input_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )

        self._feature_extractor.fc = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, self.feature_size)
        )
        self._feature_extractor.avgpool = nn.AdaptiveAvgPool2d((1, 1))


class ResNetEncoder(nn.Module):
    def __init__(
        self, in_channels: int = 1, output_dims: int = 64, name: str = "resnet50"
    ) -> None:
        super().__init__()

        self.resnet = ResNet(
            input_channels=in_channels, feature_size=output_dims, name=name
        )

    def forward(self, x):
        return self.resnet(x)
