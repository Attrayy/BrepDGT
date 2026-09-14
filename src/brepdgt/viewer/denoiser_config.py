from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from enum import Enum
from dataclass_wizard import YAMLWizard

from brepdgt.viewer.interpolation import InterpolationConfig


class SchedulerType(Enum):
    DDPM: str = "ddpm"
    DDIM: str = "ddim"


# ===============================
# POSSIBLE VALUES (for sliders)
# ===============================


SCHEDULER_TYPE_MAP = {x: i for i, x in enumerate(SchedulerType)}
SCHEDULER_TYPE_INVMAP = {i: x for i, x in enumerate(SchedulerType)}

ALLOWED_INFERENCE_STEPS = [50, 100, 250, 500, 1000]
ALLOWED_INFERENCE_STEPS_MAP = {x: i for i, x in enumerate(ALLOWED_INFERENCE_STEPS)}
ALLOWED_INFERENCE_STEPS_INVMAP = {i: x for i, x in enumerate(ALLOWED_INFERENCE_STEPS)}

ALLOWED_BATCH_SIZES = [2**i for i in range(11)]
ALLOWED_BATCH_SIZES_MAP = defaultdict(
    lambda: 0, {x: i for i, x in enumerate(ALLOWED_BATCH_SIZES)}
)
ALLOWED_BATCH_SIZES_INVMAP = {i: x for i, x in enumerate(ALLOWED_BATCH_SIZES)}

ALLOWED_EXPORT_STRIDES = [1, 10, 50, 100]
ALLOWED_EXPORT_STRIDES_MAP = {x: i for i, x in enumerate(ALLOWED_EXPORT_STRIDES)}
ALLOWED_EXPORT_STRIDES_INVMAP = {i: x for i, x in enumerate(ALLOWED_EXPORT_STRIDES)}


# ===============================
# CONFIGS
# ===============================


@dataclass
class InpaintingConfig:
    n_inpainting_steps: int = 1000
    # Overrides batch size with different empty mask sizes
    override_batch_size: bool = False
    empty_min: int = 4
    empty_max: int = 30
    # sub-batch size requested for each empty mask
    empty_batch_multiplier: int = 4


@dataclass
class ResamplingConfig:
    enabled: bool = False
    jump_length: int = 10
    # Ideally, one should be enough!
    n_repetitions: int = 1
    start_t: int = 900


@dataclass
class DenoiserConfig(YAMLWizard, key_transform="SNAKE"):
    n_inference_steps: int = 1000
    n_training_steps: int = 1000

    seed: int = 42
    n_prims: int = 4
    batch_size: int = 4
    cfg_scale: float = 1.0
    renoise_steps: int = 999
    noise_t_threshold: int = 0
    scheduler_type: SchedulerType = SchedulerType.DDPM

    fast_mode: bool = False

    # Export stride
    export_stride: int = 50

    # Inpainting
    inpainting: InpaintingConfig = field(default_factory=lambda: InpaintingConfig())

    # Resampling
    resampling: ResamplingConfig = field(default_factory=lambda: ResamplingConfig())

    # Interpolation
    interpolation: InterpolationConfig = field(
        default_factory=lambda: InterpolationConfig()
    )

    def __post_init__(self):
        # assert self.batch_size == 1
        pass
