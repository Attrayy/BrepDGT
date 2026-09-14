# This is adapted from https://github.com/lucidrains/denoising-diffusion-pytorch

import math
from dataclasses import dataclass

import torch

# constants


@dataclass(frozen=True)
class ModelPrediction:
    pred_noise: torch.Tensor
    pred_x_start: torch.Tensor


# helpers functions


def identity(t, *args, **kwargs):
    return t


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps, beta0, betaT):
    scale = 1000 / timesteps
    beta_start = scale * beta0
    beta_end = scale * betaT
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s: float = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sqrt_beta_schedule(timesteps, s: float = 0.0001):
    """
    sqrt schedule
    as proposed in https://arxiv.org/abs/2205.14217
    """
    steps = timesteps
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    return 1.0 - torch.sqrt(x / timesteps + s)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def snr_based_beta_schedule(
    timesteps: int,
    snr_max: float = 1000,
    snr_min: float = 0.01,
    snr_power: float = 1.0,
):
    """Create beta schedule that targets specific SNR values"""
    # Create normalized time steps and apply power transformation
    t = torch.linspace(0, 1, timesteps)
    t_transformed = torch.pow(t, snr_power)

    # Create log-SNR schedule with transformed time
    log_snr_max = math.log(snr_max)
    log_snr_min = math.log(snr_min)
    log_snr = log_snr_max + (log_snr_min - log_snr_max) * t_transformed

    # Convert to SNR values
    target_snr = torch.exp(log_snr)

    # SNR = alphas_cumprod / (1 - alphas_cumprod)
    # Solve for alphas_cumprod
    alphas_cumprod = target_snr / (1 + target_snr)

    # Back out betas
    alphas = torch.zeros_like(alphas_cumprod)
    alphas[0] = alphas_cumprod[0]
    for i in range(1, len(alphas)):
        alphas[i] = alphas_cumprod[i] / alphas_cumprod[i - 1]

    betas = 1 - alphas
    return betas.clamp(0, 0.999)


# normalization functions


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5
