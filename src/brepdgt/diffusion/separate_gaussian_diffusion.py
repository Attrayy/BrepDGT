from brepdgt.diffusion.gaussian_diffusion import GaussianDiffusion1D
from brepdgt.diffusion.base import Diffusion
from typing import List, Union

from copy import deepcopy
import torch
import torch.nn as nn
from brepdgt.config import Config
from brepdgt.models.backbones.dit_1d import Dit1D


class SeparateGaussianDiffusion1D(GaussianDiffusion1D):
    name: str = "separate_gaussian_diffusion_1d"

    def __init__(
        self,
        model: Dit1D,
        config: Config,
    ):
        super().__init__(model, config)

        # Get masks for each dim in token
        # assert (
        #     self.model.input_dim == 256
        # ), f"Currently hard coded for augmented uvgrid with dim 256, got {self.model.input_dim}"
        # Precompute where coordinates lies in x
        x_dim_arange = torch.arange(self.model.input_dim)
        per_points_dim = 4  # coord (3) + grid (1)
        self.x_coord_mask = (
            (x_dim_arange % per_points_dim == 0)
            | (x_dim_arange % per_points_dim == 1)
            | (x_dim_arange % per_points_dim == 2)
        )
        self.x_grid_mask = x_dim_arange % per_points_dim == 3
        self.x_masks = [self.x_coord_mask, self.x_grid_mask]

        # Define noise schedules for each attributes
        # This is dangerous since diffusion configs are different from other configs.
        # But without it, we need major refactoring
        # coord diffusion
        self.diffusions: List[GaussianDiffusion1D] = nn.ModuleList()  # Cheat on typing!
        for feat in ["coord", "grid_mask"]:
            diffusion_config = deepcopy(self.config)
            diffusion_config.diffusion.noise_schedule = (
                self.config.diffusion.__getattribute__(f"{feat}_noise_schedule")
            )
            diffusion_config.diffusion.training_timesteps = (
                self.config.diffusion.__getattribute__(f"{feat}_timesteps")
            )
            diffusion_config.diffusion.inference_timesteps = (
                self.config.diffusion.__getattribute__(f"{feat}_timesteps")
            )
            diffusion = GaussianDiffusion1D(model, diffusion_config)
            self.diffusions.append(diffusion)

    def clip_t(self, t: torch.Tensor, diffusion: Diffusion):
        t_out_mask = t >= diffusion.n_training_timesteps
        # set t to maximum allowed value
        t[t_out_mask] = diffusion.n_training_timesteps - 1
        return t

    def call_and_aggregate(
        self,
        func_name: str,
        x1: torch.Tensor,
        t: torch.Tensor,
        x2: torch.Tensor,
        is_q_sample: bool = False,
    ) -> torch.Tensor:
        """
        Calls diffusion function with func_name independently and aggregates results.
        The func_name should take x1, t, x2 as an input, where x1 and x2 have the same dimension.
        """
        out = torch.zeros_like(x1)
        for i in range(len(self.diffusions)):
            t_i = self.clip_t(t.clone(), diffusion=self.diffusions[i])
            diff_out = getattr(self.diffusions[i], func_name)(x1, t_i, x2)
            out[:, self.x_masks[i]] = diff_out[:, self.x_masks[i]]
        return out

    def undo(self, x: torch.Tensor, t: Union[torch.Tensor, int]):
        return self.call_and_aggregate_single("undo", x, t)

    def q_sample(self, x_start, t, noise=None):
        return self.call_and_aggregate("q_sample", x_start, t, noise, is_q_sample=True)

    # TODO: clean that

    def predict_start_from_noise(self, x_t, t, noise):
        return self.call_and_aggregate("predict_start_from_noise", x_t, t, noise)

    def predict_noise_from_start(self, x_t, t, x0):
        return self.call_and_aggregate("predict_noise_from_start", x_t, t, x0)

    def predict_v(self, x_start, t, noise):
        return self.call_and_aggregate("predict_v", x_start, t, noise)

    def predict_start_from_v(self, x_t, t, v):
        return self.call_and_aggregate("predict_start_from_v", x_t, t, v)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean, posterior_variance, posterior_log_variance_clipped = (
            torch.zeros_like(x_start),
            torch.zeros_like(x_start),
            torch.zeros_like(x_start),
        )
        for i in range(len(self.diffusions)):
            t_i = self.clip_t(t.clone(), diffusion=self.diffusions[i])
            (
                posterior_mean_i,
                posterior_variance_i,
                posterior_log_variance_clipped_i,
            ) = self.diffusions[i].q_posterior(x_start, x_t, t_i)
            posterior_mean[:, self.x_masks[i]] = posterior_mean_i[:, self.x_masks[i]]
            posterior_variance[:, self.x_masks[i]] = posterior_variance_i[
                :, self.x_masks[i]
            ]
            posterior_log_variance_clipped[
                :, self.x_masks[i]
            ] = posterior_log_variance_clipped_i[:, self.x_masks[i]]
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def call_and_aggregate_steps(self, func_name, x, t, **kwargs):
        """Same as below but for step sampling functions!"""
        pred_img, x_start = torch.zeros_like(x), torch.zeros_like(x)
        for i in range(len(self.diffusions)):
            # WARNING: this is hardcoded for our case!
            assert i < 2
            # Skip and return same value! i.e., identity
            if t >= self.diffusions[i].n_training_timesteps:
                pred_img[:, self.x_masks[i]] = x[:, self.x_masks[i]]
            else:
                diff_pred_img, diff_x_start = getattr(self.diffusions[i], func_name)(
                    x=x,  # Give full x as it will masked automatically by the function below
                    t=t,
                    **kwargs,
                )
                pred_img[:, self.x_masks[i]] = diff_pred_img[:, self.x_masks[i]]
                x_start[:, self.x_masks[i]] = diff_x_start[:, self.x_masks[i]]
        return pred_img, x_start

    def call_and_aggregate_single(self, func_name, x, t, **kwargs):
        """Same as above but for only one return value!"""
        out_x = torch.zeros_like(x)
        for i in range(len(self.diffusions)):
            # WARNING: this is hardcoded for our case!
            assert i < 2
            # Skip and return same value! i.e., identity
            if t >= self.diffusions[i].n_training_timesteps:
                out_x[:, self.x_masks[i]] = x[:, self.x_masks[i]]
            else:
                diff_pred_img = getattr(self.diffusions[i], func_name)(
                    x=x,  # Give full x as it will masked automatically by the function below
                    t=t,
                    **kwargs,
                )
                out_x[:, self.x_masks[i]] = diff_pred_img[:, self.x_masks[i]]
        return out_x
