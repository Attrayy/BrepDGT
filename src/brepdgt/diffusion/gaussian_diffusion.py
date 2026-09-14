# This is adapted from https://github.com/lucidrains/denoising-diffusion-pytorch
from dataclasses import dataclass
from typing import Tuple, Dict, List, Union
from functools import partial
from tqdm import tqdm
import torch
from torch import nn
from random import random
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np

from brepdgt.config import Config
from brepdgt.models.tokenizers import Tokens
from brepdgt.diffusion.utils import (
    extract,
    linear_beta_schedule,
    cosine_beta_schedule,
    sqrt_beta_schedule,
    sigmoid_beta_schedule,
    snr_based_beta_schedule,
    default,
    ModelPrediction,
    identity,
)
from brepdgt.models.backbones.dit_1d import Dit1D
from brepdgt.diffusion.base import Diffusion, DiffusionSample

from einops import reduce


def get_timesteps(
    n_training_steps: int, n_inference_steps: int, steps_offset: int = 0
) -> List[int]:
    step_ratio = n_training_steps // n_inference_steps
    # creates integer timesteps by multiplying by ratio
    # casting to int to avoid issues when num_inference_step is power of 3
    timesteps = (
        (np.arange(0, n_inference_steps) * step_ratio).round().copy().astype(np.int64)
    )
    timesteps += steps_offset

    return timesteps


@dataclass(frozen=True)
class GaussianDiffusion1DSample(DiffusionSample):
    l: torch.Tensor = None
    z: torch.Tensor = None
    mask: torch.Tensor = None

    def get_x_for_detokenizer(self) -> torch.Tensor:
        return self.x


class GaussianDiffusion1D(Diffusion):
    name: str = "gaussian_diffusion_1d"

    def __init__(
        self,
        model: Dit1D,
        config: Config,
    ):
        super().__init__()
        self.model = model
        self.config = config

        self.seq_length = self.config.max_n_prims

        assert self.config.diffusion.objective in {
            "pred_noise",
            "pred_x0",
            "pred_v",
        }, "objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])"

        # First, get schedule with TRAINING_TIMESTEPS
        self.n_training_timesteps = self.config.diffusion.training_timesteps
        if self.config.diffusion.noise_schedule == "linear":
            betas = linear_beta_schedule(
                self.n_training_timesteps,
                beta0=self.config.diffusion.linear_beta0,
                betaT=self.config.diffusion.linear_betaT,
            )
        elif self.config.diffusion.noise_schedule == "cosine":
            betas = cosine_beta_schedule(
                self.n_training_timesteps,
                s=self.config.diffusion.cosine_s,
            )
        elif self.config.diffusion.noise_schedule == "sqrt":
            betas = sqrt_beta_schedule(
                self.n_training_timesteps,
                s=self.config.diffusion.sqrt_s,
            )
        elif self.config.diffusion.noise_schedule == "sigmoid":
            betas = sigmoid_beta_schedule(
                self.n_training_timesteps,
                start=self.config.diffusion.sigmoid_start,
                end=self.config.diffusion.sigmoid_end,
                tau=self.config.diffusion.sigmoid_tau,
            )
        elif self.config.diffusion.noise_schedule == "snr_based":
            betas = snr_based_beta_schedule(
                self.n_training_timesteps,
                snr_max=self.config.diffusion.snr_max,
                snr_min=self.config.diffusion.snr_min,
                snr_power=self.config.diffusion.snr_power,
            )
        else:
            raise ValueError(
                f"unknown beta schedule {self.config.diffusion.noise_schedule}"
            )

        # sampling related paramete rs

        self.n_inference_timesteps = default(
            self.config.diffusion.inference_timesteps, self.n_training_timesteps
        )  # default num sampling timesteps to number of timesteps at training

        assert self.n_inference_timesteps <= self.n_training_timesteps
        # self.is_ddim_sampling = self.sampling_timesteps < n_timesteps
        # self.ddim_sampling_eta = self.config.diffusion.ddim_sampling_eta

        # DEBUG for now because DDIM isn't working
        # assert (
        #     self.sampling_timesteps == timesteps
        # ), f"Sampling timesteps must match training timesteps:{self.config.diffusion.noise_schedule}, {self.sampling_timesteps} != {timesteps}"

        # helper function to register buffer from float64 to float32

        # Crop timesteps base on n_inference_timesteps
        self.timesteps = get_timesteps(
            self.n_training_timesteps, self.n_inference_timesteps
        )
        betas = betas[self.timesteps]

        # ======================================
        # Compute tabulated schedule constants
        # ======================================

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        alphas_cumprod_next = F.pad(alphas_cumprod[1:], (0, 1), value=1.0)

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.alphas_cumprod_next = alphas_cumprod_next
        # register_buffer("alphas_cumprod_next", alphas_cumprod_next)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # calculate loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        if self.config.diffusion.objective == "pred_noise":
            loss_weight = torch.ones_like(snr)
        elif self.config.diffusion.objective == "pred_x0":
            loss_weight = snr
        elif self.config.diffusion.objective == "pred_v":
            loss_weight = snr / (snr + 1)

        register_buffer("loss_weight", loss_weight)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(
            self.posterior_variance, t, x_t.shape
        ) * torch.ones_like(x_t)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        ) * torch.ones_like(x_t)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(
        self,
        predicted_x: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor = None,
        mask: torch.Tensor = None,
        clip_x_start: bool = False,
        rederive_pred_noise: bool = False,
    ) -> ModelPrediction:
        """

        :param x: Tensor of B x n_seq x C
        :param t:
        :param z:
        :param mask:
        :param clip_x_start:
        :param rederive_pred_noise:
        :param predicted_x:
        :return:
        """
        model_output = predicted_x
        maybe_clip = (
            partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        )

        if self.config.diffusion.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.config.diffusion.objective == "pred_x0":
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.config.diffusion.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        else:
            raise ValueError(f"{self.config.diffusion.objective} not allowed")

        return ModelPrediction(pred_noise=pred_noise, pred_x_start=x_start)

    def p_mean_variance(
        self,
        predicted_x: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = False,
    ):
        preds = self.model_predictions(predicted_x=predicted_x, x=x, t=t)
        x_start = preds.pred_x_start

        if clip_denoised:
            assert False
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def _model_inference(
        self,
        x,
        t: int,
        cfg_scale: float,
        z=None,
        mask=None,
    ):
        """
        Performs inference and handles CFG + self-conditioning if requested
        """
        b, *_, device = *x.shape, x.device
        use_cfg = cfg_scale != 1.0
        model_fn = (
            partial(self.model.forward_with_cfg, cfg_scale=cfg_scale)
            if use_cfg
            else self.model
        )
        batched_times = torch.full((b,), t, device=x.device, dtype=torch.long)
        predicted_model_x = model_fn(
            x.permute(0, 2, 1),
            t=batched_times,
            z=z,
            mask=mask,
        ).permute(0, 2, 1)

        return predicted_model_x, batched_times

    @torch.no_grad()
    def p_sample(
        self,
        x,
        t: int,
        cfg_scale: float,
        z=None,
        mask=None,
        clip_denoised=False,
        noise_t_threshold: int = 0,
    ):
        """

        :param x: Tensor of B x c x n_seq
        :param t:
        :param cfg_scale:
        :param z:
        :param mask:
        :param clip_denoised:
        :return:
        """

        # ===================
        # INFERENCE
        # ===================
        predicted_model_x, batched_times = self._model_inference(
            x, t, cfg_scale=cfg_scale, z=z, mask=mask
        )

        # ===================
        # LOGIC
        # ===================

        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            predicted_x=predicted_model_x,
            x=x,
            t=batched_times,
            clip_denoised=clip_denoised,
        )
        noise = (
            torch.randn_like(x) if t > noise_t_threshold else 0.0
        )  # no noise if t == 0
        # noise = 0.0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def ddim_sample(
        self,
        x,
        t,
        cfg_scale: float,
        z=None,
        mask=None,
        clip_denoised=False,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """

        # ===================
        # INFERENCE
        # ===================
        predicted_model_x, batched_times = self._model_inference(
            x, t, cfg_scale=cfg_scale, z=z, mask=mask
        )

        # ===================
        # LOGIC
        # ===================

        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            predicted_x=predicted_model_x,
            x=x,
            t=batched_times,
            clip_denoised=clip_denoised,
        )

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self.predict_noise_from_start(x, batched_times, x_start)

        alpha_bar = extract(self.alphas_cumprod, batched_times, x.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, batched_times, x.shape)

        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = (
            x_start * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps
        )
        nonzero_mask = (
            (batched_times != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return sample, x_start

    @torch.no_grad()
    def ddim_reverse_sample(
        self,
        x,
        t,
        cfg_scale: float,
        z=None,
        mask=None,
        clip_denoised=False,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.

        WARNING: be careful if ever inverting with CFG (cf. null-inversion paper)
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"

        # ===================
        # INFERENCE
        # ===================
        predicted_model_x, batched_times = self._model_inference(
            x, t, cfg_scale=cfg_scale, z=z, mask=mask
        )

        # ===================
        # LOGIC
        # ===================

        model_mean, _, model_log_variance, pred_x_start = self.p_mean_variance(
            predicted_x=predicted_model_x,
            x=x,
            t=batched_times,
            clip_denoised=clip_denoised,
        )

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, batched_times, x.shape)
            * x
            - pred_x_start
        ) / _extract_into_tensor(
            self.sqrt_recipm1_alphas_cumprod, batched_times, x.shape
        )
        alpha_bar_next = _extract_into_tensor(
            self.alphas_cumprod_next, batched_times, x.shape
        )

        # Equation 12. reversed
        mean_pred = (
            pred_x_start * torch.sqrt(alpha_bar_next)
            + torch.sqrt(1 - alpha_bar_next) * eps
        )

        return mean_pred, pred_x_start

    def _x_to_sample(self, x: torch.Tensor, use_cfg: bool) -> torch.Tensor:
        if use_cfg:
            x = x[: len(x) // 2]
        return x.permute((0, 2, 1))

    def _get_jump_length_for_step(self, t: int, start_t: int, jump_length: int) -> int:
        """Returns the jump length if this timestep should jump, otherwise 0"""

        if t > start_t:
            return 0

        if t % jump_length == 0:
            return jump_length
        return 0

    @torch.no_grad()
    def _process_jump(
        self,
        t: int,
        x: torch.Tensor,
        cfg_scale: float,
        z=None,
        mask=None,
        noise_t_threshold: int = 0,
        start_t: int = 900,
        jump_length: int = 100,
        n_repetitions: int = 5,
    ) -> torch.Tensor:
        """
        Handles a single resampling jump at timestep t.
        The strategy goes as follows:
        If t is associated to a jump step, jump from t to t + jump_length
        and denoise all the way back to t (from jump_length)

        # NB: there's only one parameter now, the jump_length because the number of resampling step maps exactly to it!

        """
        jump_length = self._get_jump_length_for_step(
            t, start_t=start_t, jump_length=jump_length
        )
        if jump_length == 0:
            return x

        x_tmp = x.clone()

        for _ in range(n_repetitions):

            # Go from t to t + jump_length: [t->t+1, t+1->t+2, ..., t+jump_length-1->t+jump_length]
            for i_jump in range(jump_length):
                # Jump forward using q_sample
                t_to = t + i_jump
                x_tmp = self.undo(x_tmp, t_to)

            # Denoise the other way around!
            for i_jump in reversed(range(jump_length)):  # -1!!! = REVERSE here!

                cur_t = t + i_jump

                x_tmp, _ = self.p_sample(
                    x=x_tmp,
                    t=cur_t,
                    cfg_scale=cfg_scale,
                    z=z,
                    mask=mask,
                    noise_t_threshold=noise_t_threshold,
                )
                x_tmp = x_tmp

        return x_tmp

    @torch.no_grad()
    def p_sample_loop(
        self,
        batch_size: int,
        return_traj: bool = False,
        z: Union[torch.Tensor, None] = None,
        mask: Union[torch.Tensor, None] = None,
        cfg_scale: float = 1.0,
        traj_stride: int = 100,
        noise_t_threshold: int = 0,
        # Resampling parameters
        use_resampling: bool = False,
        resampling_jump_length: int = 100,
        resampling_repetitions: int = 5,
        resampling_start_t: int = 900,
    ) -> Tuple[GaussianDiffusion1DSample, List[GaussianDiffusion1DSample]]:
        use_cfg = (cfg_scale != 1.0) and self.model.z_conditioning
        assert not use_cfg or (
            self.config.diffusion.z_conditioning
            and self.config.diffusion.model["options"]["uncond_prob"] > 0.0
        )
        print(f"Sampling with cfg_scale={cfg_scale:.2f}")

        device = self.betas.device
        shape = (
            batch_size,
            self.model.input_dim,
            self.model.seq_length,
        )

        x = torch.randn(shape, device=device)

        if use_cfg:
            null_z = self.model.z_embedding.null_embedding[None].repeat(
                z.shape[0], 1, 1
            )
            z = torch.cat([z, null_z])
            x = x.repeat(2, 1, 1)
            if mask is not None:
                mask = mask.repeat(2, 1)

        full_traj = (
            [GaussianDiffusion1DSample(x=self._x_to_sample(x, use_cfg), z=z, mask=mask)]
            if return_traj
            else None
        )

        for t in tqdm(
            reversed(self.timesteps),
            desc="sampling loop time step",
            total=self.n_inference_timesteps,
        ):
            x, _ = self.p_sample(
                x=x,
                t=t,
                z=z,
                mask=mask,
                cfg_scale=cfg_scale,
                noise_t_threshold=noise_t_threshold,
            )

            # If requested, simple jump
            if use_resampling:
                x = self._process_jump(
                    t=t,
                    x=x,
                    cfg_scale=cfg_scale,
                    z=z,
                    mask=mask,
                    noise_t_threshold=noise_t_threshold,
                    start_t=resampling_start_t,
                    jump_length=resampling_jump_length,
                    n_repetitions=resampling_repetitions,
                )

            # Record trajectory if requested
            if return_traj:
                if t % traj_stride == 0:
                    full_traj.append(
                        GaussianDiffusion1DSample(
                            x=self._x_to_sample(x, use_cfg),
                            z=z,
                            mask=mask,
                        )
                    )

        return (
            GaussianDiffusion1DSample(x=self._x_to_sample(x, use_cfg), z=z, mask=mask),
            full_traj,
        )

    @torch.no_grad()
    def sample(self, batch_size=16):
        assert False
        seq_length, dim = self.seq_length, self.config.x_dim
        sample_fn = (
            self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        )
        return sample_fn((batch_size, dim, seq_length))

    @autocast(enabled=False)
    def q_sample(self, x_start, t, noise=None):
        """
        :param x_start: B x C x n_seq
        :param t: B x
        :param noise:
        :return:
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # Adds noise from x_t to x_{t+1}
    def undo(self, x: torch.Tensor, t: Union[torch.Tensor, int]):

        beta = _extract_into_tensor(
            self.betas,
            (
                torch.ones(x.shape[0], device=x.device, dtype=torch.long) * t
                if not isinstance(t, torch.Tensor)
                else t
            ),
            x.shape,
        )

        new_x = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * torch.randn_like(x)

        return new_x

    # DEPRECATED: I kept this one around because you use it @Mingi but it should be removed because it doesn't handle inference timesteps properly!
    @torch.no_grad()
    def q_sample_sequential(
        self, x_t: torch.Tensor, t: int, noise=None
    ) -> torch.Tensor:
        """Sample x_{t+1} from x_t (single forward step)

        Args:
            x_t: Current noisy sample at time t
            t: Current timestep
            noise: Optional noise to use (will generate if None)

        Returns:
            x_{t+1}: Next noisy sample
        """
        # Get alpha values for current and next timestep
        alpha_t = self.alphas_cumprod[t]
        alpha_next = (
            self.alphas_cumprod[t + 1]
            if t < self.n_training_timesteps - 1
            else torch.tensor(0.0)
        )

        # Calculate coefficient for noise
        c = torch.sqrt(1 - alpha_next / alpha_t)

        # Generate or use provided noise
        noise = default(noise, lambda: torch.randn_like(x_t))

        # Get next sample
        x_next = torch.sqrt(alpha_next / alpha_t) * x_t + c * noise

        return x_next

    def p_losses(self, x_start, t, z=None, mask=None, noise=None):
        """
        :param x_start: Tensor of B x C x n_seq
        :param t:
        :param z:
        :param mask:
        :param noise:
        :return:
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        if self.config.diffusion.z_augmentation:
            device = x_start.device
            z_t = torch.randint(
                0, self.config.diffusion.z_aug_step, (x_start.shape[0],), device=device
            ).long()
            z_noise = torch.randn_like(z)
            z = self.q_sample(x_start=z, t=z_t, noise=z_noise)

        model_out = self.model(
            x.permute(0, 2, 1),
            t,
            z=z,
            mask=mask,
        ).permute(0, 2, 1)
        if self.config.diffusion.objective == "pred_noise":
            target = noise
        elif self.config.diffusion.objective == "pred_x0":
            target = x_start
        elif self.config.diffusion.objective == "pred_v":
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f"unknown objective {self.config.diffusion.objective}")

        return model_out.permute(0, 2, 1), target.permute(0, 2, 1)
        # loss = F.mse_loss(model_out, target, reduction="none")

        # if mask is not None:
        #     loss = loss.masked_fill(mask.unsqueeze(1).expand(-1, c, -1), 0)
        # loss = reduce(loss, "b ... -> b", "mean")

        # loss = loss * extract(self.loss_weight, t, loss.shape)
        # return loss.mean()

    def forward(
        self,
        tokens: Tokens,
        z: torch.Tensor,
        mask: torch.Tensor,
        prefix: str,
        empty_embeddings: Union[torch.Tensor, None],
        return_timesteps: bool = False,
        t: Union[torch.Tensor, None] = None,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """

        :param tokens: sample with tensor B x n_seq x channel
        :param z:
        :param mask:
        :param prefix:
        :param empty_embeddings:
        :return:
        """
        sample: torch.Tensor = tokens.sample.permute(0, 2, 1)
        # WARNING: assumes 1D image-like convention with sequence dimension at the
        # end
        assert (
            sample.shape[2] == self.seq_length
        ), f"seq length must be {self.seq_length}"
        if t is None:
            t = torch.randint(
                0, self.n_training_timesteps, (sample.shape[0],), device=sample.device
            ).long()

        out, target = self.p_losses(sample, t, z=z, mask=mask)

        if return_timesteps:
            return out, target, t
        else:
            return out, target

        # diffusion_loss = self.p_losses(sample, t, z=z, mask=mask)
        # return diffusion_loss, {f"loss/{prefix}/diffusion": [diffusion_loss.tolist()]}

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = arr.to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
