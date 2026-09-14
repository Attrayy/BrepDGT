from __future__ import annotations
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import os
from enum import Enum
from copy import copy, deepcopy
from tqdm import tqdm
import glob

import torch
import numpy as np
import trimesh

import polyscope as ps
import polyscope.imgui as psim

from ps_utils.ui import state_button, KEY_HANDLER, save_popup
from ps_utils.structures import create_bbox

from brepdgt.primitives.uvgrid import UvGrid, stack_uvgrids, concat_uvgrids
from brepdgt.models.tokenizers.base import Tokens
from brepdgt.diffusion import Diffusion, DIFFUSION_PROCESSES
from brepdgt.config import Config
from brepdgt.models.brepdgt import BRepDGT
from brepdgt.viewer.uv_grid_widget import (
    UvGridWidget,
    RenderMode,
    RENDER_MODE_INVMAP,
    RENDER_MODE_MAP,
)
from brepdgt.viewer.denoiser_config import (
    SchedulerType,
    SCHEDULER_TYPE_INVMAP,
    SCHEDULER_TYPE_MAP,
    ALLOWED_BATCH_SIZES,
    ALLOWED_BATCH_SIZES_MAP,
    ALLOWED_BATCH_SIZES_INVMAP,
    ALLOWED_EXPORT_STRIDES,
    ALLOWED_EXPORT_STRIDES_INVMAP,
    ALLOWED_EXPORT_STRIDES_MAP,
    ALLOWED_INFERENCE_STEPS,
    ALLOWED_INFERENCE_STEPS_INVMAP,
    ALLOWED_INFERENCE_STEPS_MAP,
)
from brepdgt.viewer.trajectory import (
    Trajectory,
    TrajectoryType,
    DenoiserConfig,
    get_empty_mask,
    get_base_tokens,
)
from brepdgt.viewer.trajectory_handler import TrajectoryHandler, SLIDER_HALF_WIDTH
from brepdgt.utils.convert_utils import (
    # uvgrid_to_brep,
    PpViewerOutput,
    PpSuccessState,
    uvgrid_to_brep_or_mesh,
    brep_to_mesh,
    brep_to_uvgrid,
)
from brepdgt.models.detokenizers.pre_mask_uv_detokenizer_3d import DedupDetokenizer3D
from brepdgt.viewer.dataset_loader import DatasetLoader

from OCC.Extend.DataExchange import write_step_file

# =========================
# DENOISER CONSTANTS
# =========================


TRAJ_SAVE_FOLDER = "out/trajectories"
RECONSTRUCTED_BREP_MESH = "reconstructed_brep_denoiser"
BREP_SAVE_FOLDER = "out/breps"


# Performs denoising and step at each time
# NOTE: this is mostly an abstraction to keep the viewer much cleaner
class Denoiser:
    def __init__(
        self,
        zooc_config: Config,
        model: BRepDGT,
        config: DenoiserConfig = DenoiserConfig(),
        headless_mode: bool = False,
    ):
        self.headless_mode = headless_mode
        self.zooc_config = zooc_config

        self.model = model

        self.uv_grid_widget = UvGridWidget()

        self.render_mode = RenderMode.POINT_CLOUD
        self.render_brep = False

        self.trajectory_handler = TrajectoryHandler()

        self.denoising = False
        self.renoising = False
        self.ps_reconstructed_brep = None
        self.current_brep = None  # Store the current brep object
        self.current_pp_status = ""
        self.traj_save_path = self._get_next_save_path()
        self.uv_grids_save_path = UvGridWidget.get_next_save_path()
        self.brep_save_path = self._get_next_brep_save_path()
        self.brep_grid_res = 64  # Default starting value
        self.brep_smooth_extension = True
        self.brep_extension_len = 1.0
        self.brep_min_extension_len = 0.0
        self.brep_psr_occ_thresh = 0.5
        self.brep_verbose = True
        self.brep_try_all = False
        self.append_mode = False

        self.init_trajectory(config=config)

        # DEBUG
        # self.ps_drop_callback(
        #     "data/deepcad_30_val_300/00000178_b21aa794b6b64d87a57c245e_step_008.step"
        # )
        # self.ps_drop_callback(
        #     "data/deepcad_30_val_300/00000354_68d1a5c2319a4a419ff625c6_step_000.step"
        # )
        # self.ps_drop_callback("./out/trajectories/exported_000006.traj")
        # self.trajectory_handler.delete(0)
        # self.trajectory_handler.interp_i1 = 0
        # self.trajectory_handler.interp_i2 = 1

        self.dataset_loader = None

        self.current_pp_output = None

    @torch.no_grad()
    def init_trajectory(
        self,
        uv_grids: Optional[UvGrid] = None,
        frozen_prims: List[bool] = None,
        config: DenoiserConfig = DenoiserConfig(),
        trajectory_type: TrajectoryType = TrajectoryType.BACKWARD,
        replace_current: bool = False,
        t: int | None = None,
        i_step: int | None = None,
        i_t: int = 0,
        i_batch: int = 0,
    ):
        """
        Reset the current trajectory. If a uv_grids is provided, it is always considered to be set at t=0.
        """
        # Make sure to make a copy of the config!
        config = deepcopy(config)
        config.n_training_steps = self.zooc_config.diffusion.training_timesteps

        # Create a new ZOOC tmp_config for the diffusion
        # WARNING: this is quite DIY but the only (easy) way to get variable inference steps
        tmp_config = copy(self.zooc_config)
        tmp_config.diffusion.inference_timesteps = config.n_inference_steps
        self.diffusion_process: Diffusion = DIFFUSION_PROCESSES[
            tmp_config.diffusion.name
        ](self.model.backbone, tmp_config).cuda()

        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        if t is not None and i_step is not None:
            pass
        elif trajectory_type == TrajectoryType.BACKWARD:
            # Make sure to set n_inference_steps (e.g., 1000)
            # because i_step is automatically decremented to yield 999
            # (last of the schedule)
            # This allows to properly keep the boundaries!
            # TODO: this is wobbly (the "is None" test in particular)
            # WARNING: this doesn't work!!!
            i_step = config.n_inference_steps if uv_grids is None else self.i_step
            t: int = self.zooc_config.diffusion.training_timesteps
        elif trajectory_type == TrajectoryType.FORWARD:
            i_step = 0
            t: int = 0
        elif trajectory_type == TrajectoryType.INPAINTING:
            # NOTE: for now, inpainting is only supported with all
            assert (
                config.n_inference_steps
                == self.zooc_config.diffusion.training_timesteps
            )
            i_step = min(
                self.zooc_config.diffusion.training_timesteps,
                config.inpainting.n_inpainting_steps,
            )  # just sanity!
            t: int = min(
                self.zooc_config.diffusion.training_timesteps,
                config.inpainting.n_inpainting_steps,
            )  # just sanity!
        else:
            raise ValueError()

        if uv_grids is not None:
            # Make sure to batch and pad uv-grid
            uv_grids.to_tensor("cuda")
            if len(uv_grids.coord.shape) == 4:
                uv_grids = stack_uvgrids([uv_grids])
            uv_grids.pad(self.zooc_config.max_n_prims)
            if config.batch_size > 1:
                # TODO: copy uv-grids
                if uv_grids.coord.shape[0] == 1:
                    uv_grids = uv_grids.repeat(config.batch_size)
                elif uv_grids.coord.shape[0] == config.batch_size:
                    pass
                else:
                    raise ValueError()

            # Tokenize the uv grid
            self.x_denoised = self.model.token_vae.tokenizer(
                self.current_trajectory.base_tokens.labels,  # TODO: check that, this is dirty....
                uv_grids,
            ).sample.cuda()
            self.x_start = self.x_denoised.clone()

            # Empty mask
            empty_mask = uv_grids.empty_mask

        else:
            # Create random tokens for reverse/noise-denoise trajectories
            x = torch.randn(
                config.batch_size,
                self.zooc_config.max_n_prims,
                self.zooc_config.x_dim,
                device="cuda",
            )  # [batch_size, 10, 448]

            # TODO: this is a little bit dirty here...
            self.x_denoised = x
            self.x_start = None

            # Empty mask
            empty_mask = get_empty_mask(
                batch_size=config.batch_size,
                n_prims=config.n_prims,
                max_n_prims=self.zooc_config.max_n_prims,
            )

        # For inpainting, we need to add noise to x_denoised (and recompute the UV-grid)!
        if trajectory_type == TrajectoryType.INPAINTING:
            # NB: we need to take a -1 step because that' i_step is automatically
            # decreased in `step_denoise`
            self.x_denoised = self._add_noise(
                self.x_denoised, self._get_practical_timesteps(i_step - 1)
            )

        base_tokens = get_base_tokens(
            self.x_denoised,
            max_n_prims=self.zooc_config.max_n_prims,
            empty_mask=empty_mask,
        )

        if frozen_prims is None:
            frozen_prims = [False for _ in range(self.zooc_config.max_n_prims)]
        else:
            frozen_prims = frozen_prims

        # Force POINT_CLOUD render mode
        self.render_mode = RenderMode.POINT_CLOUD
        # NB: t is determined from the diffusion schedule directly

        # `trajectory` holds all the tokens and uv_grids of the trajectory
        uv_grids = self._detokenize(base_tokens, deduplicate=i_step == 0)

        print("####### Created trajectory ########")
        print("i_step=", i_step)
        print("t=", t)
        print("trajectory_type=", trajectory_type.value)

        trajectory = Trajectory(
            t=t,
            base_tokens=base_tokens,
            uv_grids=uv_grids,
            frozen_prims=frozen_prims,
            denoiser_config=config,
            i_step=i_step,
            trajectory_type=trajectory_type,
            i_t=i_t,
            i_batch=i_batch,
        )
        self.trajectory_handler.add(trajectory, replace_current)

        if not self.headless_mode:
            self.uv_grid_widget.set_uv_grids(
                uv_grids.extract(
                    self.i_batch
                ),  # WARNING: works because of above `.add(...)`
                render_mode=self.render_mode,
            )
            create_bbox()

    # Called after every step is made (either noise, renoise, or denoise)
    # t is the corresponding t, not i_step!!!
    def after_step(self, t: int):
        self.update_uv_grids(t)

    def _get_practical_timesteps(self, i_step: int):
        """Utility function to also return n_training_steps at the end of timesteps"""
        return (
            self.diffusion_process.timesteps[i_step]
            if i_step < len(self.diffusion_process.timesteps)
            else self.config.n_training_steps
        )

    def _update_batch_size(self, new_batch_size: int):
        """Called when batch size is changed"""
        t = self.current_trajectory.keymap[self.i_t]
        if self.current_trajectory.has(t):
            uv_grids = self.current_trajectory.traj_uv_grids.get(t)
            current_batch_size = self.config.batch_size
            if new_batch_size < current_batch_size:
                uv_grids = uv_grids.extract(
                    (np.arange(new_batch_size) + self.i_batch)
                    % new_batch_size  # This makes sure to take the  current i_batch (which will be set to 0 but that's fine!)
                )
            else:
                # Should always be true because a power of 2
                assert new_batch_size % current_batch_size == 0
                n_repeat = new_batch_size // current_batch_size
                uv_grids = uv_grids.repeat(n_repeat=n_repeat)

            # Only then can we update the uv-grid size!
            new_config = deepcopy(self.config)
            new_config.batch_size = new_batch_size
            self.init_trajectory(
                uv_grids=uv_grids,
                trajectory_type=(
                    TrajectoryType.FORWARD if t == 0 else TrajectoryType.BACKWARD
                ),
                config=new_config,
                replace_current=self.current_trajectory.size == 1,
                t=t,
            )
        else:
            print(
                f"WARNING: cannot update batch size because the current trajectory doesn't have t={t}!"
            )

    # =====================================
    # Shortcuts for current trajectory
    # =====================================

    @property
    def config(self) -> DenoiserConfig:
        return self.trajectory_handler.current_trajectory.denoiser_config

    @property
    def current_trajectory(self) -> Trajectory:
        return self.trajectory_handler.current_trajectory

    @property
    def i_step(self) -> int:
        return self.trajectory_handler.current_trajectory.i_step

    def set_i_step(self, i_step: int) -> None:
        self.trajectory_handler.current_trajectory.i_step = i_step

    @property
    def i_t(self) -> int:
        return self.trajectory_handler.current_trajectory.i_t

    def set_i_t(self, i_t: int) -> None:
        self.trajectory_handler.current_trajectory.i_t = i_t

    @property
    def i_batch(self) -> int:
        return self.trajectory_handler.current_trajectory.i_batch

    def set_i_batch(self, i_batch: int) -> None:
        self.trajectory_handler.current_trajectory.i_batch = i_batch

    # =====================================
    # Shortcuts for state handling
    # =====================================

    @property
    def active(self):
        return self.denoising or self.renoising

    # =====================================
    # Diffusion Stuff (Noise/Denoise)
    # =====================================

    # Simply add noise
    def _add_noise(self, x: torch.Tensor, t: int):
        t_tensor = (
            torch.ones(
                x.shape[0],
                device=x.device,
                dtype=torch.long,
            )
            * t
        )
        return self.model.diffusion_process.q_sample(
            x_start=x.permute(0, 2, 1), t=t_tensor
        ).permute(0, 2, 1)

    def _scheduler_step_backward(self, t: int):
        if self.config.scheduler_type == SchedulerType.DDPM:
            self.x_denoised, _ = self.model.diffusion_process.p_sample(
                x=self.x_denoised.permute(0, 2, 1),
                t=t,
                cfg_scale=self.config.cfg_scale,  # TODO: make this configurable
                z=self.current_trajectory.base_tokens.condition,
                mask=self.current_trajectory.base_tokens.mask,
                clip_denoised=False,
                noise_t_threshold=self.config.noise_t_threshold,
            )
            self.x_denoised = self.x_denoised.permute(0, 2, 1)
        elif self.config.scheduler_type == SchedulerType.DDIM:
            self.x_denoised, _ = self.model.diffusion_process.ddim_sample(
                x=self.x_denoised.permute(0, 2, 1),
                t=t,
                cfg_scale=self.config.cfg_scale,  # TODO: make this configurable
                z=self.current_trajectory.base_tokens.condition,
                mask=self.current_trajectory.base_tokens.mask,
                clip_denoised=False,
            )
            self.x_denoised = self.x_denoised.permute(0, 2, 1)
        else:
            raise ValueError()

    def _scheduler_step_forward(self, t: int):
        if self.config.scheduler_type == SchedulerType.DDPM:
            # t_tensor = (
            #     torch.ones(
            #         self.config.batch_size,
            #         device=self.x_denoised.device,
            #         dtype=torch.long,
            #     )
            #     * t
            # )
            self.x_denoised = self.model.diffusion_process.undo(
                x=self.x_denoised.permute(0, 2, 1), t=t
            ).permute(0, 2, 1)
        elif self.config.scheduler_type == SchedulerType.DDIM:
            self.x_denoised, _ = self.model.diffusion_process.ddim_reverse_sample(
                x=self.x_denoised.permute(0, 2, 1),
                t=t,
                cfg_scale=self.config.cfg_scale,  # TODO: make this configurable
                z=self.current_trajectory.base_tokens.condition,
                mask=self.current_trajectory.base_tokens.mask,
                clip_denoised=False,
            )
            self.x_denoised = self.x_denoised.permute(0, 2, 1)
        else:
            raise ValueError()

    # Sorry, this is hacky but we don't want to deduplicate until the end so I added this
    def _detokenize(self, tokens: Tokens, deduplicate: bool = False):
        # Detokenize again
        if isinstance(self.model.token_vae.detokenizer, DedupDetokenizer3D):
            uv_grids = self.model.token_vae.detokenizer.decode(
                tokens, self.model.config.max_n_prims, deduplicate=deduplicate
            )
        else:
            uv_grids = self.model.token_vae.detokenizer.decode(
                tokens, self.model.config.max_n_prims
            )
        return uv_grids

    def _get_jump_length_for_step(self, t: int) -> int:
        """Returns the jump length if this timestep should jump, otherwise 0"""
        if not self.config.resampling.enabled:
            return 0

        if t > self.config.resampling.start_t:
            return 0

        if t % self.config.resampling.jump_length == 0:
            return self.config.resampling.jump_length
        return 0

    @torch.no_grad()
    def _process_jump(self, t: int, x: torch.Tensor) -> torch.Tensor:
        """
        Handles a single resampling jump at timestep t.
        The strategy goes as follows:
        If t is associated to a jump step, jump from t to t + jump_length
        and denoise all the way back to t (from jump_length)

        # NB: there's only one parameter now, the jump_length because the number of resampling step maps exactly to it!

        """
        jump_length = self._get_jump_length_for_step(t)
        if jump_length == 0:
            return x

        x_tmp = x.clone()

        for _ in range(self.config.resampling.n_repetitions):

            # Go from t to t + jump_length: [t->t+1, t+1->t+2, ..., t+jump_length-1->t+jump_length]
            for i_jump in range(self.config.resampling.jump_length):
                # Jump forward using q_sample
                t_to = t + i_jump
                x_tmp = self.diffusion_process.undo(
                    x_tmp.permute(0, 2, 1), t_to
                ).permute(0, 2, 1)

            # Denoise the other way around!
            for i_jump in reversed(
                range(self.config.resampling.jump_length)
            ):  # -1!!! = REVERSE here!

                cur_t = t + i_jump

                x_tmp, _ = self.diffusion_process.p_sample(
                    x=x_tmp.permute(0, 2, 1),
                    t=cur_t,
                    cfg_scale=self.config.cfg_scale,
                    z=self.current_trajectory.base_tokens.condition,
                    mask=self.current_trajectory.base_tokens.mask,
                    clip_denoised=False,
                    noise_t_threshold=self.config.noise_t_threshold,
                )
                x_tmp = x_tmp.permute(0, 2, 1)

        return x_tmp

    @torch.no_grad()
    def step_denoise(self) -> bool:
        if self.i_step <= 0:
            return False

        self.set_i_step(self.i_step - 1)
        t = self._get_practical_timesteps(self.i_step)

        # Inpainting: 1. add noise to x_start (and overwrite)
        if self.current_trajectory.trajectory_type == TrajectoryType.INPAINTING:
            assert self.x_start is not None and any(
                self.current_trajectory.frozen_prims
            )
            frozen_indices = torch.tensor(
                self.current_trajectory.frozen_prims,
                device=self.x_denoised.device,
                dtype=torch.bool,
            )
            x_start_noised = self._add_noise(self.x_start, t)
            self.x_denoised[:, frozen_indices] = x_start_noised[:, frozen_indices]

        # Scheduler step
        self._scheduler_step_backward(t)

        # Process resampling jump if needed
        # NB: this can be used both for inpainting and robustifying denoising
        # NB: this has to be done before the inpainting post-step below!
        if self.config.resampling.enabled:
            self.x_denoised = self._process_jump(t, self.x_denoised)

        # Handle inpainting post-step
        if (
            self.current_trajectory.trajectory_type == TrajectoryType.INPAINTING
            and self.i_step == 0
        ):
            frozen_indices = torch.tensor(
                self.current_trajectory.frozen_prims,
                device=self.x_denoised.device,
                dtype=torch.bool,
            )
            self.x_denoised[:, frozen_indices] = self.x_start[:, frozen_indices]

        # Create new tokens with denoised sample
        if not self.config.fast_mode or self.current_trajectory._is_exportable_t(t):
            denoised_tokens = Tokens(
                sample=self.x_denoised,
                mask=self.current_trajectory.base_tokens.mask,
                condition=self.current_trajectory.base_tokens.condition,
                center_coord=self.current_trajectory.base_tokens.center_coord,
            )

            uvgrids_denoised: UvGrid = self._detokenize(
                denoised_tokens, self.i_step == 0
            )
            uvgrids_denoised.to_cpu()  # Move all tensors to CPU

            self.current_trajectory.add(t, uv_grids=uvgrids_denoised)
            self.set_i_t(0)
            self.after_step(t)

        return True

    @torch.no_grad()
    def step_renoise(self) -> bool:
        if (
            self.i_step >= self.config.renoise_steps
            or self.i_step >= self.config.n_inference_steps
        ):
            return False

        t = self._get_practical_timesteps(self.i_step)

        self._scheduler_step_forward(t)

        self.set_i_step(self.i_step + 1)

        t = self._get_practical_timesteps(self.i_step)
        if not self.config.fast_mode or self.current_trajectory._is_exportable_t(t):
            noisy_tokens = Tokens(
                sample=self.x_denoised,
                mask=self.current_trajectory.base_tokens.mask,
                condition=self.current_trajectory.base_tokens.condition,
                center_coord=self.current_trajectory.base_tokens.center_coord,
            )
            uvgrids_noisy = self._detokenize(noisy_tokens)
            uvgrids_noisy.to_cpu()  # Move all tensors to CPU

            # We write on the next one!

            self.current_trajectory.add(t, uv_grids=uvgrids_noisy)

            # i_t is always set to  trajectory_size - 1 during denoising!
            self.set_i_t(self.current_trajectory.size - 1)
            self.after_step(t)

        return True

    @torch.no_grad()
    def _inpaint_current(self):
        uv_grids, frozen_prims = self.uv_grid_widget.get_current()

        new_config = deepcopy(self.config)
        if self.config.inpainting.override_batch_size:
            new_config.batch_size = self.config.inpainting.empty_batch_multiplier * (
                self.config.inpainting.empty_max - self.config.inpainting.empty_min + 1
            )
            all_uv_grids = []
            for i_mask in range(
                self.config.inpainting.empty_min, self.config.inpainting.empty_max + 1
            ):
                # Fine because things are duplicated!
                this_uv_grids = uv_grids.repeat(
                    self.config.inpainting.empty_batch_multiplier
                )
                this_uv_grids.unmask_N_first(i_mask)
                all_uv_grids.append(this_uv_grids)
            uv_grids = concat_uvgrids(all_uv_grids, cat_dim=0)

        self.init_trajectory(
            uv_grids=uv_grids,
            frozen_prims=frozen_prims,
            trajectory_type=TrajectoryType.INPAINTING,
            config=new_config,
            i_batch=self.i_batch,  # Make sure to keep i_batch
        )
        self.denoising = True

    # =====================================
    # GUI
    # =====================================

    def gui(self) -> None:

        psim.SeparatorText("Render")

        psim.SetNextItemWidth(150)
        clicked, render_mode_idx = psim.SliderInt(
            "Render Mode",
            RENDER_MODE_MAP[self.render_mode],
            v_min=0,
            v_max=len(RenderMode) - 1,
            format=f"{self.render_mode.value}",
        )

        if clicked:
            self.update_uv_grids()
            self.render_mode = RENDER_MODE_INVMAP[render_mode_idx]
            self.uv_grid_widget.update_ps_structures(self.render_mode)

        if self.ps_reconstructed_brep is not None:
            psim.SameLine()
            clicked, self.render_brep = psim.Checkbox("Render Brep##", self.render_brep)
            if clicked:
                self.ps_reconstructed_brep.set_enabled(self.render_brep)

        # ==================================
        # Denoising / Renoising/ Inpainting
        # ==================================

        psim.SeparatorText("Control")

        is_current_first_t = self.current_trajectory.keymap[self.i_t] == 0

        # Disable Control if Denoiser is active
        psim.BeginDisabled(self.active)
        psim.BeginDisabled(is_current_first_t)
        clicked, self.denoising = state_button(
            self.denoising, "Stop##denoiser", "Denoise##denoiser"
        )
        psim.EndDisabled()

        psim.SameLine()
        if psim.Button("Reset##denoiser"):
            self.init_trajectory(config=self.config)
        psim.SameLine()
        if psim.Button("Copy##denoiser"):
            t = self._get_practical_timesteps(self.i_t)
            if self.current_trajectory.has(t):
                self.init_trajectory(
                    self.current_trajectory.traj_uv_grids.get(t),
                    frozen_prims=self.current_trajectory.frozen_prims,
                    config=self.config,
                    trajectory_type=(
                        TrajectoryType.FORWARD
                        if is_current_first_t
                        else TrajectoryType.BACKWARD
                    ),
                )

        # If it the first step, then we can renoise of inpaint
        psim.BeginDisabled(not is_current_first_t)
        # Renoise
        psim.SameLine()
        clicked, self.renoising = state_button(
            self.renoising,
            "Stop##denoiser_renoise",
            "Renoise/Invert##denoiser_renoise",
        )
        # If is was just clicked, get the corresponding uv-grid and create a new denoising trajectory
        if clicked and self.renoising:
            assert is_current_first_t
            uv_grids, frozen_prims = self.uv_grid_widget.get_current()
            self.init_trajectory(
                uv_grids=uv_grids,
                frozen_prims=frozen_prims,
                trajectory_type=TrajectoryType.FORWARD,
                config=self.config,
                replace_current=self.current_trajectory.size == 1,
            )

        # Inpaint
        # NB: this will create a new trajectory and automatically start denoising!
        psim.BeginDisabled(not any(self.current_trajectory.frozen_prims))
        psim.SameLine()
        if psim.Button("Inpaint##denoiser"):
            self._inpaint_current()

        psim.EndDisabled()
        psim.EndDisabled()

        psim.SameLine()
        if psim.Button("Clear"):
            self.trajectory_handler.clear()
            self.update_uv_grids()

        # ========================
        # Denoising
        # ========================

        if self.denoising:
            self.denoising = self.step_denoise()

        else:
            update = False
            # TODO: completely decorrelate this counter from the denoiser counter!
            clicked, i_t = psim.SliderInt(
                "t##denoiser",
                self.i_t,
                v_min=0,
                v_max=self.current_trajectory.size - 1,
                format=f"{self.current_trajectory.keymap[self.i_t]}",
            )
            self.set_i_t(i_t)
            update |= clicked
            # Arrow control (with KEY_HANDLER)
            if KEY_HANDLER("right_arrow"):
                self.set_i_t(
                    min(
                        self.i_t + 1,
                        self.current_trajectory.size - 1,
                    )
                )
                update = True
            if KEY_HANDLER("left_arrow"):
                self.set_i_t(max(self.i_t - 1, 0))
                update = True

            if update:
                self.update_uv_grids()

        # ========================
        # Renoising
        # ========================
        if self.renoising:
            self.renoising = self.step_renoise()

        # ========================
        # Batch
        # ========================

        clicked, i_batch = psim.SliderInt(
            "i_batch##denoiser",
            self.i_batch,
            v_min=0,
            v_max=self.config.batch_size - 1,
        )
        if clicked:
            self.set_i_batch(i_batch)
            self.update_uv_grids()

        # ========================
        # Dataset Loader
        # ========================
        if psim.TreeNode("Dataset Loader"):
            # Load it if necessary
            # NB: this is to avoid slowdowns
            if self.dataset_loader is None:
                self.dataset_loader = DatasetLoader(
                    zooc_config=self.model.config, model=self.model
                )
            updated = self.dataset_loader.gui()
            if updated:
                uv_grids = self.dataset_loader.get_current()
                self.init_trajectory(
                    uv_grids=uv_grids,
                    trajectory_type=TrajectoryType.FORWARD,
                    config=self.config,
                    replace_current=self.current_trajectory.size == 1,
                )

            psim.TreePop()

        # ========================
        # Save/Load
        # ========================

        psim.SeparatorText("Save/Load")

        clicked, self.uv_grids_save_path = save_popup(
            "uv_grids_denoiser", self.uv_grids_save_path, "Save UvGrid"
        )
        if clicked:
            self.uv_grid_widget.uv_grids.export_npz(self.uv_grids_save_path, 0)
            self.uv_grids_save_path = UvGridWidget.get_next_save_path()

        psim.SameLine()
        clicked, self.traj_save_path = save_popup(
            "denoiser", self.traj_save_path, "Save Trajectories"
        )
        if clicked:
            self.trajectory_handler.save(self.traj_save_path)
            self.traj_save_path = self._get_next_save_path()
        psim.SameLine()
        psim.SetNextItemWidth(100)
        clicked, export_stride_idx = psim.SliderInt(
            "export_stride##denoiser",
            ALLOWED_EXPORT_STRIDES_MAP[self.config.export_stride],
            v_min=0,
            v_max=len(ALLOWED_EXPORT_STRIDES) - 1,
            format=f"{self.config.export_stride}",
        )
        if clicked:
            self.config.export_stride = ALLOWED_EXPORT_STRIDES_INVMAP[export_stride_idx]

        _, self.append_mode = psim.Checkbox(
            "Append Load Mode##denoiser", self.append_mode
        )

        # ========================
        # Brep
        # ========================

        psim.SeparatorText("Brep (Post-processing)")

        # NB: only possible if it the first step!
        if self.i_t == 0:
            if psim.Button("Extract Brep (Auto)##denoiser"):
                uv_grids, _ = self.uv_grid_widget.get_current()
                # Try with different min_extension_len values
                for min_ext_len in [0.0, 0.2]:
                    # Try decreasing values
                    for ext_len in [1.0, 0.5, 1.5]:
                        if self._try_brep_conversion(
                            uv_grids, ext_len, min_ext_len, self.brep_psr_occ_thresh
                        ):
                            break
                    else:
                        continue  # Continue to next min_extension_len if no ext_len worked
                    break  # Break out if we found a working combination

            psim.SameLine()
            psim.Text(f"{self.current_pp_status}")
            psim.SameLine()
            psim.Text(
                ", "
                + (
                    "NO BREP"
                    if self.current_trajectory.brep_solid[self.i_batch] is None
                    else "HAS BREP"
                )
            )

            # If all is enabled, all breps will be saved
            if self.current_pp_output is not None:
                if self.brep_try_all:
                    clicked, self.brep_save_path = save_popup(
                        "brep_denoiser", self.brep_save_path, "Save All"
                    )
                    if clicked:
                        self.save_all_breps()
                else:
                    clicked, self.brep_save_path = save_popup(
                        "brep_denoiser", self.brep_save_path, "Save Geometry"
                    )
                    if clicked:
                        self.save_geometry()

            if psim.TreeNode("Advanced Parameters##brep_denoiser"):
                if psim.Button("Manual Brep##denoiser"):
                    if self.brep_try_all:
                        self._compute_all_breps()
                    else:
                        self._compute_current_brep()

                psim.SameLine()
                _, self.brep_verbose = psim.Checkbox("Verbose##brep", self.brep_verbose)
                psim.SameLine()
                _, self.brep_try_all = psim.Checkbox("All##brep", self.brep_try_all)

                # Add controls for BRep parameters
                # psim.SameLine()
                clicked, grid_res_idx = psim.SliderInt(
                    "Grid Res##brep",
                    0 if self.brep_grid_res == 64 else 1,
                    v_min=0,
                    v_max=1,
                    format="Low(64)" if self.brep_grid_res == 64 else "High(256)",
                )
                if clicked:
                    self.brep_grid_res = 64 if grid_res_idx == 0 else 256

                psim.SameLine()
                clicked, self.brep_smooth_extension = psim.Checkbox(
                    "Smooth Extension##brep", self.brep_smooth_extension
                )

                # psim.SameLine()
                clicked, self.brep_extension_len = psim.SliderFloat(
                    "Extension Length##brep",
                    self.brep_extension_len,
                    v_min=0.1,
                    v_max=1.5,
                    format="%.1f",
                )

                # psim.SameLine()
                clicked, self.brep_min_extension_len = psim.SliderFloat(
                    "Min Extension Length##brep",
                    self.brep_min_extension_len,
                    v_min=0.0,
                    v_max=0.5,
                    format="%.1f",
                )

                # Add PSR occupancy threshold slider near other BRep parameters
                clicked, self.brep_psr_occ_thresh = psim.SliderFloat(
                    "PSR Occ Threshold##brep",
                    self.brep_psr_occ_thresh,
                    v_min=0.3,
                    v_max=0.7,
                    format="%.2f",
                )

                psim.TreePop()

        else:
            # Disable if it isn't the first state
            if self.ps_reconstructed_brep is not None:
                self.ps_reconstructed_brep.set_enabled(False)

        # ========================
        # Trajectory Handler
        # ========================

        psim.SeparatorText("Trajectories")

        # If trajectory handler was updated, reset the current i_step and render the corresponding uv_grid
        if self.trajectory_handler.gui(self.model):
            # TODO: that's a partial fix, this shouldn't be handled like that...
            t = (
                self.current_trajectory.keymap[self.i_t]
                if ((self.i_t >= 0) and (self.i_t < self.current_trajectory.size))
                else self.current_trajectory.keymap[0]
            )
            assert (self.i_t >= 0) and (self.i_t < self.current_trajectory.size)
            # TODO: update state handler here!
            print("WARNING: this won't get the proper scheduler!")
            self.update_uv_grids(t)

        uv_grids_interpolated = self.trajectory_handler.interpolate_gui(self.model)
        if uv_grids_interpolated is not None:
            new_config = deepcopy(self.config)
            new_config.batch_size = self.config.interpolation.interp_batch_size
            self.init_trajectory(
                uv_grids_interpolated,
                self.current_trajectory.frozen_prims,
                config=new_config,
                trajectory_type=TrajectoryType.BACKWARD,
                t=self.config.interpolation.interp_t,
                i_step=self.config.interpolation.interp_t,
            )

        # ========================
        # Parameters
        # ========================

        psim.SeparatorText("Sampling")

        psim.SetNextItemWidth(SLIDER_HALF_WIDTH)
        clicked, scheduler_type_idx = psim.SliderInt(
            "scheduler##denoiser",
            SCHEDULER_TYPE_MAP[self.config.scheduler_type],
            v_min=0,
            v_max=len(SchedulerType) - 1,
            format=f"{self.config.scheduler_type.value}",
        )

        if clicked:
            self.config.scheduler_type = SCHEDULER_TYPE_INVMAP[scheduler_type_idx]

        psim.SameLine()
        update = False
        psim.SetNextItemWidth(SLIDER_HALF_WIDTH)
        clicked, self.config.seed = psim.InputInt(
            "seed##denoiser", self.config.seed, step=1
        )
        update |= clicked

        psim.SetNextItemWidth(SLIDER_HALF_WIDTH)
        clicked, self.config.n_prims = psim.SliderInt(
            "n_primitives##denoiser",
            self.config.n_prims,
            v_min=0,
            v_max=self.zooc_config.max_n_prims,
        )
        update |= clicked
        # clicked, self.config.cfg_scale = psim.SliderFloat(
        #     "Cfg Scale##denoiser",
        #     self.config.cfg_scale,
        #     v_min=0,
        #     v_max=5.0,
        # )
        # update |= clicked
        if update:
            self.init_trajectory(
                config=self.config,
                replace_current=self.current_trajectory.size == 1,
            )

        # DEPRECATED!
        # psim.SetNextItemWidth(SLIDER_HALF_WIDTH)
        # clicked, self.config.renoise_steps = psim.InputInt(
        #     "n_renoise##denoiser",
        #     self.config.renoise_steps,
        #     step=1,
        #     step_fast=5,
        # )
        # # Clamp the value to valid range
        # self.config.renoise_steps = max(
        #     0, min(self.config.renoise_steps, self.config.n_inference_steps - 1)
        # )
        # psim.SetNextItemWidth(SLIDER_HALF_WIDTH)
        # _, self.config.noise_t_threshold = psim.SliderInt(
        #     "noise_t_thresh##denoiser",
        #     self.config.noise_t_threshold,
        #     v_min=0,
        #     v_max=self.config.n_training_steps - 1,
        # )
        # psim.SameLine()

        psim.SetNextItemWidth(SLIDER_HALF_WIDTH)
        # NB: changing batch size is only allowed on size one trajectory!
        psim.SameLine()
        clicked, batch_size_idx = psim.SliderInt(
            "batch_size##denoiser",
            ALLOWED_BATCH_SIZES_MAP[self.config.batch_size],
            v_min=0,
            v_max=len(ALLOWED_BATCH_SIZES) - 1,
            format=f"{self.config.batch_size}",
        )
        if clicked:
            new_batch_size = ALLOWED_BATCH_SIZES_INVMAP[batch_size_idx]
            # When batch size changes
            self._update_batch_size(new_batch_size)

        if psim.TreeNode("Inpainting##denoiser"):
            psim.PushItemWidth(SLIDER_HALF_WIDTH)
            _, self.config.inpainting.n_inpainting_steps = psim.SliderInt(
                "n_inpainting##denoiser",
                self.config.inpainting.n_inpainting_steps,
                v_min=1,
                v_max=self.zooc_config.diffusion.training_timesteps,
            )
            psim.SameLine()
            _, self.config.inpainting.override_batch_size = psim.Checkbox(
                "batch_size##denoiser_inpainting",
                self.config.inpainting.override_batch_size,
            )

            if self.config.inpainting.override_batch_size:
                clicked, self.config.inpainting.empty_min = psim.SliderInt(
                    "min##denoiser_inpainting",
                    self.config.inpainting.empty_min,
                    v_min=4,
                    v_max=self.zooc_config.max_n_prims,
                )
                if clicked:
                    # Make sure it's always inferior to the upper bound
                    self.config.inpainting.empty_min = min(
                        self.config.inpainting.empty_min,
                        self.config.inpainting.empty_max,
                    )
                psim.SameLine()
                clicked, self.config.inpainting.empty_max = psim.SliderInt(
                    "max##denoiser_inpainting",
                    self.config.inpainting.empty_max,
                    v_min=4,
                    v_max=self.zooc_config.max_n_prims,
                )
                if clicked:
                    # Make sure it's always inferior to the upper bound
                    self.config.inpainting.empty_max = max(
                        self.config.inpainting.empty_min,
                        self.config.inpainting.empty_max,
                    )
                psim.SameLine()
                _, self.config.inpainting.empty_batch_multiplier = psim.SliderInt(
                    "mul##denoiser_inpainting",
                    self.config.inpainting.empty_batch_multiplier,
                    v_min=1,
                    v_max=32,
                )

            psim.PopItemWidth()

            # Resampling

            clicked, self.config.resampling.enabled = psim.Checkbox(
                "Enable Resampling", self.config.resampling.enabled
            )

            if self.config.resampling.enabled:
                psim.SetNextItemWidth(150)
                _, self.config.resampling.jump_length = psim.InputInt(
                    "Jump Length##resampling",
                    self.config.resampling.jump_length,
                    step=1,
                )

                psim.SetNextItemWidth(150)
                _, self.config.resampling.n_repetitions = psim.InputInt(
                    "Jump repetitions##resampling",
                    self.config.resampling.n_repetitions,
                    step=1,
                )

                psim.SetNextItemWidth(150)
                _, self.config.resampling.start_t = psim.InputInt(
                    "Start Timestep##resampling", self.config.resampling.start_t, step=1
                )

            psim.TreePop()

        # ========================
        # Editing
        # ========================

        psim.SeparatorText("UV Grid")

        _, _, frozen_prims_modified = self.uv_grid_widget.gui()
        # TODO: dirty state handling!
        if frozen_prims_modified:
            uv_grids, frozen_prims = self.uv_grid_widget.get_current()
            self.init_trajectory(
                uv_grids=uv_grids,
                frozen_prims=frozen_prims,
                trajectory_type=(
                    TrajectoryType.FORWARD
                    if self.i_step == 0
                    else TrajectoryType.BACKWARD
                ),
                config=self.config,
                replace_current=self.current_trajectory.size == 1,
                i_batch=self.i_batch,
            )

        psim.EndDisabled()

    # =====================================
    # Display
    # =====================================

    def update_uv_grids(self, t: int = None) -> None:
        if self.headless_mode:
            return
        if t is None:
            t = self.current_trajectory.keymap[self.i_t]
        if self.current_trajectory.has(t):
            self.uv_grid_widget.set_uv_grids(
                self.current_trajectory.traj_uv_grids[t].extract(self.i_batch),
                self.render_mode,
            )
        if (
            t == 0
            and (self.current_trajectory.brep_mesh_vertices[self.i_batch] is not None)
            and (self.current_trajectory.brep_mesh_faces[self.i_batch] is not None)
        ):
            self.ps_reconstructed_brep = ps.register_surface_mesh(
                RECONSTRUCTED_BREP_MESH,
                self.current_trajectory.brep_mesh_vertices[self.i_batch],
                self.current_trajectory.brep_mesh_faces[self.i_batch],
                enabled=self.render_brep,
            )
        else:
            # self.render_brep = False
            if ps.has_surface_mesh(RECONSTRUCTED_BREP_MESH):
                ps.get_surface_mesh(RECONSTRUCTED_BREP_MESH).set_enabled(False)

    # =====================================
    # Save/Load utils
    # =====================================

    def ps_drop_callback(self, input: str) -> None:
        extension = os.path.splitext(input)[1]
        if extension == ".npz":
            try:
                uv_grids = UvGrid.load_from_npz_data(
                    np.load(input), max_prims=self.zooc_config.max_n_prims
                )
                if self.append_mode and self.i_t == 0:
                    self.uv_grid_widget.append_uv_grids(uv_grids)
                    uv_grids, _ = self.uv_grid_widget.get_current()
                    self.init_trajectory(
                        uv_grids=uv_grids,
                        trajectory_type=TrajectoryType.FORWARD,
                        config=self.config,
                    )
                else:
                    self.init_trajectory(
                        uv_grids=uv_grids,
                        trajectory_type=TrajectoryType.FORWARD,
                        config=self.config,
                    )

            except Exception as e:
                # handle the exception
                print("Could not import from:", input)
                print("Error:\n", e)
        elif extension == ".step":
            try:
                from OCC.Extend.DataExchange import read_step_file

                solid = read_step_file(input)
                uv_grids = brep_to_uvgrid(solid)

                if self.append_mode and self.i_t == 0:
                    self.uv_grid_widget.append_uv_grids(uv_grids)
                    uv_grids, _ = self.uv_grid_widget.get_current()
                else:
                    self.init_trajectory(
                        uv_grids=uv_grids,
                        trajectory_type=TrajectoryType.FORWARD,
                        config=self.config,
                    )

            except Exception as e:
                # handle the exception
                print("Could not import from:", input)
                print("Error:\n", e)
        elif extension == ".traj":
            # try:
            import deepdish as dd

            data = dd.io.load(input)
            self.trajectory_handler.deserialize(data, model=self.model)
            # Display!
            self.update_uv_grids()
            # except Exception as e:
            #     # handle the exception
            #     print("Could not import from:", input)
            #     print("Error:\n", e)
        else:
            print("Only .npz, .step or .traj files are accepted!")

    def _get_next_save_path(self):
        os.makedirs(TRAJ_SAVE_FOLDER, exist_ok=True)
        all_exported_paths = glob.glob(
            os.path.join(TRAJ_SAVE_FOLDER, "exported_*.traj")
        )
        return os.path.join(
            TRAJ_SAVE_FOLDER, f"exported_{len(all_exported_paths):06d}.traj"
        )

    def _get_next_brep_save_path(self):
        os.makedirs(BREP_SAVE_FOLDER, exist_ok=True)
        all_exported_paths = glob.glob(
            os.path.join(BREP_SAVE_FOLDER, "exported_*.step")
        )
        return os.path.join(
            BREP_SAVE_FOLDER, f"exported_{len(all_exported_paths):06d}.step"
        )

    def _try_brep_conversion(
        self, uv_grids, extension_len, min_extension_len, psr_occ_thresh
    ):
        """Helper method to attempt brep conversion with given parameters"""
        try:
            pp_viewer_output: PpViewerOutput = uvgrid_to_brep_or_mesh(
                uvgrid=uv_grids,
                grid_res=self.brep_grid_res,
                smooth_extension=self.brep_smooth_extension,
                uvgrid_extension_len=extension_len,
                min_extension_len=min_extension_len,
                psr_occ_thresh=psr_occ_thresh,
                verbose=self.brep_verbose,
            )
            self.current_pp_status = pp_viewer_output.state.value
            self.current_pp_output = pp_viewer_output  # Store the output

            if pp_viewer_output.state == PpSuccessState.FAILURE:
                raise RuntimeError()
            if (
                pp_viewer_output.state == PpSuccessState.NON_WATERTIGHT
                or pp_viewer_output.state == PpSuccessState.SUCCESS
            ):
                brep = pp_viewer_output.brep_occ
                mesh: trimesh.Trimesh = brep_to_mesh(brep)
                assert isinstance(mesh.vertices, np.ndarray) and isinstance(
                    mesh.faces, np.ndarray
                )
                self.current_trajectory.brep_solid[self.i_batch] = brep
                self.current_trajectory.brep_mesh_vertices[self.i_batch] = mesh.vertices
                self.current_trajectory.brep_mesh_faces[self.i_batch] = mesh.faces
                self.ps_reconstructed_brep = ps.register_surface_mesh(
                    RECONSTRUCTED_BREP_MESH,
                    self.current_trajectory.brep_mesh_vertices[self.i_batch],
                    self.current_trajectory.brep_mesh_faces[self.i_batch],
                    enabled=True,
                )
                self.render_brep = True
                self.current_brep = brep
                self.brep_extension_len = extension_len
                self.brep_min_extension_len = min_extension_len
                print(
                    f"Success with extension_len={extension_len}, min_extension_len={min_extension_len}"
                )
                return True
            elif (
                pp_viewer_output == PpSuccessState.PATCH_ONLY
                or PpSuccessState.PATCH_WITH_EDGES
            ):
                mesh: trimesh.Trimesh = trimesh.util.concatenate(
                    pp_viewer_output.patches
                )
                assert isinstance(mesh.vertices, np.ndarray) and isinstance(
                    mesh.faces, np.ndarray
                )
                self.current_trajectory.brep_mesh_vertices[self.i_batch] = mesh.vertices
                self.current_trajectory.brep_mesh_faces[self.i_batch] = mesh.faces
                self.ps_reconstructed_brep = ps.register_surface_mesh(
                    RECONSTRUCTED_BREP_MESH,
                    self.current_trajectory.brep_mesh_vertices[self.i_batch],
                    self.current_trajectory.brep_mesh_faces[self.i_batch],
                    enabled=True,
                )
                self.render_brep = True
                self.current_brep = None
                self.brep_extension_len = extension_len
                self.brep_min_extension_len = min_extension_len

            return False
        except Exception as e:
            print(
                f"Failed with extension_len={extension_len}, min_extension_len={min_extension_len}"
            )
            print("Error:\n", e)
            return False

    def _compute_current_brep(self):
        try:
            uv_grids, _ = self.uv_grid_widget.get_current()
            self._try_brep_conversion(
                uv_grids,
                self.brep_extension_len,
                self.brep_min_extension_len,
                self.brep_psr_occ_thresh,
            )
        except Exception as e:
            # handle the exception
            print("Could not reconstruct brep")
            print("Error:\n", e)
        else:
            # TODO: add feedback directly in the UI
            print("...")

    def _compute_all_breps(self):
        current_i_batch = self.i_batch
        for i_batch in tqdm(range(self.current_trajectory.batch_size)):
            self.set_i_batch(i_batch)
            self.update_uv_grids()
            self._compute_current_brep()
        self.set_i_batch(current_i_batch)
        self.update_uv_grids()

    def _save_patches_and_edges(self, save_path: str):
        """Save patches and edges to NPZ format compatible with render_step.py"""
        if self.current_pp_output is None:
            print("No postprocessing output to save")
            return False

        # Concatenate all patch vertices and faces into a single mesh
        if self.current_pp_output.patches:
            all_vertices = []
            all_faces = []
            vertex_offset = 0

            for patch in self.current_pp_output.patches:
                all_vertices.append(patch.vertices)
                all_faces.append(patch.faces + vertex_offset)
                vertex_offset += len(patch.vertices)

            vertices = np.concatenate(all_vertices, axis=0)
            triangles = np.concatenate(all_faces, axis=0)
        else:
            vertices = np.array([])
            triangles = np.array([])

        # Convert edges to the format expected by render_step.py
        edges = []
        if (
            hasattr(self.current_pp_output, "edges")
            and self.current_pp_output.edges is not None
        ):
            for edge in self.current_pp_output.edges:
                if len(edge) > 0:
                    edges.append(edge)

        # Normalize vertices and edges to [-1, 1]
        if len(vertices) > 0:
            # Calculate bounds for both vertices and edges
            v_min = vertices.min(axis=0, keepdims=True)
            v_max = vertices.max(axis=0, keepdims=True)

            # Include edges in bounds calculation
            if edges:
                edges_array = np.concatenate([np.array(edge) for edge in edges], axis=0)
                v_min = np.minimum(v_min, edges_array.min(axis=0, keepdims=True))
                v_max = np.maximum(v_max, edges_array.max(axis=0, keepdims=True))

            # Calculate center and scale
            center = (v_max + v_min) / 2
            scale = np.max(v_max - v_min)
            scale = max(scale, 1e-6)  # Prevent division by zero

            # Normalize vertices
            vertices = (vertices - center) / scale

            # Move to ground plane at z=-0.5
            z_min = vertices[:, 2].min()
            vertices[:, 2] -= z_min + 0.5

            # Normalize edges
            normalized_edges = []
            for edge in edges:
                edge = np.array(edge)
                edge = (edge - center) / scale
                edge[:, 2] -= z_min + 0.5
                normalized_edges.append(edge)
            edges = normalized_edges

        # Save in the format expected by render_step.py
        np.savez(
            save_path,
            vertices=vertices,  # Combined vertices from all patches
            triangles=triangles,  # Combined faces from all patches
            edges=edges,  # List of edge point arrays
            state=self.current_pp_output.state.value,
        )
        return True

    def save_all_breps(self):
        """Save all precomputed breps in the current trajectory and adds _b000.step to the input step file."""
        base_path = self.brep_save_path.replace(".step", "")
        folder_path = os.path.dirname(self.brep_save_path)
        os.makedirs(folder_path, exist_ok=True)

        all_saved = []
        for i_batch in range(self.current_trajectory.batch_size):
            brep = self.current_trajectory.brep_solid[i_batch]
            if brep is not None:
                try:
                    brep_path = f"{base_path}_b{i_batch:03d}.step"
                    write_step_file(brep, brep_path)
                    print(f"Saved BREP to {brep_path}")

                    # Save corresponding UV grid
                    uvgrid_save_path = f"{base_path}_b{i_batch:03d}_uvgrid.npz"
                    uv_grids, _ = self.uv_grid_widget.get_current()
                    uv_grids.export_npz(uvgrid_save_path)
                    print(f"Saved UV grid to {uvgrid_save_path}")

                    all_saved.append(i_batch)
                except Exception as e:
                    print("Failed to save BREP:", e)

        print(f"Successfully saved breps for batch indices: {all_saved}")

    def save_geometry(self):
        """Save geometry based on current state"""
        if self.current_pp_output is None:
            print("No geometry to save")
            return

        base_path = self.brep_save_path.replace(".step", "")

        if self.current_pp_output.state in [
            PpSuccessState.SUCCESS,
            PpSuccessState.NON_WATERTIGHT,
        ]:
            # Save BREP
            if self.current_brep is not None:
                try:
                    write_step_file(self.current_brep, self.brep_save_path)
                    print(f"Saved BREP to {self.brep_save_path}")

                    # Save corresponding UV grid
                    uvgrid_save_path = base_path + "_uvgrid.npz"
                    uv_grids, _ = self.uv_grid_widget.get_current()
                    uv_grids.export_npz(uvgrid_save_path)
                    print(f"Saved UV grid to {uvgrid_save_path}")

                    self.brep_save_path = self._get_next_brep_save_path()
                    return True
                except Exception as e:
                    print("Failed to save BREP:", e)
                    return False

        elif self.current_pp_output.state in [
            PpSuccessState.PATCH_ONLY,
            PpSuccessState.PATCH_WITH_EDGES,
        ]:
            # Save patches and edges
            patches_save_path = base_path + "_patches.npz"
            if self._save_patches_and_edges(patches_save_path):
                print(f"Saved patches and edges to {patches_save_path}")
                self.brep_save_path = self._get_next_brep_save_path()
                return True

        return False
