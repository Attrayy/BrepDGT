import PIL.Image
import torch
import wandb
import os
import numpy as np
from PIL import Image, ImageDraw
from collections import defaultdict

from tqdm import tqdm
from typing import List, Tuple, Any, Dict, Union
from dataclasses import dataclass
from brepdgt.models.base_model import BaseModel
from brepdgt.models.backbones.sequence_dm import (
    SEQUENCE_DIFFUSION_BACKBONES,
)
from brepdgt.models.tokenizers import Tokens
from brepdgt.models.uv_vae import UvVae, UvVaeOutput
from brepdgt.diffusion import (
    DIFFUSION_PROCESSES,
    Diffusion,
)
from brepdgt.config import Config

from brepdgt.datasets.abc_dataset import ABCDatasetOutput
from brepdgt.utils.vis import concat_h_pil, concat_v_pil, save_vid_from_img_seq
from brepdgt.primitives.uvgrid import UvGrid, stack_batched_uvgrids
from brepdgt.metrics.pc_metrics import compute_pc_metrics
from brepdgt.utils.vis import save_and_render_uvgrid


@dataclass(frozen=True)
class BRepDGTReconstruction:
    # Float tensor sampled with diffusion
    x: torch.Tensor
    uvgrids: UvGrid
    # ------------------------------
    # (optional) trajectories
    # ------------------------------
    uvgrids_traj: List[UvGrid]


class BRepDGT(BaseModel):
    name = "brepdgt"

    def __init__(self, config: Config, acc_logger):
        super().__init__(config, acc_logger)

        self.token_dim_split = [self.config.data_dim]  # coord
        self.token_dim_split.append(1)  # grid mask

        # Load token Vae
        if self.config.token_vae_ckpt_path == "":
            print("Warning: not using a pretrained token vae")
            self.token_vae = UvVae(self.config, acc_logger)
        else:
            self.config_token_vae, state_dict_token_vae = self._load_vae_config(
                self.config.token_vae_ckpt_path
            )
            if self.config.diffusion.z_conditioning:
                self.config_token_vae.diffusion.z_conditioning = True
            self.token_vae = UvVae(self.config_token_vae, acc_logger)

            self.token_vae.load_state_dict(state_dict_token_vae)
            self.token_vae.requires_grad_(False)
            assert (
                self.config.x_dim == self.config_token_vae.x_dim
            ), f"current x_dim: {self.config.x_dim}, token_vae_x_dim: {self.config_token_vae.x_dim}"

        self.seq_len = self.config.max_n_prims

        # Diffusion backbone
        self.backbone = SEQUENCE_DIFFUSION_BACKBONES[
            self.config.diffusion.model["name"]
        ](
            input_dim=self.config.x_dim,
            seq_length=self.seq_len,
            z_dim=self.config.z_dim,
            n_z=self.config.n_z,
            z_conditioning=self.config.diffusion.z_conditioning,
            **self.config.diffusion.model["options"],
        )

        # Diffusion process
        self.diffusion_process: Diffusion = DIFFUSION_PROCESSES[
            self.config.diffusion.name
        ](self.backbone, self.config)

    def _load_vae_config(self, vae_ckpt_path: str) -> Tuple[Config, Dict]:
        """
        Loads the TokenVae config file
        """
        tmp = torch.load(vae_ckpt_path)["state_dict"]
        state_dict_vae = {}
        for k, v in tmp.items():
            if k.startswith("model."):
                state_dict_vae[k[len("model.") :]] = v

        config_vae_path = os.path.join(
            os.path.dirname(os.path.dirname(vae_ckpt_path)), "config.yaml"
        )
        config_vae = Config.from_yaml(open(config_vae_path, "r"))

        return config_vae, state_dict_vae

    def forward(
        self,
        x: torch.Tensor,
    ):
        raise NotImplementedError()

    def compute_loss(self, batch: ABCDatasetOutput, epoch: int, split="train"):
        device = next(self.parameters()).device

        if batch.uvgrid is not None:
            batch.uvgrid.to_tensor(device)

        # ------------------
        # TOKENIZE
        # ------------------
        token_vae_output: UvVaeOutput = self.token_vae(batch)

        # ------------------
        # DIFFUSION
        # ------------------
        out, target, t = self.diffusion_process(
            token_vae_output.tokens,
            z=token_vae_output.tokens.condition,
            mask=token_vae_output.tokens.mask,
            prefix=split,
            empty_embeddings=None,
            return_timesteps=True,
        )
        loss, log_dict = self.uv_loss(
            out=out,
            target=target,
            empty_mask_gt=batch.uvgrid.empty_mask,
            split=split,
            timesteps=t,
            epoch=epoch,
        )

        log_dict[f"tokens/{split}/x0_rescaled_norm"] = [
            token_vae_output.tokens.sample.norm(dim=-1).mean().tolist()
        ]
        self.acc_logger.log_scalar_and_hist_dict(log_dict)
        return loss

    @torch.no_grad()
    def sample(
        self,
        batch: ABCDatasetOutput,
        # Specifies whether to return full denoising trajectories
        return_traj: bool = False,
        cfg_scale: float = 1.0,
    ) -> BRepDGTReconstruction:
        n_batch = len(batch.name)

        token_vae_output: UvVaeOutput = self.token_vae(batch)
        z_sample = token_vae_output.tokens.condition
        attn_mask = token_vae_output.tokens.mask

        if self.config.sample_mode == "fixed":
            # Distribute different numbers of faces evenly across the batch
            num_faces_list = np.linspace(2, self.seq_len, n_batch).round().astype(int)
            attn_mask = torch.zeros(
                (n_batch, self.seq_len),
                dtype=bool,
                device=token_vae_output.tokens.mask.device,
            )
            for i, n_faces in enumerate(num_faces_list):
                attn_mask[i, n_faces:] = True  # Mask out faces after n_faces

        # -----------------
        # SAMPLE DIFFUSION
        # -----------------

        diff_sample, traj = self.diffusion_process.p_sample_loop(
            batch_size=n_batch,
            return_traj=return_traj,
            z=z_sample,
            mask=attn_mask,
            cfg_scale=cfg_scale,
            traj_stride=self.config.vis.trajectory_stride,
            # Use resampling if it is requested!
            use_resampling=self.config.test_use_resampling,
            resampling_jump_length=self.config.test_resampling_jump_length,
            resampling_repetitions=self.config.test_resampling_repetitions,
            resampling_start_t=self.config.test_resampling_start_t,
        )

        # -------------------------
        # DETOKENIZE & RECONSTRUCT
        # -------------------------
        uvgrids = self.token_vae.detokenizer.decode(
            tokens=Tokens(
                sample=diff_sample.get_x_for_detokenizer(),
                labels=diff_sample.l,
                condition=z_sample,
                mask=attn_mask,
            ),
            max_n_prims=self.seq_len,
        )

        uvgrids_traj = []
        if return_traj:
            for i in tqdm(
                range(0, len(traj)),
                desc="Reconstruction trajectory",
            ):
                x = traj[i].get_x_for_detokenizer()
                x = x[: self.config.vis.max_trajectory_points]
                if traj[i].l is not None:
                    labels = traj[i].l[: self.config.vis.max_trajectory_points]
                else:
                    labels = None
                if traj[i].z is not None:
                    z = traj[i].z[: self.config.vis.max_trajectory_points]
                else:
                    z = None
                if traj[i].mask is not None:
                    mask = traj[i].mask[: self.config.vis.max_trajectory_points]
                else:
                    mask = None

                tokens = Tokens(
                    sample=x,
                    labels=labels,
                    condition=z,
                    mask=mask,
                )
                uvgrids_traj_i = self.token_vae.detokenizer.decode(
                    tokens=tokens, max_n_prims=self.seq_len
                )
                uvgrids_traj.append(uvgrids_traj_i)

        return BRepDGTReconstruction(
            x=diff_sample.x,
            uvgrids=uvgrids,
            uvgrids_traj=uvgrids_traj,
        )

    def vis(
        self,
        batch: ABCDatasetOutput,
        batch_idx: int,
        step: int,
        split: str,
        vis_traj: bool = True,
        render_blender: bool = True,
        render_gt: bool = True,
        log_wandb: bool = True,
    ):
        """
        Save uvgrid and visualize reconstructions
        """
        training = self.training
        self.eval()

        # Get local batch size and rank info
        n_batch = len(batch.name)
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        # Directories
        gt_dir = os.path.join(self.config.log_dir, "vis", split, "gt")
        os.makedirs(gt_dir, mode=0o777, exist_ok=True)

        step_dir = os.path.join(
            self.config.log_dir, "vis", split, f"step-{str(step).zfill(9)}"
        )
        if split == "test":
            # to distinguish with weighted version
            step_dir = os.path.join(step_dir, self.config.dataset)
            if self.config.test_use_resampling:
                step_dir += "-resample"
        total_dir = os.path.join(step_dir, "total")
        os.makedirs(total_dir, mode=0o777, exist_ok=True)

        # Generate samples
        samples, uvgrids_samples, sample_dirs = [], [], []
        for cfg_scale in self.config.diffusion.cfg_scales:
            sample: BRepDGTReconstruction = self.sample(
                batch, return_traj=vis_traj, cfg_scale=cfg_scale
            )
            samples.append(sample)
            uvgrids_sample: UvGrid = sample.uvgrids
            uvgrids_sample.grid_mask = uvgrids_sample.grid_mask > 0
            uvgrids_samples.append(uvgrids_sample)

            sample_dir = os.path.join(step_dir, f"cfg_{cfg_scale:.2f}", "uvgrid")
            os.makedirs(sample_dir, mode=0o777, exist_ok=True)
            sample_dirs.append(sample_dir)

        uvgrids_gt = batch.uvgrid
        names = batch.name

        # select examples to visualize
        vis_config = self.config.vis
        n_examples = vis_config.n_examples

        if split == "test":
            # visualize and save all when testing
            vis_idxs = torch.arange(0, n_batch)
        else:
            if len(vis_config.vis_idxs) != 0:
                if vis_config.vis_idxs == "all":
                    vis_idxs = torch.arange(0, n_batch)
                else:
                    # Adjust vis_idxs for distributed training
                    global_vis_idxs = torch.tensor(vis_config.vis_idxs)
                    # Calculate which indices belong to this GPU
                    local_vis_idxs = []
                    for idx in global_vis_idxs:
                        # Calculate which GPU this index belongs to
                        target_rank = (idx // n_batch) % world_size
                        if target_rank == rank:
                            # Convert to local index
                            local_idx = idx % n_batch
                            if local_idx < n_batch:  # Make sure index is valid
                                local_vis_idxs.append(local_idx)
                    vis_idxs = torch.tensor(local_vis_idxs)
            else:
                if n_examples is None:
                    # visualize all local samples
                    vis_idxs = torch.arange(0, n_batch)
                else:
                    # Distribute n_examples across GPUs
                    n_examples_per_gpu = max(1, n_examples // world_size)
                    n_examples_local = min(n_examples_per_gpu, n_batch)
                    vis_idxs = torch.linspace(0, n_batch - 1, n_examples_local).int()

        torch.cuda.empty_cache()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Create a list to store all images and their metadata for this rank
        rank_images = []

        try:
            for vis_idx in tqdm(vis_idxs, desc="rendering and saving"):
                img_list = []

                gt_name = names[vis_idx]
                if render_gt and (uvgrids_gt is not None):
                    img_gt = save_and_render_uvgrid(
                        save_dir=gt_dir,
                        save_name=gt_name,
                        uvgrids=uvgrids_gt,
                        vis_idx=vis_idx,
                        render_blender=render_blender,
                        use_cached_if_exists=True,
                    )
                    img_list.append(img_gt)

                if split == "train":
                    batch_size = self.config.batch_size
                elif split == "val":
                    batch_size = self.config.val_batch_size
                elif split == "test":
                    batch_size = self.config.test_batch_size
                else:
                    raise ValueError(f"{split} not allowed")
                vis_global_idx = vis_idx + batch_size * (batch_idx * world_size + rank)
                vis_global_idx = vis_global_idx.cpu().item()

                try:
                    for cfg_idx, cfg_scale in enumerate(
                        self.config.diffusion.cfg_scales
                    ):
                        img_sample = save_and_render_uvgrid(
                            save_dir=sample_dirs[cfg_idx],
                            save_name=str(vis_global_idx).zfill(5),
                            uvgrids=uvgrids_samples[cfg_idx],
                            vis_idx=vis_idx,
                            render_blender=render_blender,
                        )
                        img_list.append(img_sample)
                except KeyboardInterrupt:
                    print("\nInterrupted. Cleaning up...")
                    raise
                except Exception as e:
                    print(f"Error in vis at batch {batch_idx}, index {vis_idx}: {e}")
                    continue

                if render_blender and log_wandb:
                    img = concat_v_pil(img_list)
                    img_path = os.path.join(total_dir, names[vis_idx] + ".png")
                    img.save(img_path)
                    rank_images.append(
                        {
                            "path": img_path,
                            "caption": f"{gt_name}",
                            "tag": f"imgs/{split}/{gt_name}",
                        }
                    )

        except KeyboardInterrupt:
            print("\nVisualization interrupted by user")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()  # Ensure all processes get the interrupt
            raise
        finally:
            self.train(training)  # Ensure model is returned to training state if needed

        # Synchronize all processes before gathering images
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Gather images from all ranks and log them
        if torch.distributed.is_initialized():
            # Create a list to store image metadata from all ranks
            gathered_images = [None] * world_size
            torch.distributed.all_gather_object(gathered_images, rank_images)

            # Log all images to wandb (all ranks will have the data but only rank 0 will log)
            if rank == 0 and log_wandb:
                for rank_data in gathered_images:
                    for img_data in rank_data:
                        img = wandb.Image(img_data["path"], caption=img_data["caption"])
                        self.acc_logger.log_imgs(img_data["tag"], [img])
        else:
            # Non-distributed case - log directly
            if log_wandb:
                for img_data in rank_images:
                    img = wandb.Image(img_data["path"], caption=img_data["caption"])
                    self.acc_logger.log_imgs(img_data["tag"], [img])

        # Similar modification for trajectory visualization
        if vis_traj:
            print("Visualizing trajectories")
            n_traj_points = self.config.vis.max_trajectory_points
            traj_points_per_gpu = max(1, n_traj_points // world_size)

            rank_videos = []

            for vis_idx in range(traj_points_per_gpu):
                vis_traj_dir = os.path.join(
                    self.config.log_dir,
                    "vis",
                    "traj",
                    f"step-{str(step).zfill(9)}",
                    str(vis_idx).zfill(4),
                )
                os.makedirs(vis_traj_dir, mode=0o777, exist_ok=True)
                img_seq = []

                # visualize trajectory only for the first cfg_scale
                for t, uvgrid_traj in enumerate(samples[0].uvgrids_traj):
                    uvgrid_traj.grid_mask = uvgrid_traj.grid_mask > 0
                    traj_t = t * self.config.vis.trajectory_stride
                    try:
                        render_objects = ["coord"]
                        img = save_and_render_uvgrid(
                            save_dir=vis_traj_dir,
                            save_name=f"{rank}_{str(traj_t).zfill(4)}",
                            uvgrids=uvgrid_traj,
                            vis_idx=vis_idx,
                            render_objects=render_objects,  # render only pc
                            render_blender=render_blender,
                        )
                    except Exception as e:
                        print(f"Error in vis at batch {vis_idx}: {e}")
                        continue
                    d = ImageDraw.Draw(img)
                    d.text(
                        (10, 10),
                        f"{str(traj_t).zfill(4)}",
                        fill=(0, 0, 0, 255),
                    )
                    img_seq.append(img)

                # add last image
                for i in range(10):
                    img_seq.append(img_seq[-1])

                if render_blender and log_wandb:
                    vid_path = os.path.join(vis_traj_dir, f"vid_{rank}.mp4")
                    save_vid_from_img_seq(vid_path, img_seq)
                    rank_videos.append(
                        {
                            "path": vid_path,
                            "tag": f"traj/{str(vis_idx).zfill(4)}_{rank}",
                        }
                    )

            # Synchronize and gather videos from all ranks
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                gathered_videos = [None] * world_size
                torch.distributed.all_gather_object(gathered_videos, rank_videos)

                if rank == 0 and log_wandb:
                    for rank_data in gathered_videos:
                        for vid_data in rank_data:
                            self.acc_logger.log_vid(vid_data["tag"], vid_data["path"])
            else:
                if log_wandb:
                    for vid_data in rank_videos:
                        self.acc_logger.log_vid(vid_data["tag"], vid_data["path"])

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.train(training)

    def test(
        self,
        batch: ABCDatasetOutput,
        global_step: int,
        batch_idx: int,
    ):
        """
        Tests the model on the given batch and saves results.

        Args:
        - batch: The batch of data to test on.
        - global_step: The current global step of the model.
        - batch_idx: The index of the current batch.
        - save: Whether to save the outputs or not.

        Returns:
        - A dictionary with the real, reconstructed, and generated zone graphs and point clouds.
        """
        print(f"Sampling test batch {batch_idx}...")
        device = next(self.parameters()).device
        training = self.training
        self.eval()

        n_batch = len(batch.name)
        labels = batch.uvgrid.prim_type
        sample: BRepDGTReconstruction = self.sample(batch)
        uvgrids_pred = sample.uvgrids
        uvgrids_gt = batch.uvgrid

        self.train(training)

        return {
            "uvgrid_gts": uvgrids_gt,
            "uvgrid_preds": uvgrids_pred,
        }

    def compute_metrics(self, outputs, split):
        """
        Computes various metrics for the model outputs.

        Args:
        - outputs: The outputs from the model. (test_step)
        - split: The dataset split (e.g., 'val', 'test').

        Returns:
        - A dictionary containing the computed metrics.
        """
        print("Computing metrics...")
        device = next(self.parameters()).device

        uvgrid_gts = stack_batched_uvgrids([output["uvgrid_gts"] for output in outputs])
        coord_gts = uvgrid_gts.sample_pts(self.config.test_num_pts).to(device)[
            : self.config.n_pc_metric_samples
        ]  # B x N x 3
        uvgrid_preds = stack_batched_uvgrids(
            [output["uvgrid_preds"] for output in outputs]
        )
        uvgrid_preds.grid_mask = uvgrid_preds.grid_mask > 0
        coord_preds = uvgrid_preds.sample_pts(self.config.test_num_pts).to(device)[
            : self.config.n_pc_metric_samples
        ]  # B x N x 3

        metrics = {}
        with torch.no_grad():
            # -------------------------------------
            # Point cloud metrics (generation)
            # -------------------------------------
            print("Computing point cloud metrics...")
            gen_pc_metrics = compute_pc_metrics(
                coord_preds, coord_gts, normalize="longest_axis"
            )

            # Update metrics with point cloud metrics
            metrics.update(
                {
                    **{f"test/gen/{k}": v for k, v in gen_pc_metrics.items()},
                }
            )

        for k, v in metrics.items():
            if isinstance(metrics[k], torch.Tensor):
                metrics[k] = v.cpu().item()
            print(f"  {k}: {metrics[k]}")
            metrics[k] = [metrics[k]]
        print("Computing metrics finished!")

        return metrics

    def uv_loss(
        self,
        out: torch.Tensor,  # [B, n_prims, x_dim]
        target: torch.Tensor,  # [B, n_prims, x_dim]
        empty_mask_gt: torch.Tensor,  # [B, n_prims]
        split: str,
        timesteps: torch.Tensor = None,  # [B]
        epoch: int = None,
    ):
        n_batch = out.shape[0]
        n_prims = self.config.max_n_prims
        n_grid = self.config.n_grid

        # Use pre-calculated SNR and loss weights from diffusion process
        # snr = self.diffusion_process.snr[timesteps]  # [B]
        loss_weights = self.diffusion_process.loss_weight[timesteps]  # [B]
        # Reshape to match loss dimensions
        loss_weights = loss_weights.view(-1, 1).expand(-1, n_prims)  # [B, n_prims]

        # Store unweighted losses for logging
        unweighted_losses = {}

        # Reshape tensors for grid operations
        out = out.view(
            n_batch, n_prims, n_grid, n_grid, -1
        )  # [B, n_prims, n_grid, n_grid, C]
        target = target.view(
            n_batch, n_prims, n_grid, n_grid, -1
        )  # [B, n_prims, n_grid, n_grid, C]
        coord_out, grid_mask_out = torch.split(
            out, self.token_dim_split, dim=-1
        )  # [B, n_prims, n_grid, n_grid, 3], [..., 1], [..., 3]
        coord_target, grid_mask_target = torch.split(
            target, self.token_dim_split, dim=-1
        )  # Same shapes as above

        # Coordinate loss
        loss_coord = torch.mean(
            (coord_out - coord_target) ** 2, dim=-1
        )  # [B, n_prims, n_grid, n_grid]
        loss_coord = torch.mean(loss_coord, dim=[-1, -2])  # [B, n_prims]
        unweighted_losses["coord"] = loss_coord[~empty_mask_gt].mean()  # scalar
        masked_loss_coord = (
            loss_coord[~empty_mask_gt] * loss_weights[~empty_mask_gt]
        )  # [num_non_masked]
        loss_coord = masked_loss_coord.mean()  # scalar

        loss_grid_mask = torch.mean((grid_mask_out - grid_mask_target) ** 2, dim=-1)
        loss_grid_mask = torch.mean(loss_grid_mask, dim=[-1, -2])
        unweighted_losses["grid_mask"] = loss_grid_mask[~empty_mask_gt].mean()
        masked_loss_grid_mask = (
            loss_grid_mask[~empty_mask_gt] * loss_weights[~empty_mask_gt]
        )
        loss_grid_mask = masked_loss_grid_mask.mean()

        loss = (
            self.config.alpha_coord * loss_coord
            + self.config.alpha_grid_mask * loss_grid_mask
        )

        log_dict = {}
        with torch.no_grad():
            # Log unweighted losses as default
            log_dict[f"loss/{split}/total"] = [loss.tolist()]
            log_dict[f"loss/{split}/coord"] = [unweighted_losses["coord"].tolist()]
            log_dict[f"loss/{split}/grid_mask"] = [
                unweighted_losses["grid_mask"].tolist()
            ]

        return loss, log_dict
