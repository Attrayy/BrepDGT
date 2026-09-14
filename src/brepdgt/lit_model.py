import lightning.pytorch as pl
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from brepdgt.datasets import DATASETS
from brepdgt.utils.solvers import build_lr_scheduler, build_optimizer
from brepdgt.models import MODELS
from brepdgt.models.base_model import BaseModel
from brepdgt.utils.acc_logger import AccLogger
from brepdgt.config import Config
from brepdgt.utils.common import count_parameters

import torch
import traceback
import gc


class LitModel(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.automatic_optimization = False
        self.config = config
        self.acc_logger = AccLogger()
        self.model: BaseModel = MODELS[config.model](config, self.acc_logger)
        if self.config.compile:
            self.model = torch.compile(self.model)
        self.num_workers = self.config.num_workers
        self.test_step_outputs = []
        self.validation_count = 0

    def optimize(self, loss):
        # take gradient descent
        self.zero_grad()
        loss.backward()

        opt = self.optimizers()
        self.clip_gradients(
            opt,
            gradient_clip_val=self.config.clip_grad["options"]["max_norm"],
            gradient_clip_algorithm=self.config.clip_grad["type"],
        )
        opt.step()
        self.lr_schedulers().step()

    def training_step(self, batch, batch_idx):
        loss = self.model.compute_loss(batch, self.current_epoch)
        self.optimize(loss)
        self.log(
            "train_loss",
            loss.detach().item(),
            on_step=True,
            prog_bar=True,
            logger=False,
            sync_dist=True,
        )

        if self.check_vis_step(self.global_step):
            self.model.vis(
                batch=batch, batch_idx=batch_idx, step=self.global_step, split="train"
            )

        if self.check_summary_step(self.global_step):
            sum_weights = 0.0
            max_weights = 0.0
            n_weights = count_parameters(self)
            for param in self.parameters():
                sum_weights += torch.abs(param.flatten()).sum().item()
                max_weights = max(
                    max_weights, torch.max(torch.abs(param.flatten())).item()
                )
            log_dict = {
                "model/weight_magnitude_avg": [sum_weights / float(n_weights)],
                "model/weight_magnitude_max": [max_weights],
            }
            self.acc_logger.log_scalar_and_hist_dict(log_dict)
            self.log_resources()
            self.acc_logger.write_summary(self.logger, self.global_step)

        self.acc_logger.write_img_summary(self.logger, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model.compute_loss(batch, self.current_epoch, split="val")
        self.log(
            "val_loss",
            loss,
            on_step=True,
            prog_bar=True,
            logger=False,
            batch_size=self.config.val_batch_size,
            sync_dist=True,
        )

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if self.config.vis.vis_batch == "all":
            self.model.vis(
                batch,
                batch_idx=batch_idx,
                step=self.global_step,
                split="val",
                vis_traj=self.config.vis.vis_trajectory,
                render_blender=self.config.vis.render_blender,
            )
        elif batch_idx == 0:
            self.model.vis(
                batch,
                batch_idx=batch_idx,
                step=self.global_step,
                split="val",
                vis_traj=self.config.vis.vis_trajectory,
                render_blender=self.config.vis.render_blender,
            )

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if self.config.test_in_loop and (
            self.validation_count % self.config.test_nth_val == 0
        ):
            # Get batch-wise reconstructions and generations
            test_step_output = self.model.test(
                batch,
                self.global_step,
                batch_idx,
            )
            self.test_step_outputs.append(test_step_output)
            return loss

        return loss

    def on_validation_end(self) -> None:
        self.acc_logger.write_summary(self.logger, self.global_step)
        self.acc_logger.write_img_summary(self.logger, self.global_step)
        self.acc_logger.write_vid_summary(self.logger, self.global_step)

        gc.collect()

    def on_validation_epoch_end(self) -> None:
        if self.config.test_in_loop and (
            self.validation_count % self.config.test_nth_val == 0
        ):
            # Gather batch-wise reconstructions and generations
            # Compute metrics on the whole set
            metrics = self.model.compute_metrics(self.test_step_outputs, "val")
            self.acc_logger.log_scalar_dict(metrics)
            self.log_dict(
                {k: v[0] for k, v in metrics.items()},
                on_epoch=True,
                logger=False,
                sync_dist=True,
            )
            self.test_step_outputs.clear()
            if self.config.monitor == "1_nna":
                self.log("1_nna", metrics[f"test/gen/1-NN-CD-acc"][0], sync_dist=True)
        self.validation_count += 1
        torch.cuda.empty_cache()

    def test_step(self, batch, batch_idx):
        global_step = None
        ckpt_path = self.trainer._checkpoint_connector._ckpt_path
        for ckpt_split in ckpt_path.split("-"):
            ckpt_split_split = ckpt_split.split("_")
            if ckpt_split_split[0] == "step":
                global_step = int(ckpt_split_split[1])
                break
        assert (
            global_step is not None
        ), f"global step should not be none, loaded checkpoint name {ckpt_path}"

        loss = self.model.compute_loss(batch, self.current_epoch, split="test")
        self.log(
            "test_loss",
            loss,
            on_step=True,
            prog_bar=True,
            logger=False,
            batch_size=self.config.test_batch_size,
            sync_dist=True,
        )

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if self.config.vis.vis_test:
            self.model.vis(
                batch,
                batch_idx=batch_idx,
                step=global_step,
                split="test",
                vis_traj=False,
                render_blender=False,
                render_gt=False,  # do not render gt for fast testing
                log_wandb=False,
            )

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # test_step_output = self.model.test(batch, global_step, batch_idx)
        # self.test_step_outputs.append(test_step_output)
        return loss

    def on_test_epoch_end(self) -> None:
        # print("Computing TEST Metrics...")
        # metrics = self.model.compute_metrics(self.test_step_outputs, "test")
        # self.acc_logger.log_scalar_dict(metrics)
        # self.acc_logger.write_summary(self.logger, self.global_step)
        # self.acc_logger.write_img_summary(self.logger, self.global_step)
        # self.acc_logger.write_vid_summary(self.logger, self.global_step)
        # self.test_step_outputs.clear()
        # self.log_dict(
        #     {k: v[0] for k, v in metrics.items()}, on_epoch=True, sync_dist=True
        # )
        torch.cuda.empty_cache()
        print("Test finished")

    def train_dataloader(self):
        dataset = DATASETS[self.config.dataset](self.config, split="train")
        if torch.distributed.is_initialized():
            sampler = DistributedSampler(
                dataset,
                shuffle=True,
                drop_last=True,
            )
            shuffle = False  # distributed sampler does it for us
        else:
            sampler = None
            shuffle = True
        batch_size = self.config.batch_size
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=True,
            # persistent_workers=True,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_fn,
            pin_memory=True,
            shuffle=shuffle,
            sampler=sampler,
        )

    def val_dataloader(self):
        dataset = DATASETS[self.config.dataset](self.config, split="val")
        if torch.distributed.is_initialized():
            sampler = DistributedSampler(
                dataset,
                shuffle=False,  # No shuffling for evaluation/testing
                drop_last=True,
            )
        else:
            sampler = None
        return DataLoader(
            dataset,
            batch_size=self.config.val_batch_size,
            # persistent_workers=True,
            num_workers=self.config.num_workers,
            collate_fn=dataset.collate_fn,
            pin_memory=True,
            shuffle=False,
            sampler=sampler,
        )

    def test_dataloader(self):
        # for testing, we use training set to sample number of faces distribution
        dataset = DATASETS[self.config.dataset](self.config, split="train_for_test")
        if torch.distributed.is_initialized():
            sampler = DistributedSampler(
                dataset,
                shuffle=False,  # No shuffling for evaluation/testing
                drop_last=False,
            )
        else:
            sampler = None
        return DataLoader(
            dataset,
            batch_size=self.config.test_batch_size,
            # persistent_workers=True,
            num_workers=self.config.num_workers,
            collate_fn=dataset.collate_fn,
            pin_memory=True,
            shuffle=False,
            sampler=sampler,
        )

    def configure_optimizers(self):
        optimizer = build_optimizer(
            self.config.optimizer, filter(lambda p: p.requires_grad, self.parameters())
        )
        lr_scheduler = build_lr_scheduler(self.config.lr_scheduler, optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def check_summary_step(self, step):
        if step == 0:
            return False
        return step % self.config.summary_step == 0

    def check_vis_step(self, step):
        if step == 0:
            return False
        return step % self.config.vis_train_step == 0

    @rank_zero_only
    def log_resources(self):
        self.acc_logger.log_scalar("lr", self.lr_schedulers().get_last_lr())
        resource_prefix = "resources"
        self.acc_logger.log_scalar(
            f"{resource_prefix}/time/time_per_step",
            [self.acc_logger.get_step_time() / self.config.summary_step],
        )
