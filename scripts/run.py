import glob
import os
import shutil
import lightning.pytorch as pl
import typer
import torch
import resource
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
import wandb
from brepdgt.lit_model import LitModel
from brepdgt.config import load_config
from typing import Optional

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8196, rlimit[1]))
torch.multiprocessing.set_sharing_strategy("file_system")

app = typer.Typer(pretty_exceptions_enable=False)


def run_cmd(cmd):
    print(cmd)
    os.system(cmd)


@app.command()
def main(
    log_dir: str,  # log_dir  # should be in format of './logs/{group}/{log_name}
    config_path: str = "./configs/default.yaml",  # config
    override: str = "",  # should be string in format of {key1}={value1}|{key2}={value2}
    ckpt_path: str = None,  # path to the checkpoint
    resume_latest_ckpt: bool = False,  # use the latest ckpt
    test: bool = False,  # test mode
    wandb_offline: bool = False,  # set wandb offline
    debug: bool = False,  # debug mode
):

    # load config if ckpt exists
    if resume_latest_ckpt:
        assert os.path.isdir(log_dir), "log_dir {} does not exist".format(log_dir)
        latest_path = os.path.join(log_dir, "ckpts", "last.ckpt")
        print("Loading latest checkpoint: {}".format(latest_path))
        base_dir = os.path.dirname(os.path.dirname(latest_path))
        config_path = os.path.join(base_dir, "config.yaml")
        ckpt_path = latest_path
    elif ckpt_path is not None:
        base_dir = os.path.dirname(os.path.dirname(ckpt_path))
        config_path = os.path.join(base_dir, "config.yaml")
        wandb_offline = True

    if test:
        wandb_offline = True

    if wandb_offline:
        run_cmd("wandb disabled")
    else:
        run_cmd("wandb online")

    if debug:
        log_dir_split = log_dir.split("/")
        log_dir_split[-2] = "debug"
        log_dir = "/".join(log_dir_split)

    # Load config first to get default devices if needed
    config = load_config(config_path, override)
    config.debug = debug
    config.log_dir = log_dir

    # Check available GPUs (this will only see GPUs set by CUDA_VISIBLE_DEVICES)
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        print("No GPUs available. Running on CPU.")
        devices = None
        accelerator = "cpu"
    else:
        print(f"Found {available_gpus} GPUs")
        accelerator = config.accelerator if hasattr(config, "accelerator") else "cuda"
        devices = available_gpus

    # Initialize wandb based on DDP status
    is_main_process = os.environ.get("LOCAL_RANK", "0") == "0"

    if is_main_process:
        if (ckpt_path is not None) and (not test):
            run_id = config.wandb_run_id
        else:
            run_id = None

        wandb_logger = WandbLogger(
            project="cadgen",
            name=log_dir.split("/")[-1],
            group=log_dir.split("/")[-2],
            id=run_id,
            resume="allow",
            entity="mingimango",
        )
    else:
        wandb_logger = None

    # build model
    model = LitModel(config)

    if wandb_logger is not None:
        wandb_logger.watch(model, log_freq=config.summary_step)
        if wandb.run is not None:
            config.wandb_run_id = wandb.run.id

    # Save config only on main process
    if is_main_process:
        os.umask(0)
        os.makedirs(config.log_dir, mode=0o777, exist_ok=True)
        if not test:
            config_save_path = os.path.join(config.log_dir, "config.yaml")
            config.to_yaml_file(config_save_path)
            print("Config saved to {}".format(config.log_dir))

    ckpt_callback = ModelCheckpoint(
        dirpath=os.path.join(config.log_dir, "ckpts"),
        save_last=True,
        save_top_k=config.save_top_k,
        monitor=config.monitor,
        every_n_epochs=config.test_nth_val * config.val_epoch,
        filename=config.ckpt_file_name,
    )
    ModelCheckpoint.CHECKPOINT_EQUALS_CHAR = "_"

    if debug:
        max_epochs = 50 if config.num_epochs > 50 else config.num_epochs
        max_steps = -1
    else:
        max_epochs = config.num_epochs
        max_steps = -1

    strategy = "auto"
    use_distributed_sampler = False  # we use our own distributed sampler in lit model

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        default_root_dir=config.log_dir,
        fast_dev_run=config.fast_dev_run,
        val_check_interval=None,
        max_epochs=max_epochs,
        max_steps=max_steps,
        strategy=strategy,
        use_distributed_sampler=use_distributed_sampler,
        num_sanity_val_steps=0,
        log_every_n_steps=config.summary_step,
        limit_val_batches=config.limit_val_batches,
        check_val_every_n_epoch=config.val_epoch,
        enable_checkpointing=True,
        logger=wandb_logger,
        callbacks=[ckpt_callback],
    )

    if test:
        trainer.test(model, ckpt_path=ckpt_path)
    else:
        print(config)
        trainer.fit(model, ckpt_path=ckpt_path)
        if is_main_process:  # Only on main process
            best_ckpt = ckpt_callback.best_model_path
            if config.test_on_training_end:
                trainer.test(model, ckpt_path=best_ckpt)


if __name__ == "__main__":
    app()
