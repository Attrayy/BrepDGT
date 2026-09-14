from dataclasses import dataclass, field
from dataclass_wizard import YAMLWizard
from typing import List
import yaml


@dataclass
class VisConfig:
    vis_batch: str = "first"
    vis_idxs: List[int] = field(
        default_factory=lambda: [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
    )  # idx to visualize
    n_examples: int = 0
    img: dict = field(
        default_factory=lambda: {
            "height": 75,
            "width": 75,
            "alpha": 0.9,
            "axis_ranges": [
                [-1.1, 1.1],
                [-1.1, 1.1],
            ],
        }
    )
    n_gen_samples: int = 12
    max_trajectory_points: int = 4  # diffusion trajectory points
    trajectory_stride: int = 20
    vis_trajectory: bool = True
    vis_test: bool = False
    render_blender: bool = True


@dataclass
class TokenizerConfig:
    name: str = ""


@dataclass
class DetokenizerConfig:
    name: str = ""


@dataclass
class DiffusionConfig:
    name: str = "gaussian_diffusion_1d"

    training_timesteps: int = 1000
    inference_timesteps: int = 1000
    objective: str = "pred_noise"  # ["pred_noise", "pred_x0", "pred_v"]
    noise_schedule: str = (
        "linear"  # ["linear", "cosine", "sqrt", "sigmoid", "snr_based"]
    )

    # Linear schedule params
    linear_beta0: float = 0.0001
    linear_betaT: float = 0.02

    # Cosine schedule params
    cosine_s: float = 0.008

    # Sqrt schedule params
    sqrt_s: float = 0.0001

    # Sigmoid schedule params
    sigmoid_start: float = -3
    sigmoid_end: float = 3
    sigmoid_tau: float = 0.5

    # SNR schedule params
    snr_min: float = 0.01
    snr_max: float = 1000
    snr_power: float = 1.0  # Power for log-SNR curve shaping

    ddim_sampling_eta: float = 0.0

    z_conditioning: bool = False
    z_augmentation: bool = False
    z_aug_step: int = 15

    cfg_scales: List[float] = field(default_factory=lambda: [3.0])

    coord_noise_schedule: str = "snr_based"
    coord_timesteps: int = 1000
    grid_mask_noise_schedule: str = "linear"
    grid_mask_timesteps: int = 500
    normal_noise_schedule: str = "snr_based"
    normal_timesteps: int = 500

    self_condition: bool = False

    model: dict = field(
        default_factory=lambda: {
            "name": "dit_1d_cross_attn",
            "options": {
                "hidden_size": 384,
                "depth": 12,
                "num_heads": 6,
                "use_pe": False,
                "final_residual": False,
                "uncond_prob": 0.1,
                "dropout": 0.0,
                "t_low_timesteps": 1000,
            },
        }
    )


@dataclass
class Config(YAMLWizard, key_transform="SNAKE"):
    log_dir: str = ""
    debug: bool = False

    ##### DATA #####
    dataset: str = "abc_50"
    data_dim: int = 3
    n_grid: int = 8
    max_n_prims: int = (
        50  # maximum nuber of primitives to encode. Tune this parameter later
    )
    h5_path: str = "./data/abc_processed/v1_1_grid8.h5"
    random_augmentations: str = (
        "none"  # ["none", "scale_translate", "scale_translate_rotation"]
    )
    scale_min: float = 0.5
    scale_max: float = 1.2
    translate_min: float = -0.2
    translate_max: float = 0.2

    ##### MODEL #####
    model: str = "brepdgt"
    z_dim: int = 256  # dimension of global latent
    n_z: int = 1  # number of z
    x_dim: int = 256  # dimension of token to diffusion

    tokenizer: TokenizerConfig = field(default_factory=lambda: TokenizerConfig())
    tokenizer_dedup_thresh: float = (
        0.01  # Average grid distance within this absolute value will be deduplicated.
    )
    tokenizer_dedup_rel_thresh: float = 0.5  #  Average grid distance within this value times min(grid_x_dist, grid_y_dist) will be deduplicated.
    detokenizer: DetokenizerConfig = field(default_factory=lambda: DetokenizerConfig())

    token_vae_ckpt_path: str = ""

    diffusion: DiffusionConfig = field(default_factory=lambda: DiffusionConfig())

    ##### TRAIN #####
    alpha_coord: float = 1.0
    alpha_grid_mask: float = 1.0

    batch_size: int = 128
    num_epochs: int = 1000
    summary_step: int = 100

    optimizer: dict = field(
        default_factory=lambda: {
            "type": "Adam",
            "options": {"lr": 0.0005, "weight_decay": 0.0},
        }
    )

    clip_grad: dict = field(
        default_factory=lambda: {
            "type": "norm",
            "options": {"max_norm": 0.5},
        }
    )

    lr_scheduler: dict = field(
        default_factory=lambda: {
            "type": "StepLR",
            "options": {"step_size": 100000, "gamma": 1.0},
        }
    )
    test_on_training_end: bool = False  # test on end of training with best ckpt

    ##### POSTPROCESSING #####
    pp_grid_res: int = 256  # resolution for postprocessing grid res, such as winding numbers or occupancy

    ##### EVAL #####
    val_epoch: int = 10  # do validation every 10 epochs
    limit_val_batches: float = 1.0
    val_batch_size: int = 64
    val_num_sample: int = 2048

    ##### TEST #####
    test_in_loop: bool = False  # test while training
    test_nth_val: int = 5  # test will be called every n-th validation
    n_pc_metric_samples: int = (
        512  # number of testing shapes to be used for pc_metric eval during training
    )
    test_num_sample: int = 2048  # number of total test samples
    test_num_pts: int = 1024  # number of points to sample during testing, if less, pts will be duplicated
    test_batch_size: int = 64
    test_chunk: str = ""  # chunk testset

    # use resampling during diffusion
    test_use_resampling: bool = False
    test_resampling_jump_length: int = 100
    test_resampling_repetitions: int = 5
    test_resampling_start_t: int = 900

    # ##### VIS #####
    sample_mode: str = "data"  # ["data", "fixed"], "data" samples following the num_prims in dataset, "fixed" samples evenly from 2~max_n_prims
    vis_train_step: int = 1000
    vis: VisConfig = field(default_factory=lambda: VisConfig())

    ##### CHECKPOINT #####
    ckpt_every_n_train_steps: int = 2000
    save_top_k: int = 3
    ckpt_file_name: str = "{epoch:04d}-{step:06d}-{val_loss:.2f}"
    monitor: str = "params_error"  # ["val_loss", "params_error"]

    ##### UTILS #####
    num_workers: int = 16
    seed: int = 0
    accelerator: str = "gpu"
    devices: int = 1
    fast_dev_run: bool = False
    overfit: bool = False
    wandb_run_id: str = None
    compile: bool = False  # use torch.compile

    ##### DEBUG #####
    debug_data_size: int = 1024
    overfit_data_repetition: int = 2048
    overfit_data_size: int = 10


def load_config(config_path: str, override: str) -> Config:
    config: dict = yaml.load(open(config_path), Loader=yaml.FullLoader)
    # override options
    for option in override.split("|"):
        if not option:
            continue
        address, value = option.split("=")
        keys = address.split(".")
        here = config
        for key in keys[:-1]:
            if key not in here:
                raise ValueError(
                    "{} is not defined in config file. "
                    "Failed to override.".format(address)
                )
            here = here[key]
        if keys[-1] not in here:
            raise ValueError(
                "{} is not defined in config file. "
                "Failed to override.".format(address)
            )
        here[keys[-1]] = yaml.load(value, Loader=yaml.FullLoader)

    config: Config = Config.from_yaml(yaml.dump(config))
    return config
