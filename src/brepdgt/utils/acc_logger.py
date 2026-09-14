import time
from typing import List, Dict
from pytorch_lightning.utilities import rank_zero_only
from torch.distributed import get_rank
from collections import defaultdict
import numpy as np
from time import time
import wandb


class AccLogger:
    def __init__(self):
        self.scalar_summaries = defaultdict(list)
        self.hist_summaries = defaultdict(list)
        self.img_summaries = defaultdict(list)
        self.vid_summaries = {}
        self.step_time = time()

    @rank_zero_only
    def log_scalar(self, k, v: List):
        assert isinstance(
            v, list
        ), f"type of key {k} has value type {type(v)}. Only lists are allowed"
        self.scalar_summaries[k] += v

    @rank_zero_only
    def log_scalar_dict(self, summary_dict: Dict):
        for k, v in summary_dict.items():
            assert isinstance(
                v, list
            ), f"type of key {k} has value type {type(v)}. Only lists are allowed"
            self.scalar_summaries[k] += v

    @rank_zero_only
    def log_hist(self, k, v: List):
        assert isinstance(
            v, list
        ), f"type of key {k} has value type {type(v)}. Only lists are allowed"
        self.hist_summaries[k] += v

    @rank_zero_only
    def log_hist_dict(self, summary_dict: Dict):
        for k, v in summary_dict.items():
            assert isinstance(
                v, list
            ), f"type of key {k} has value type {type(v)}. Only lists are allowed"
            self.hist_summaries[k] += v

    def log_scalar_and_hist(self, k, v: List):
        assert isinstance(
            v, list
        ), f"type of key {k} has value type {type(v)}. Only lists are allowed"
        self.scalar_summaries[k] += v
        self.hist_summaries[k + "_hist"] += v

    def log_scalar_and_hist_dict(self, summary_dict):
        for k, v in summary_dict.items():
            assert isinstance(
                v, list
            ), f"type of key {k} has value type {type(v)}. Only lists are allowed"
            self.scalar_summaries[k] += v
            self.hist_summaries[k + "_hist"] += v

    @rank_zero_only
    def log_imgs(self, k, v: List):
        assert isinstance(
            v, list
        ), f"type of key {k} has value type {type(v)}. Only lists are allowed"
        self.img_summaries[k] += v

    @rank_zero_only
    def log_vid(self, k, v: str):
        assert isinstance(
            v, str
        ), f"type of key {k} has value type {type(v)}. Only vid_path is allowed"
        self.vid_summaries[k] = v

    @rank_zero_only
    def write_summary(self, logger, step):
        # write all the averaged summaries
        # write scalar summaries
        scalar_summaries_avg = {}
        for k, v in self.scalar_summaries.items():
            v = np.array(v).mean().item()
            scalar_summaries_avg[k] = v
        scalar_summaries_avg["global_step"] = step
        logger.experiment.log(scalar_summaries_avg)

        # write hist summaries
        hist_summaries = {k: wandb.Histogram(v) for k, v in self.hist_summaries.items()}
        hist_summaries["global_step"] = step
        logger.experiment.log(hist_summaries)

        self.scalar_summaries.clear()
        self.hist_summaries.clear()

    @rank_zero_only
    def write_img_summary(self, logger, step):
        self.img_summaries["global_step"] = step
        # write image summaries
        logger.experiment.log(self.img_summaries)
        self.img_summaries.clear()

    @rank_zero_only
    def write_vid_summary(self, logger, step):
        vid_summaries_new = {}
        for k, v in self.vid_summaries.items():
            vid_summaries_new[k] = wandb.Video(v, format="mp4")
        vid_summaries_new["global_step"] = step
        logger.experiment.log(vid_summaries_new)
        self.vid_summaries.clear()

    @rank_zero_only
    def get_step_time(self):
        step_time = time() - self.step_time
        self.step_time = time()
        return step_time


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        ret = func(*args, **kwargs)
        time_executed = time() - start_time
        if hasattr(args[0], "acc_logger"):  # args[0] is self
            args[0].acc_logger.log_scalar(
                f"resources/time/{type(args[0]).__name__}_{func.__name__}",
                [time_executed],
            )
        return ret

    return wrapper
