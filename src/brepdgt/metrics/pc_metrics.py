from brepdgt.utils.chamfer_distance import ChamferDistance
from plyfile import PlyData
from pathlib import Path
from glob import glob
from tqdm import tqdm
import numpy as np
import os
from multiprocessing import Pool
from typing import Dict
import torch
import typer
import time

app = typer.Typer()

NUM_TRHEADS = 6


def read_ply(file_path):
    ply_data = PlyData.read(file_path)
    vertex_data = ply_data["vertex"]
    points = np.vstack([vertex_data["x"], vertex_data["y"], vertex_data["z"]]).T
    return points


def normalize_pcs_tensor(point_clouds: torch.Tensor) -> torch.Tensor:
    """
    Normalize a batch of point clouds so each shape's longest axis fits within [-1, 1]
    and the bounding box is centered at the origin.

    Parameters:
        point_clouds (torch.Tensor): A batch of point clouds of shape (N_shapes, N_points, 3).

    Returns:
        torch.Tensor: The normalized batch of point clouds.
    """
    max_coords = point_clouds.max(dim=1, keepdim=True)[0]  # Shape: (N_shapes, 1, 3)
    min_coords = point_clouds.min(dim=1, keepdim=True)[0]  # Shape: (N_shapes, 1, 3)

    bbox_center = (max_coords + min_coords) / 2  # Shape: (N_shapes, 1, 3)
    centered_pcs = point_clouds - bbox_center  # bounding box is centered at the origin

    ranges = max_coords - min_coords  # Shape: (N_shapes, 1, 3)
    max_range, _ = ranges.max(dim=2, keepdim=True)  # Shape: (N_shapes, 1, 1)
    max_range = torch.where(
        max_range == 0, torch.tensor(1e-8, device=point_clouds.device), max_range
    )
    normalized_pcs = centered_pcs / (max_range / 2)

    return normalized_pcs


def normalize_pc_brepgen(points: torch.Tensor):
    """

    :param points: tensor of B x N x 3
    :return:
    """
    # normalize
    mean = points.mean(dim=1, keepdim=True)
    points = points - mean
    # fit to unit cube
    scale = torch.amax(torch.abs(points), dim=(1, 2), keepdim=True)
    points = points / scale
    return points


def distChamfer(a, b):
    chamfer_dist = ChamferDistance()
    dl, dr, idx1, idx2 = chamfer_dist(a, b)
    return dl, dr


# Adapted from https://github.com/xuqiantong/
# GAN-Metrics/blob/master/framework/metric.py
def knn(Mxx, Mxy, Myy, k, sqrt=False):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    M = torch.cat(
        [torch.cat((Mxx, Mxy), 1), torch.cat((Mxy.transpose(0, 1), Myy), 1)], 0
    )
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float("inf")
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(
        k, 0, False
    )

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1).to(Mxx)).float()

    s = {
        "tp": (pred * label).sum(),
        "fp": (pred * (1 - label)).sum(),
        "fn": ((1 - pred) * label).sum(),
        "tn": ((1 - pred) * (1 - label)).sum(),
    }

    s.update(
        {
            "precision": s["tp"] / (s["tp"] + s["fp"] + 1e-10),
            "recall": s["tp"] / (s["tp"] + s["fn"] + 1e-10),
            "acc_t": s["tp"] / (s["tp"] + s["fn"] + 1e-10),
            "acc_f": s["tn"] / (s["tn"] + s["fp"] + 1e-10),
            "acc": torch.eq(label, pred).float().mean(),
        }
    )
    return s


def _pairwise_CD(sample_pcs, ref_pcs, batch_size=None, verbose=False):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    all_cd = []
    iterator = range(N_sample)
    if verbose:
        import tqdm

        iterator = tqdm.tqdm(iterator)
    if batch_size is None:
        batch_size = N_ref

    for sample_b_start in iterator:
        sample_batch = sample_pcs[sample_b_start]

        cd_lst = []
        sub_iterator = range(0, N_ref, batch_size)
        for ref_b_start in sub_iterator:
            ref_b_end = min(N_ref, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]

            batch_size_ref = ref_batch.size(0)
            point_dim = ref_batch.size(2)
            sample_batch_exp = sample_batch.view(1, -1, point_dim).expand(
                batch_size_ref, -1, -1
            )
            sample_batch_exp = sample_batch_exp.contiguous()

            dl, dr = distChamfer(sample_batch_exp, ref_batch)
            cd_lst.append((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1))

        cd_lst = torch.cat(cd_lst, dim=1)
        all_cd.append(cd_lst)

    all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref

    return all_cd


def lgan_mmd_cov(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        "lgan_cov": cov,
        "lgan_mmd": mmd,
        # "lgan_mmd_smp": mmd_smp,
    }


def compute_pc_metrics(
    sample_pcs, ref_pcs, batch_size=None, verbose=False, normalize: str = "longest_axis"
) -> Dict:
    """Compute point cloud metrics between sample and reference point clouds

    Args:
        sample_pcs: Sample point clouds
        ref_pcs: Reference point clouds
        batch_size: Batch size for processing
        verbose: Whether to show progress bars
        normalize: Whether to normalize point clouds before computing metrics

    Returns:
        Dict of metrics
    """
    results = {}

    # Optionally normalize point clouds
    if normalize == "longest_axis":
        sample_pcs = normalize_pcs_tensor(sample_pcs)
        ref_pcs = normalize_pcs_tensor(ref_pcs)
    elif normalize == "points_mean":
        sample_pcs = normalize_pc_brepgen(sample_pcs)
        ref_pcs = normalize_pc_brepgen(ref_pcs)
    else:
        raise ValueError(f"{normalize} not allowed")

    M_sr_cd = _pairwise_CD(sample_pcs, ref_pcs, batch_size, verbose=verbose)
    res_cd = lgan_mmd_cov(M_sr_cd)

    if ref_pcs.shape[0] == sample_pcs.shape[0]:
        M_rr_cd = _pairwise_CD(ref_pcs, ref_pcs, batch_size, verbose=verbose)
        M_ss_cd = _pairwise_CD(sample_pcs, sample_pcs, batch_size, verbose=verbose)

        # 1-NN results
        one_nn_cd_res = knn(M_ss_cd, M_sr_cd, M_rr_cd, 1, sqrt=False)
        results.update(
            {
                # "CD-rr": M_rr_cd.mean(),
                # "CD-rs": M_sr_cd.mean(),
                # "CD-ss": M_ss_cd.mean(),
                **{"%s-CD" % k: v for k, v in res_cd.items()},
                **{"1-NN-CD-%s" % k: v for k, v in one_nn_cd_res.items() if "acc" in k},
            }
        )
    else:
        results.update(
            {
                **{"%s-CD" % k: v for k, v in res_cd.items()},
            }
        )
        pass

    return results
