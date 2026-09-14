from typing import Tuple, Dict, Any, List, Union, Optional
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from brepdgt.models.base_model import BaseModule
from brepdgt.models.tokenizers import Tokens
from brepdgt.primitives.uvgrid import UvGrid


@dataclass(frozen=True)
class DetokenizerOutput:
    """
    Output of detokenizer
    """

    # logits of slicer.
    # Tensor of B x n_slicers x allowed_n_slicers
    type_logits: Union[torch.Tensor, None]
    # Tensor of B x n_slicers x allowed_n_slicers x uvgrid_dim
    uvgrid: UvGrid


class Detokenizer(BaseModule):
    """
    Maps tokens to slicer
    """

    def forward(self, tokens: Tokens) -> DetokenizerOutput:
        """
        :param tokens:
        :return:
        """
        raise NotImplementedError()

    @staticmethod
    def type_loss(
        logits_pred, labels_gt, prefix
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Computes slicer type loss

        :param logits_pred: Tensor of B x n_slicers x n_labels
        :param labels_gt: Tensor of B x n_slicers
        :param prefix: str
            prefix for log_dict
        :return:
            loss: Tensor of B'
                B', number of non -1 values
            log_dict: dict
        """

        # compute loss
        logits_pred_flat = logits_pred.reshape(-1, logits_pred.shape[-1])
        labels_gt_flat = labels_gt.flatten()  # B * n_slicers
        loss_type = F.cross_entropy(
            logits_pred_flat,
            labels_gt_flat,
            reduction="none",
        )

        log_dict = {
            f"loss/{prefix}/type": loss_type.tolist(),  # for every slicer
        }

        return loss_type, log_dict

    def decode(
        self,
        tokens: Tokens,
        max_n_prims: int,
    ) -> UvGrid:
        """
        Generate zone graphs from tokens and auxiliary arguments
        :param tokens: Tokens
        :param max_n_prims: maximum number of slicers
        :return:
            UvGrid
        """
        raise NotImplementedError()

    def get_empty_mask_by_removing_duplicates(
        self,
        x: torch.Tensor,
        tokenizer_dedup_thresh: float,
        tokenizer_dedup_rel_thresh: float,
        empty_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Deduplicates masks based on feature distances.
        :param x: Tensor of shape B x n_seq x h x w x feat_dim
        :param tokenizer_dedup_thresh: float
            Average grid distance within this value will be deduplicated.
        :param empty_mask: Optional tensor of shape B x n_seq
            Preset empty_mask given by token
        :param tokenizer_dedup_rel_thresh: float
            Average grid distance within this value times min(grid_x_dist, grid_y_dist) will be deduplicated.
        :return: Boolean mask of shape B x n_seq indicating which elements to keep
        """
        batch_size, n_seq = x.shape[:2]

        # Compute pairwise distances within each batch
        x_flat = x.view(*x.shape[:2], -1, x.shape[4])  # B x n_prims x h*w x feat_dim
        dists = torch.norm(x_flat.unsqueeze(1) - x_flat.unsqueeze(2), dim=-1).mean(
            dim=-1
        )  # Shape: (B, N, N)

        # Create a mask for unique elements
        if empty_mask is not None:
            mask = empty_mask
        else:
            mask = torch.zeros((batch_size, n_seq), dtype=torch.bool, device=x.device)

        for b in range(batch_size):
            for i in range(n_seq):
                if not mask[b, i]:
                    # Check only if the current element is marked as unique
                    # Mask out all elements within threshold
                    if x.shape[-1] == 3:
                        # x is comprised of only coordinates
                        dist_thresh = min(
                            tokenizer_dedup_thresh,
                            tokenizer_dedup_rel_thresh
                            * torch.norm(x[b, i, 0, 0] - x[b, i, 1, 0]),
                            tokenizer_dedup_rel_thresh
                            * torch.norm(x[b, i, 0, 0] - x[b, i, 0, 1]),
                        )
                    else:
                        # x is comprised of coordinates + other features
                        dist_thresh = tokenizer_dedup_thresh
                    close_idxs = dists[b, i] < dist_thresh
                    close_idxs[i] = False  # Keep the current element as unique
                    mask[b, close_idxs] = True

        return mask
