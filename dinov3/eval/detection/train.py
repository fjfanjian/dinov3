# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
Training utilities for object detection with DINOv3 backbone.
Modified from Plain-DETR (https://github.com/SJ567/Plain-DETR)
"""

import math
import os
import sys
from typing import Iterable
import copy

import torch
import torch.nn.functional as F
from torch import nn

import dinov3.distributed as distributed

from .models.detr import build_model
from .config import DetectionHeadConfig
from .util import box_ops
from .util.misc import (
    NestedTensor,
    nested_tensor_from_tensor_list,
    accuracy,
    interpolate,
    is_dist_avail_and_initialized,
    inverse_sigmoid,
    _get_clones,
    reduce_dict,
    MetricLogger,
    SmoothedValue,
    get_total_grad_norm,
)


class HungarianMatcher(nn.Module):
    """Hungarian algorithm for matching predictions to ground truth targets.

    This module computes the assignment between targets and predictions for
    training the detector. It uses a combination of classification cost,
    bounding box L1 cost, and GIoU cost for optimal matching.
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        cost_bbox_type: str = "l1",
    ):
        """Initialize the matcher.

        Args:
            cost_class: Relative weight of classification error in matching cost
            cost_bbox: Relative weight of L1 error of bounding box coordinates
            cost_giou: Relative weight of GIoU loss in matching cost
            cost_bbox_type: Type of box loss calculation ("l1" or "reparam")
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_bbox_type = cost_bbox_type
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs can't be 0"

    def forward(self, outputs, targets):
        """Perform matching.

        Args:
            outputs: Dict containing:
                - pred_logits: [batch_size, num_queries, num_classes]
                - pred_boxes: [batch_size, num_queries, 4]
            targets: List of dicts, each containing:
                - labels: [num_target_boxes]
                - boxes: [num_target_boxes, 4]

        Returns:
            List of tuples (pred_idx, target_idx) for each batch element
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # Flatten to compute cost matrices in batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)

            # Concat target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Classification cost using focal loss
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # L1 cost between boxes
            if self.cost_bbox_type == "l1":
                cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            elif self.cost_bbox_type == "reparam":
                out_delta = outputs["pred_deltas"].flatten(0, 1)
                out_bbox_old = outputs["pred_boxes_old"].flatten(0, 1)
                tgt_delta = box_ops.bbox2delta(out_bbox_old, tgt_bbox)
                cost_bbox = torch.cdist(out_delta[:, None], tgt_delta, p=1).squeeze(1)
            else:
                raise NotImplementedError(f"Unknown cost_bbox_type: {self.cost_bbox_type}")

            # GIoU cost between boxes
            cost_giou = -box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(out_bbox),
                box_ops.box_cxcywh_to_xyxy(tgt_bbox)
            )

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            from scipy.optimize import linear_sum_assignment
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

            return [
                (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices
            ]


class SetCriterion(nn.Module):
    """Compute loss for DETR-style detectors.

    The process happens in two steps:
        1. Compute Hungarian assignment between ground truth boxes and predictions
        2. Supervise each matched pair (classification and bounding box)
    """

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: dict,
        losses: list,
        focal_alpha: float = 0.25,
        reparam: bool = False,
    ):
        """Initialize the criterion.

        Args:
            num_classes: Number of object categories (excluding no-object class)
            matcher: Module to compute matching between targets and proposals
            weight_dict: Dict of loss names to relative weights
            losses: List of losses to apply
            focal_alpha: Alpha parameter in Focal Loss
            reparam: Whether to use reparameterized box loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.loss_bbox_type = 'reparam' if reparam else 'l1'

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss using Focal Loss."""
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        # One-hot encoding
        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        loss_ce = (
            self._focal_loss(
                src_logits,
                target_classes_onehot,
                num_boxes,
                alpha=self.focal_alpha,
                gamma=2,
            )
            * src_logits.shape[1]
        )
        losses = {"loss_ce": loss_ce}

        if log:
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def _focal_loss(self, src_logits, target_classes_onehot, num_boxes, alpha, gamma):
        """Focal loss for dense prediction."""
        prob = src_logits.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(src_logits, target_classes_onehot, reduction="none")
        p_t = prob * target_classes_onehot + (1 - prob) * (1 - target_classes_onehot)
        modulating_factor = (1 - p_t) ** gamma
        alpha_factor = alpha * target_classes_onehot + (1 - alpha) * (1 - target_classes_onehot)

        loss = alpha_factor * modulating_factor * ce_loss
        return loss.sum() / num_boxes

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Bounding box regression loss (L1 + GIoU)."""
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}

        if self.loss_bbox_type == "l1":
            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        elif self.loss_bbox_type == "reparam":
            src_deltas = outputs["pred_deltas"][idx]
            src_boxes_old = outputs["pred_boxes_old"][idx]
            target_deltas = box_ops.bbox2delta(src_boxes_old, target_boxes)
            loss_bbox = F.l1_loss(src_deltas, target_deltas, reduction="none")
        else:
            raise NotImplementedError(f"Unknown loss_bbox_type: {self.loss_bbox_type}")

        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Cardinality error for logging purposes."""
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        return {"cardinality_error": card_err}

    def _get_src_permutation_idx(self, indices):
        """Permute predictions following indices."""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        """Permute targets following indices."""
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        """Get loss for a specific loss type."""
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"Unknown loss type: {loss}"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """Compute loss.

        Args:
            outputs: Model output dict
            targets: List of target dicts, len(targets) == batch_size

        Returns:
            Dict of losses
        """
        # Remove auxiliary outputs for matching
        outputs_without_aux = {k: v for k, v in outputs.items() if k not in ("aux_outputs", "enc_outputs")}

        # Compute matching
        indices = self.matcher(outputs_without_aux, targets)

        # Compute average number of target boxes for normalization
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / distributed.get_world_size(), min=1).item()

        # Compute all requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # Auxiliary losses
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == "labels":
                        kwargs["log"] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Encoder losses
        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["labels"] = torch.zeros_like(bt["labels"])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                kwargs = {}
                if loss == "labels":
                    kwargs["log"] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


def build_matcher(args, reparam=False):
    """Build Hungarian matcher from config."""
    return HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
        cost_bbox_type='reparam' if reparam else 'l1',
    )
