# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
Training and evaluation engine for object detection with DINOv3 backbone.
Modified from Plain-DETR (https://github.com/SJ567/Plain-DETR)
"""

import copy
import math
import os
import sys
from typing import Iterable

import torch
import torch.distributed as dist

from .train import SetCriterion
from .util import box_ops
from .util.misc import MetricLogger, SmoothedValue, reduce_dict, get_total_grad_norm


def train_hybrid(outputs, targets, k_one2many, criterion, lambda_one2many):
    """Compute hybrid one-to-one and one-to-many loss."""
    # one-to-one loss
    loss_dict = criterion(outputs, targets)

    # Prepare one-to-many targets
    multi_targets = copy.deepcopy(targets)
    for target in multi_targets:
        target["boxes"] = target["boxes"].repeat(k_one2many, 1)
        target["labels"] = target["labels"].repeat(k_one2many)

    # Prepare one-to-many outputs
    outputs_one2many = {
        "pred_logits": outputs["pred_logits_one2many"],
        "pred_boxes": outputs["pred_boxes_one2many"],
        "aux_outputs": outputs["aux_outputs_one2many"],
    }
    if "pred_boxes_old_one2many" in outputs:
        outputs_one2many["pred_boxes_old"] = outputs["pred_boxes_old_one2many"]
        outputs_one2many["pred_deltas"] = outputs["pred_deltas_one2many"]

    # one-to-many loss
    loss_dict_one2many = criterion(outputs_one2many, multi_targets)
    for key, value in loss_dict_one2many.items():
        if key + "_one2many" in loss_dict:
            loss_dict[key + "_one2many"] += value * lambda_one2many
        else:
            loss_dict[key + "_one2many"] = value * lambda_one2many

    return loss_dict


def data_prefetcher(data_loader, device, prefetch=True):
    """Data prefetcher to speed up data loading."""
    stream = torch.cuda.Stream() if prefetch else None
    first = True
    data_iter = iter(data_loader)

    def next_batch():
        nonlocal stream, first, data_iter
        if not prefetch:
            return next(data_iter)

        with torch.cuda.stream(stream):
            batch = next(data_iter)
            samples = batch[0].to(device, non_blocking=True)
            targets = [{k: v.to(device) for k, v in t.items()} for t in batch[1]]
            if first:
                # Warm up CUDA stream - handle NestedTensor
                if hasattr(samples, 'tensors'):
                    _ = samples.tensors[0].sum()
                else:
                    _ = samples[0].sum()
                first = False
            return samples, targets

    return next_batch


def train_one_epoch(
    model: torch.nn.Module,
    criterion: SetCriterion,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    lr_scheduler,
    max_norm: float = 0,
    k_one2many: int = 0,
    lambda_one2many: float = 1.0,
    use_fp16: bool = False,
    scaler: torch.cuda.amp.GradScaler = None,
    print_freq: int = 10,
):
    """Train for one epoch."""
    model.train()
    criterion.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("class_error", SmoothedValue(window_size=1, fmt="{value:.2f}"))
    metric_logger.add_meter("grad_norm", SmoothedValue(window_size=1, fmt="{value:.2f}"))

    header = f"Epoch: [{epoch}]"
    epoch_size = len(data_loader)

    # Use prefetcher for faster data loading
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher()

    for idx in range(epoch_size):
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_fp16):
            outputs = model(samples)

            if k_one2many > 0:
                loss_dict = train_hybrid(outputs, targets, k_one2many, criterion, lambda_one2many)
            else:
                loss_dict = criterion(outputs, targets)

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Reduce losses over all GPUs for logging
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        if use_fp16:
            scaler.scale(losses).backward()
            if max_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            else:
                grad_norm = get_total_grad_norm(model.parameters(), norm_type=2)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            if max_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            else:
                grad_norm = get_total_grad_norm(model.parameters(), norm_type=2)
            optimizer.step()

        lr_scheduler.step()

        # Load next batch
        samples, targets = prefetcher()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        metric_logger.update(class_error=loss_dict_reduced.get("class_error", 0))
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_norm)

        if idx % print_freq == 0:
            print(f"{header} [{idx}/{epoch_size}]")

    metric_logger.synchronize_between_processes()
    print(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model,
    criterion,
    postprocessors,
    data_loader,
    base_ds,
    device,
    output_dir=None,
    use_wandb=False,
    step=0,
    print_freq=10,
    reparam=False,
):
    """Evaluate on dataset."""
    # Save and modify query settings for evaluation
    save_num_queries = model.module.num_queries
    save_two_stage_num_proposals = model.module.transformer.two_stage_num_proposals
    model.module.num_queries = model.module.num_queries_one2one
    model.module.transformer.two_stage_num_proposals = model.module.num_queries

    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("class_error", SmoothedValue(window_size=1, fmt="{value:.2f}"))
    header = "Test:"

    from .util.coco_eval import CocoEvaluator

    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    prefetcher = data_prefetcher(data_loader, device, prefetch=False)
    epoch_size = len(data_loader)

    for idx in range(epoch_size):
        samples, targets = next(prefetcher)

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # Reduce losses
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict
        }
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()), **loss_dict_reduced_scaled)
        metric_logger.update(class_error=loss_dict_reduced.get("class_error", 0))

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors["bbox"](outputs, orig_target_sizes)

        res = {target["image_id"].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if idx % print_freq == 0:
            print(f"{header} [{idx}/{epoch_size}]")

    # Gather stats
    metric_logger.synchronize_between_processes()
    print(f"Averaged stats: {metric_logger}")

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None and "bbox" in iou_types:
        stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()

    # Restore model settings
    model.module.num_queries = save_num_queries
    model.module.transformer.two_stage_num_proposals = save_two_stage_num_proposals

    return stats, coco_evaluator
