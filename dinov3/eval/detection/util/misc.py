# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

# ------------------------------------------------------------------------
# Plain-DETR
# Copyright (c) 2023 Xi'an Jiaotong University & Microsoft Research Asia.
# Licensed under The MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import copy
import datetime
from typing import List, Optional

import dinov3.distributed as distributed
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
from torch import Tensor


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = distributed.get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

    def __len__(self):
        return len(self.tensors)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
        norm_type,
    )
    return total_norm


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def get_param_dict(model, args, return_name=False, use_layerwise_decay=False):
    # sanity check: a variable could not match backbone_names and linear_proj_names at the same time
    for n, p in model.named_parameters():
        if match_name_keywords(n, args.lr_backbone_names) and match_name_keywords(n, args.lr_linear_proj_names):
            raise ValueError

    param_dicts = [
        {
            "params": [
                p if not return_name else n
                for n, p in model.named_parameters()
                if not match_name_keywords(n, args.lr_backbone_names)
                and not match_name_keywords(n, args.lr_linear_proj_names)
                and not match_name_keywords(n, args.wd_norm_names)
                and p.requires_grad
            ],
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p if not return_name else n
                for n, p in model.named_parameters()
                if match_name_keywords(n, args.lr_backbone_names)
                and not match_name_keywords(n, args.lr_linear_proj_names)
                and not match_name_keywords(n, args.wd_norm_names)
                and p.requires_grad
            ],
            "lr": args.lr_backbone,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p if not return_name else n
                for n, p in model.named_parameters()
                if not match_name_keywords(n, args.lr_backbone_names)
                and match_name_keywords(n, args.lr_linear_proj_names)
                and not match_name_keywords(n, args.wd_norm_names)
                and p.requires_grad
            ],
            "lr": args.lr * args.lr_linear_proj_mult,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p if not return_name else n
                for n, p in model.named_parameters()
                if not match_name_keywords(n, args.lr_backbone_names)
                and not match_name_keywords(n, args.lr_linear_proj_names)
                and match_name_keywords(n, args.wd_norm_names)
                and p.requires_grad
            ],
            "lr": args.lr,
            "weight_decay": args.weight_decay * args.wd_norm_mult,
        },
        {
            "params": [
                p if not return_name else n
                for n, p in model.named_parameters()
                if match_name_keywords(n, args.lr_backbone_names)
                and not match_name_keywords(n, args.lr_linear_proj_names)
                and match_name_keywords(n, args.wd_norm_names)
                and p.requires_grad
            ],
            "lr": args.lr_backbone,
            "weight_decay": args.weight_decay * args.wd_norm_mult,
        },
        {
            "params": [
                p if not return_name else n
                for n, p in model.named_parameters()
                if not match_name_keywords(n, args.lr_backbone_names)
                and match_name_keywords(n, args.lr_linear_proj_names)
                and match_name_keywords(n, args.wd_norm_names)
                and p.requires_grad
            ],
            "lr": args.lr * args.lr_linear_proj_mult,
            "weight_decay": args.weight_decay * args.wd_norm_mult,
        },
    ]
    return param_dicts


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class SmoothedValue:
    """Track a series of values and provide access to smoothed values."""

    def __init__(self, window_size=20, fmt=None):
        from collections import deque
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.window_size = window_size
        self.reset()
        self.fmt = fmt

    def reset(self):
        from collections import deque
        self.deque = deque()
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        if len(self.deque) > self.window_size:
            self.deque.popleft()
        self.total += value
        self.count += 1

    def synchronize_between_processes(self):
        """Synchronize values across distributed processes."""
        world_size = distributed.get_world_size()
        if world_size < 2:
            return

        # Gather values from all processes
        value_list = all_gather([list(self.deque), self.total, self.count])
        all_deques = [deque for sublist in value_list for deque in sublist[0]]
        all_totals = [v[1] for v in value_list]
        all_counts = [v[2] for v in value_list]

        # Merge deques (keep only last window_size items)
        merged_deque = []
        for d in all_deques:
            merged_deque.extend(d)
        while len(merged_deque) > self.window_size:
            merged_deque = merged_deque[-self.window_size:]

        self.deque = merged_deque
        self.total = sum(all_totals)
        self.count = sum(all_counts)

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        if len(d) == 0:
            return 0.0
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        if len(d) == 0:
            return 0.0
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count if self.count > 0 else 0.0

    @property
    def max(self):
        if len(self.deque) == 0:
            return 0.0
        return max(self.deque)

    @property
    def value(self):
        if len(self.deque) == 0:
            return 0.0
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )


def all_gather(data):
    """Gather data from all processes."""
    world_size = distributed.get_world_size()
    if world_size < 2:
        return data

    output = [None] * world_size
    dist.all_gather_object(output, data)
    return output


def is_dist_avail_and_initialized():
    """Check if distributed training is available and initialized."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


class MetricLogger:
    def __init__(self, delimiter=" "):
        from collections import defaultdict
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header):
        import time
        i = 0
        if not hasattr(iterable, "__len__"):
            raise TypeError("iterable must have __len__ method")
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        while i < len(iterable):
            obj = iterable[i]
            end = time.time()
            data_time.update(end - start_time)
            yield obj
            iter_time.update(end - start_time)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=iter_time,
                            data=data_time,
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=iter_time,
                            data=data_time,
                        )
                    )
            i += 1
            start_time = time.time()
