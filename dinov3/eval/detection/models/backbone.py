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
Backbone modules.
"""
import logging
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from ..util.misc import NestedTensor
from .position_encoding import build_position_encoding
from .utils import LayerNorm2D
from .windows import WindowsWrapper

logger = logging.getLogger("dinov3")


class DINOBackbone(nn.Module):
    """DINOv3 骨干网络适配器

    将 DINOv3 骨干网络的输出转换为检测器所需的格式:
    1. 获取骨干网络的多层特征
    2. 应用 LayerNorm
    3. 拼接多层特征
    4. 封装为 NestedTensor
    """

    def __init__(
        self,
        backbone_model: nn.Module,  # DINOv3 骨干网络实例
        train_backbone: bool,        # 是否训练骨干网络
        blocks_to_train: Optional[List[str]] = None,  # 指定要训练哪些 block
        layers_to_use: Union[int, List] = 1,          # 使用最后几层特征
        use_layernorm: bool = True,   # 是否使用 LayerNorm
    ):
        super().__init__()
        self.backbone = backbone_model  # DINOv3 骨干网络
        self.blocks_to_train = blocks_to_train  # 要训练的 block 列表
        self.patch_size = self.backbone.patch_size  # patch 大小 (16)
        self.use_layernorm = use_layernorm  # 是否使用 LayerNorm

        # -------------------------------------------------------------
        # 参数冻结逻辑: 根据配置冻结/解冻骨干网络参数
        # -------------------------------------------------------------
        for _, (name, parameter) in enumerate(self.backbone.named_parameters()):
            # 如果指定了 blocks_to_train，检查当前参数是否在要训练的 block 中
            train_condition = any(f".{b}." in name for b in self.blocks_to_train) \
                if self.blocks_to_train else True

            # 冻结条件:
            # 1. train_backbone=False (不训练骨干)
            # 2. 参数名包含 "mask_token" (mask_token 是随机初始化的)
            # 3. 当前 block 不在要训练的列表中
            if (not train_backbone) or "mask_token" in name or (not train_condition):
                parameter.requires_grad_(False)

        # 设置输出特征图的 stride (即 patch_size)
        self.strides = [self.backbone.patch_size]

        # -------------------------------------------------------------
        # 计算要输出的层及其通道数
        # -------------------------------------------------------------
        n_all_layers = self.backbone.n_blocks  # ViT-B 有 12 层

        # 确定要提取哪些层的特征
        # layers_to_use=1  -> 取最后 1 层 (index 11)
        # layers_to_use=4  -> 取最后 4 层 (index 8,9,10,11)
        blocks_to_take = (
            range(n_all_layers - layers_to_use, n_all_layers)
            if isinstance(layers_to_use, int)
            else layers_to_use
        )

        # 获取每层的通道数
        # 如果模型有 embed_dims 属性则使用，否则假设所有层通道数相同
        embed_dims = getattr(
            self.backbone,
            "embed_dims",
            [self.backbone.embed_dim] * self.backbone.n_blocks
        )
        # 筛选出要使用的层的通道数
        embed_dims = [embed_dims[i] for i in range(n_all_layers) if i in blocks_to_take]

        # 为每层特征创建 LayerNorm
        if self.use_layernorm:
            self.layer_norms = nn.ModuleList([
                LayerNorm2D(embed_dim) for embed_dim in embed_dims
            ])

        # 输出通道数列表 (拼接后的总通道数)
        self.num_channels = [sum(embed_dims)]
        self.layers_to_use = layers_to_use

    def forward(self, tensor_list: NestedTensor):
        """前向传播

        Args:
            tensor_list: 包含以下属性的 NestedTensor:
                - tensors: 输入图像 [B, 3, H, W]
                - mask: 填充掩码 [B, H, W]

        Returns:
            out: NestedTensor 列表，每个包含:
                - tensors: 特征图 [B, C, H', W']
                - mask: 下采样掩码 [B, H', W']
        """
        # -------------------------------------------------------------
        # Step 1: 获取骨干网络的多层特征
        # -------------------------------------------------------------
        xs = self.backbone.get_intermediate_layers(
            tensor_list.tensors,  # 输入图像 [B, 3, H, W]
            n=self.layers_to_use,  # 输出层数
            reshape=True  # 恢复 2D 空间结构
        )
        # xs: [feat_L-4, feat_L-3, feat_L-2, feat_L-1]
        # 每个 feat: [B, 768, H', W']

        # -------------------------------------------------------------
        # Step 2: 对每层特征应用 LayerNorm
        # -------------------------------------------------------------
        if self.use_layernorm:
            xs = [ln(x).contiguous() for ln, x in zip(self.layer_norms, xs)]

        # -------------------------------------------------------------
        # Step 3: 拼接多层特征
        # 拼接后通道数 = embed_dim * layers_to_use
        # -------------------------------------------------------------
        xs = [torch.cat(xs, axis=1)]
        # xs[0]: [B, 768*4, H', W'] = [B, 3072, H', W']

        # -------------------------------------------------------------
        # Step 4: 封装为 NestedTensor 并处理掩码
        # -------------------------------------------------------------
        out: list[NestedTensor] = []
        for x in xs:
            m = tensor_list.mask  # 原始掩码 [B, H, W]
            assert m is not None  # 确保掩码存在
            # 插值掩码到特征图大小
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out.append(NestedTensor(x, mask))
        return out


class BackboneWithPositionEncoding(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        out: List[NestedTensor] = list(self[0](tensor_list))
        pos = [self[1][idx](x).to(x.tensors.dtype) for idx, x in enumerate(out)]
        return out, pos


def build_backbone(backbone_model, args):
    position_embedding = build_position_encoding(args)
    train_backbone = False
    backbone = DINOBackbone(
        backbone_model, train_backbone, args.blocks_to_train, args.layers_to_use, args.backbone_use_layernorm
    )
    if args.n_windows_sqrt > 0:
        logger.info(f"Wrapping with {args.n_windows_sqrt} x {args.n_windows_sqrt} windows")
        backbone = WindowsWrapper(
            backbone, n_windows_w=args.n_windows_sqrt, n_windows_h=args.n_windows_sqrt, patch_size=backbone.patch_size
        )
    else:
        logger.info("Not wrapping with windows")

    return BackboneWithPositionEncoding(backbone, position_embedding)
