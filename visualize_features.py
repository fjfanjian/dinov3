#!/usr/bin/env python3
"""
DINOv3特征可视化脚本
生成README中展示的彩虹色PCA特征可视化效果
"""

import os
import argparse
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from scipy import signal
import torch
import torchvision.transforms.functional as TF


# 配置参数
PATCH_SIZE = 16
IMAGE_SIZE = 768
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 模型列表
MODELS = {
    "dinov3_vits16": "DINOv3 ViT-S/16",
    "dinov3_vits16plus": "DINOv3 ViT-S+/16",
    "dinov3_vitb16": "DINOv3 ViT-B/16",
    "dinov3_vitl16": "DINOv3 ViT-L/16",
    "dinov3_vith16plus": "DINOv3 ViT-H+/16",
    "dinov3_vit7b16": "DINOv3 ViT-7B/16",
}

# 模型对应的层数
MODEL_TO_NUM_LAYERS = {
    "dinov3_vits16": 12,
    "dinov3_vits16plus": 12,
    "dinov3_vitb16": 12,
    "dinov3_vitl16": 24,
    "dinov3_vith16plus": 32,
    "dinov3_vit7b16": 40,
}


def load_image(image_path: str) -> Image.Image:
    """
    加载图像，支持本地文件路径和URL

    Args:
        image_path: 本地文件路径或URL

    Returns:
        PIL.Image.Image: 加载的RGB图像
    """
    if image_path.startswith(('http://', 'https://')):
        # 从URL加载
        print(f"从URL加载图像: {image_path}")
        with urllib.request.urlopen(image_path) as f:
            image = Image.open(f).convert("RGB")
    elif os.path.isfile(image_path):
        # 从本地文件加载
        print(f"从本地文件加载图像: {image_path}")
        image = Image.open(image_path).convert("RGB")
    else:
        raise ValueError(f"图像路径不存在: {image_path}")

    return image


def resize_transform(
    image: Image.Image,
    image_size: int = IMAGE_SIZE,
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    """
    将图像调整为patch大小的倍数

    Args:
        image: 输入图像
        image_size: 目标图像大小
        patch_size: patch大小

    Returns:
        torch.Tensor: 调整大小后的图像张量
    """
    w, h = image.size
    h_patches = image_size // patch_size
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(TF.resize(image, (h_patches * patch_size, w_patches * patch_size)))


def load_model(model_name: str, weights_path: Optional[str] = None, device: str = "cuda"):
    """
    加载DINOv3模型

    Args:
        model_name: 模型名称
        weights_path: 模型权重路径（可选）
        device: 设备（cuda/cpu）

    Returns:
        model: 加载的模型
    """
    print(f"加载模型: {MODELS.get(model_name, model_name)}")

    # 使用当前目录作为DINOv3仓库路径
    repo_dir = "."

    # 尝试查找默认权重文件
    if weights_path is None:
        # 常见权重文件名格式: dinov3_{model_name}_pretrain_*.pth
        possible_patterns = [
            f"dinov3_{model_name}_pretrain_*.pth",
            f"dinov3_{model_name}_*.pth",
            f"{model_name}*.pth"
        ]

        import glob
        for pattern in possible_patterns:
            matches = glob.glob(pattern)
            if matches:
                weights_path = matches[0]
                print(f"自动找到权重文件: {weights_path}")
                break

    if weights_path and os.path.exists(weights_path):
        print(f"使用权重文件: {weights_path}")
    else:
        print(f"警告: 未找到权重文件 {weights_path}")
        print("你可以从以下地址下载权重:")
        print("https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/")
        weights_path = None

    model = torch.hub.load(
        repo_or_dir=repo_dir,
        model=model_name,
        source="local",
        weights=weights_path,
    )

    model.to(device)
    model.eval()

    return model


def extract_features(
    model,
    image: torch.Tensor,
    device: str = "cuda",
    n_last_layers: int = 1
) -> torch.Tensor:
    """
    从图像中提取DINOv3特征

    Args:
        model: DINOv3模型
        image: 输入图像张量
        device: 设备
        n_last_layers: 使用最后几层的特征

    Returns:
        torch.Tensor: 提取的特征 (num_patches, feature_dim)
    """
    with torch.inference_mode():
        with torch.autocast(device_type=device, dtype=torch.float32):
            # 使用最后一层的输出特征
            feats = model.get_intermediate_layers(
                image.unsqueeze(0).to(device),
                n=n_last_layers,
                reshape=True,
                norm=True
            )
            # 取最后一层的特征
            x = feats[-1].squeeze().detach().cpu()
            # reshape: (dim, h, w) -> (num_patches, dim)
            dim = x.shape[0]
            x = x.view(dim, -1).permute(1, 0)

    return x


def load_foreground_classifier(model_path: Optional[str] = None):
    """
    加载前景分类器（可选）

    Args:
        model_path: 分类器路径

    Returns:
        classifier: 加载的分类器或None
    """
    if model_path and os.path.exists(model_path):
        import pickle
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
        print(f"已加载前景分类器: {model_path}")
        return clf
    return None


def create_foreground_mask(
    features: torch.Tensor,
    image_shape: Tuple[int, int],
    classifier=None,
    patch_size: int = PATCH_SIZE
) -> torch.Tensor:
    """
    创建前景掩码

    Args:
        features: 特征张量 (num_patches, feature_dim)
        image_shape: 图像形状 (height, width)
        classifier: 前景分类器（可选）
        patch_size: patch大小

    Returns:
        torch.Tensor: 前景掩码
    """
    h_patches, w_patches = [d // patch_size for d in image_shape]

    if classifier is not None:
        # 使用分类器预测前景概率
        fg_score = classifier.predict_proba(features.numpy())[:, 1]
        fg_score = fg_score.reshape(h_patches, w_patches)
        # 应用中值滤波平滑
        fg_score_mf = torch.from_numpy(signal.medfilt2d(fg_score, kernel_size=3))
        # 创建二值掩码
        mask = fg_score_mf > 0.5
    else:
        # 如果没有分类器，使用所有patches
        mask = torch.ones((h_patches, w_patches), dtype=torch.bool)

    return mask


def visualize_pca_features(
    features: torch.Tensor,
    mask: torch.Tensor,
    image_shape: Tuple[int, int],
    patch_size: int = PATCH_SIZE,
    save_path: Optional[str] = None,
    raw_output: bool = False
) -> np.ndarray:
    """
    可视化PCA特征（彩虹色效果）

    Args:
        features: 特征张量 (num_patches, feature_dim)
        mask: 前景掩码
        image_shape: 图像形状 (height, width)
        patch_size: patch大小
        save_path: 保存路径
        raw_output: 是否使用原始输出（像素点对点，无框架）

    Returns:
        np.ndarray: 可视化结果
    """
    h_patches, w_patches = [d // patch_size for d in image_shape]

    # 只在前景区域拟合PCA
    foreground_selection = mask.view(-1)
    if foreground_selection.sum() > 0:
        fg_patches = features[foreground_selection]
    else:
        fg_patches = features

    # 拟合PCA（3个主成分）
    print("拟合PCA...")
    pca = PCA(n_components=3, whiten=True)
    pca.fit(fg_patches.numpy())

    # 应用PCA投影
    projected_image = torch.from_numpy(
        pca.transform(features.numpy())
    ).view(h_patches, w_patches, 3)

    # 乘以2.0并通过sigmoid函数获得鲜艳的颜色
    projected_image = torch.nn.functional.sigmoid(
        projected_image.mul(2.0)
    ).permute(2, 0, 1)

    # 使用掩码将背景设为黑色
    if mask.sum() > 0:
        projected_image *= mask.unsqueeze(0).float()

    if raw_output:
        # 原始输出：像素点对点，无框架无标题
        # 转换为PIL图像并保存
        print("使用原始输出模式（像素点对点）...")

        # 转换为numpy数组 (H, W, C)
        image_array = projected_image.permute(1, 2, 0).numpy()

        # 转换为uint8格式（0-255）
        image_uint8 = (image_array * 255).astype(np.uint8)

        # 转换为PIL图像
        image_pil = Image.fromarray(image_uint8, mode='RGB')

        if save_path:
            # 确保输出目录存在
            output_dir = os.path.dirname(save_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # 保存图像（无压缩）
            image_pil.save(save_path, format='PNG', optimize=False, compress_level=0)
            print(f"原始PCA图像已保存: {save_path}")
            print(f"  - 分辨率: {image_pil.width} x {image_pil.height} 像素")
            print(f"  - 模式: RGB，无框架，无标题")
            print(f"  - 文件大小: {os.path.getsize(save_path) / 1024:.1f} KB")

        return image_array

    else:
        # 使用matplotlib渲染（带框架和标题）
        print("使用matplotlib渲染模式...")

        plt.figure(figsize=(12, 6), dpi=150)

        # PCA可视化
        plt.imshow(projected_image.permute(1, 2, 0))
        plt.axis('off')
        plt.title('PCA of DINOv3 Patch Features (Rainbow Visualization)', fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"可视化结果已保存: {save_path}")

        plt.close()

        return projected_image.permute(1, 2, 0).numpy()


def process_single_image(
    image_path: str,
    model,
    args,
    output_dir: str
) -> bool:
    """
    处理单个图像

    Args:
        image_path: 输入图像路径
        model: DINOv3模型
        args: 命令行参数
        output_dir: 输出目录

    Returns:
        bool: 是否成功
    """
    try:
        print(f"\n{'='*60}")
        print(f"处理图像: {os.path.basename(image_path)}")
        print(f"{'='*60}")

        # 加载图像
        image = load_image(image_path)
        print(f"原始图像大小: {image.size}")

        # 调整图像大小
        image_resized = resize_transform(image, args.image_size, PATCH_SIZE)
        print(f"调整后图像大小: {image_resized.shape}")

        # 归一化
        image_normalized = TF.normalize(image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        # 提取特征
        features = extract_features(model, image_normalized, args.device)
        print(f"特征形状: {features.shape}")

        # 加载前景分类器（可选）
        classifier = load_foreground_classifier(args.classifier)

        # 创建前景掩码
        mask = create_foreground_mask(
            features,
            image_resized.shape[1:],
            classifier,
            PATCH_SIZE
        )
        print(f"前景patches数量: {mask.sum().item()}/{mask.numel()}")

        # 确定输出路径
        if args.output_dir:
            # 保持原文件名，添加_pca后缀
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_pca.png")
        else:
            output_path = args.output

        # 可视化PCA特征
        print("生成PCA可视化...")
        visualize_pca_features(
            features,
            mask,
            image_resized.shape[1:],
            PATCH_SIZE,
            output_path,
            raw_output=args.raw_output
        )

        print(f"\n✓ 完成: {os.path.basename(output_path)}")
        return True

    except Exception as e:
        print(f"\n✗ 处理失败: {os.path.basename(image_path)}")
        print(f"错误: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="DINOv3特征可视化脚本 - 生成彩虹色PCA可视化效果"
    )
    # 输入选项（互斥）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        type=str,
        help="输入图像路径或URL"
    )
    input_group.add_argument(
        "--input-dir",
        type=str,
        help="输入图像目录路径"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="dinov3_vitl16",
        choices=list(MODELS.keys()),
        help="DINOv3模型名称"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="输出图像路径（单图像模式）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="输出目录路径（批量处理模式）"
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default=None,
        help="前景分类器路径（可选）"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=IMAGE_SIZE,
        help="输入图像大小"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备（cuda/cpu）"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="显示可视化结果（仅单图像模式）"
    )
    parser.add_argument(
        "--image-ext",
        type=str,
        default="jpg,jpeg,png",
        help="图像文件扩展名（批量处理，逗号分隔）"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="最大处理图像数量（批量处理模式，按文件名排序）"
    )
    parser.add_argument(
        "--raw-output",
        action="store_true",
        help="原始输出模式（像素点对点，无框架无标题）"
    )

    args = parser.parse_args()

    # 加载模型（只需要加载一次）
    print("=" * 60)
    print("DINOv3 特征可视化")
    print("=" * 60)
    print(f"模型: {MODELS[args.model]}")
    print(f"设备: {args.device}")
    print("=" * 60)

    model = load_model(args.model, device=args.device)

    # 判断是单图像模式还是批量处理模式
    if args.image:
        # 单图像模式
        if args.output is None:
            # 自动生成输出路径，保持原文件名
            base_name = os.path.splitext(os.path.basename(args.image))[0]
            args.output = f"{base_name}_pca.png"

        # 创建输出目录
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 处理单个图像
        process_single_image(args.image, model, args, output_dir)

        if args.show:
            plt.show()

    elif args.input_dir:
        # 批量处理模式
        if args.output_dir is None:
            args.output_dir = os.path.join(args.input_dir, "visualization_results")

        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"输出目录: {args.output_dir}")

        # 获取支持的图像扩展名
        supported_exts = [ext.strip().lower() for ext in args.image_ext.split(',')]
        supported_exts = [ext if ext.startswith('.') else f'.{ext}' for ext in supported_exts]

        # 查找所有图像文件
        image_files = []
        for file in os.listdir(args.input_dir):
            if any(file.lower().endswith(ext) for ext in supported_exts):
                image_files.append(os.path.join(args.input_dir, file))

        if not image_files:
            print(f"✗ 在目录 {args.input_dir} 中未找到支持的图像文件")
            print(f"支持的扩展名: {', '.join(supported_exts)}")
            return

        # 按文件名排序
        image_files.sort()

        # 应用最大数量限制
        total_files = len(image_files)
        if args.max_images is not None and args.max_images > 0:
            image_files = image_files[:args.max_images]
            print(f"找到 {total_files} 个图像文件，将处理前 {len(image_files)} 个")
        else:
            print(f"找到 {len(image_files)} 个图像文件")
        print("=" * 60)

        # 批量处理
        success_count = 0
        for i, image_path in enumerate(image_files, 1):
            print(f"\n进度: [{i}/{len(image_files)}]")
            if process_single_image(image_path, model, args, args.output_dir):
                success_count += 1

        # 总结
        print("\n" + "=" * 60)
        print(f"批量处理完成!")
        print(f"成功: {success_count}/{len(image_files)} 个图像")
        print(f"结果保存在: {args.output_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
