#!/usr/bin/env python3
"""
DINOv3特征可视化脚本
用于生成README中展示的彩虹色特征可视化效果
"""

import os
import argparse
import urllib.request
import tarfile
import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from scipy import signal
import torch
import torchvision.transforms.functional as TF

# 配置参数
PATCH_SIZE = 16
IMAGE_SIZE = 768
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 模型名称和路径
MODEL_NAME = "dinov3_vitb16"
MODEL_PATH = "/home/wh/fj/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

# 数据集URL
IMAGES_URI = "https://dl.fbaipublicfiles.com/dinov3/notebooks/foreground_segmentation/foreground_segmentation_images.tar.gz"
LABELS_URI = "https://dl.fbaipublicfiles.com/dinov3/notebooks/foreground_segmentation/foreground_segmentation_labels.tar.gz"

# 保存路径
SAVE_DIR = "/home/wh/fj/dinov3/visualization_results"
os.makedirs(SAVE_DIR, exist_ok=True)


def load_images_from_remote_tar(tar_uri: str) -> list[Image.Image]:
    """从远程tar.gz文件加载图像"""
    print(f"Loading images from {tar_uri}...")
    images = []
    with urllib.request.urlopen(tar_uri) as f:
        tar = tarfile.open(fileobj=io.BytesIO(f.read()))
        for member in tar.getmembers():
            if member.isfile():
                image_data = tar.extractfile(member)
                image = Image.open(image_data)
                images.append(image)
    print(f"Loaded {len(images)} images")
    return images


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
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    return image


def resize_transform(
    mask_image: Image,
    image_size: int = IMAGE_SIZE,
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    """将图像调整为patch_size的整数倍"""
    w, h = mask_image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))


def load_dinov3_model():
    """加载DINOv3模型"""
    print(f"Loading DINOv3 model: {MODEL_NAME}...")
    
    # 加载本地模型
    model = torch.hub.load(
        repo_or_dir=".",
        model=MODEL_NAME,
        source="local",
        weights=MODEL_PATH
    )
    
    # 移动到GPU（如果可用）
    if torch.cuda.is_available():
        model.cuda()
        print("Model moved to GPU")
    else:
        print("GPU not available, using CPU")
    
    return model


def train_foreground_classifier(model):
    """训练前景分割分类器"""
    print("\n=== 训练前景分割分类器 ===")
    
    # 加载训练数据
    images = load_images_from_remote_tar(IMAGES_URI)
    labels = load_images_from_remote_tar(LABELS_URI)
    n_images = len(images)
    assert n_images == len(labels), f"图像和标签数量不匹配: {len(images)} vs {len(labels)}"
    
    # 创建量化滤波器
    patch_quant_filter = torch.nn.Conv2d(1, 1, PATCH_SIZE, stride=PATCH_SIZE, bias=False)
    patch_quant_filter.weight.data.fill_(1.0 / (PATCH_SIZE * PATCH_SIZE))
    
    # 模型层数
    MODEL_TO_NUM_LAYERS = {
        "dinov3_vits16": 12,
        "dinov3_vits16plus": 12,
        "dinov3_vitb16": 12,
        "dinov3_vitl16": 24,
        "dinov3_vith16plus": 32,
        "dinov3_vit7b16": 40,
    }
    n_layers = MODEL_TO_NUM_LAYERS[MODEL_NAME]
    
    xs = []
    ys = []
    
    print("提取特征和标签...")
    with torch.inference_mode():
        with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32):
            for i in range(n_images):
                # 加载标签
                mask_i = labels[i].split()[-1] if labels[i].mode == 'RGBA' else labels[i]
                mask_i_resized = resize_transform(mask_i)
                mask_i_quantized = patch_quant_filter(mask_i_resized).squeeze().view(-1).detach().cpu()
                ys.append(mask_i_quantized)
                
                # 加载图像
                image_i = images[i].convert('RGB')
                image_i_resized = resize_transform(image_i)
                image_i_resized = TF.normalize(image_i_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
                image_i_resized = image_i_resized.unsqueeze(0).cuda() if torch.cuda.is_available() else image_i_resized.unsqueeze(0)
                
                # 提取特征
                feats = model.get_intermediate_layers(image_i_resized, n=range(n_layers), reshape=True, norm=True)
                dim = feats[-1].shape[1]
                xs.append(feats[-1].squeeze().view(dim, -1).permute(1, 0).detach().cpu())
    
    # 合并所有数据
    xs = torch.cat(xs)
    ys = torch.cat(ys)
    
    # 只保留明确的前景/背景标签
    idx = (ys < 0.01) | (ys > 0.99)
    xs = xs[idx]
    ys = ys[idx]
    
    print(f"设计矩阵大小: {xs.shape}")
    print(f"标签矩阵大小: {ys.shape}")
    
    # 训练逻辑回归模型
    print("训练逻辑回归分类器...")
    clf = LogisticRegression(random_state=0, C=0.1, max_iter=10000)
    clf.fit(xs.numpy(), (ys > 0.5).long().numpy())
    
    print("前景分类器训练完成!")
    return clf


def visualize_pca(model, clf, image_path="https://dl.fbaipublicfiles.com/dinov3/notebooks/pca/test_image.jpg"):
    """可视化PCA特征"""
    print("\n=== 生成PCA可视化 ===")

    # 加载测试图像
    image = load_image(image_path)
    
    # 图像预处理
    image_resized = resize_transform(image)
    image_resized_norm = TF.normalize(image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    
    # 模型层数
    MODEL_TO_NUM_LAYERS = {
        "dinov3_vits16": 12,
        "dinov3_vits16plus": 12,
        "dinov3_vitb16": 12,
        "dinov3_vitl16": 24,
        "dinov3_vith16plus": 32,
        "dinov3_vit7b16": 40,
    }
    n_layers = MODEL_TO_NUM_LAYERS[MODEL_NAME]
    
    # 提取特征
    print("提取图像特征...")
    with torch.inference_mode():
        with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32):
            # 添加batch维度并移动到设备
            input_tensor = image_resized_norm.unsqueeze(0)
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            
            feats = model.get_intermediate_layers(input_tensor, n=range(n_layers), reshape=True, norm=True)
            x = feats[-1].squeeze().detach().cpu()
            dim = x.shape[0]
            x = x.view(dim, -1).permute(1, 0)
    
    # 计算前景概率
    print("计算前景概率...")
    h_patches, w_patches = [int(d / PATCH_SIZE) for d in image_resized.shape[1:]]
    fg_score = clf.predict_proba(x.numpy())[:, 1].reshape(h_patches, w_patches)
    fg_score_mf = torch.from_numpy(signal.medfilt2d(fg_score, kernel_size=3))
    
    # 提取前景补丁
    foreground_selection = fg_score_mf.view(-1) > 0.5
    fg_patches = x[foreground_selection]
    
    # 拟合PCA
    print("拟合PCA...")
    pca = PCA(n_components=3, whiten=True)
    pca.fit(fg_patches.numpy())
    
    # 应用PCA
    print("生成可视化图像...")
    projected_image = torch.from_numpy(pca.transform(x.numpy())).view(h_patches, w_patches, 3)
    
    # 增强颜色
    projected_image = torch.nn.functional.sigmoid(projected_image.mul(2.0)).permute(2, 0, 1)
    
    # 遮罩背景
    projected_image *= (fg_score_mf.unsqueeze(0) > 0.5)
    
    # 可视化结果
    print("保存可视化结果...")
    
    # 原始图像
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title("原始图像")
    plt.axis('off')
    
    # 前景分数
    plt.subplot(2, 2, 2)
    plt.imshow(fg_score_mf)
    plt.title("前景分数")
    plt.axis('off')
    plt.colorbar()
    
    # PCA可视化
    plt.subplot(2, 2, 3)
    plt.imshow(projected_image.permute(1, 2, 0))
    plt.title("DINOv3特征PCA可视化")
    plt.axis('off')
    
    # 合并显示
    plt.subplot(2, 2, 4)
    plt.imshow(image)
    plt.imshow(projected_image.permute(1, 2, 0), alpha=0.7)
    plt.title("叠加效果")
    plt.axis('off')
    
    # 保存结果
    save_path = os.path.join(SAVE_DIR, "dinov3_pca_visualization.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化结果已保存到: {save_path}")
    
    return save_path


def main():
    """主函数"""
    print("DINOv3特征可视化脚本")
    print("=" * 50)

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='DINOv3特征可视化工具')
    parser.add_argument(
        '--image',
        type=str,
        default="https://dl.fbaipublicfiles.com/dinov3/notebooks/pca/test_image.jpg",
        help='要可视化的图像路径或URL（默认使用官方测试图片）'
    )
    args = parser.parse_args()

    print(f"使用图像: {args.image}")

    # 步骤1: 加载DINOv3模型
    model = load_dinov3_model()

    # 步骤2: 训练前景分类器
    clf = train_foreground_classifier(model)

    # 步骤3: 生成PCA可视化
    save_path = visualize_pca(model, clf, image_path=args.image)

    print("\n" + "=" * 50)
    print("可视化完成!")
    print(f"结果文件: {save_path}")
    print("您可以使用以下命令查看结果:")
    print(f"  display {save_path}")
    print(f"或使用图像查看器打开: {save_path}")


if __name__ == "__main__":
    main()
