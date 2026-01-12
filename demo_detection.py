"""
DINOv3 骨干网络权重加载与目标检测演示脚本

本脚本演示如何使用本地预训练的 DINOv3 ViT-B/16 权重文件进行目标检测。
主要功能包括：
1. 加载本地权重文件创建骨干网络模型
2. 构建基于 Deformable DETR 架构的目标检测器
3. 对输入图像进行预处理（resize、归一化）
4. 执行目标检测推理
5. 可视化检测结果

依赖项：
- torch: 深度学习框架
- torchvision: 图像处理工具
- PIL: Python Imaging Library
- cv2: OpenCV 图像处理
- dinov3: DINOv3 模型库

使用示例：
    python demo_detection.py
"""

# 导入必要的系统库，用于文件路径操作和模块路径管理
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

# 导入深度学习框架和图像处理库
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

# 将当前脚本所在目录添加到 Python 路径，确保能够正确导入 dinov3 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入 DINOv3 模型相关模块
import dinov3.hub.backbones as backbones
from dinov3.hub.backbones import Weights as BackboneWeights
from dinov3.models.vision_transformer import DinoVisionTransformer
from dinov3.hub.detectors import (
    DetectionWeights,
    DetectorWithProcessor,
    dinov3_vit7b16_de,
)
from dinov3.eval.detection.util.misc import NestedTensor


# ============================================================================
# 全局常量定义
# ============================================================================

# DINOv3 ViT-B/16 预训练权重文件路径
BACKBONE_WEIGHTS_PATH = "/home/wh/fj/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

# COCO 数据集类别名称映射（91 个类别，包含背景类）
# DINOv3 检测器在 COCO 数据集上训练，类别索引从 1 开始（0 为背景）
COCO_CATEGORY_NAMES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 12: "stop sign", 13: "parking meter", 14: "bench",
    15: "bird", 16: "cat", 17: "dog", 18: "horse", 19: "sheep",
    20: "cow", 21: "elephant", 22: "bear", 23: "zebra", 24: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 45: "wine glass", 46: "cup", 47: "fork",
    48: "knife", 49: "spoon", 50: "bowl", 51: "banana", 52: "apple",
    53: "sandwich", 54: "orange", 55: "broccoli", 56: "carrot", 57: "hot dog",
    58: "pizza", 59: "donut", 60: "cake", 61: "chair", 62: "couch",
    63: "potted plant", 64: "bed", 65: "dining table", 66: "toilet", 67: "tv",
    68: "laptop", 69: "mouse", 70: "remote", 71: "keyboard", 72: "cell phone",
    73: "microwave", 74: "oven", 75: "toaster", 76: "sink", 77: "refrigerator",
    78: "book", 79: "clock", 80: "vase", 81: "scissors", 82: "teddy bear",
    83: "hair drier", 84: "toothbrush"
}


def load_backbone(weights_path: str) -> DinoVisionTransformer:
    """
    加载 DINOv3 骨干网络模型

    该函数使用指定的本地权重文件创建 DINOv3 ViT-B/16 骨干网络模型。
    模型加载后会自动进入评估模式（eval），适用于推理场景。

    参数：
        weights_path: str，本地权重文件的路径

    返回：
        DinoVisionTransformer: 已加载权重的骨干网络模型实例
    """
    print(f"加载骨干网络权重: {weights_path}")

    backbone = backbones.dinov3_vitb16(
        pretrained=False,
        weights=weights_path
    )

    backbone.eval()
    print(f"骨干网络加载完成")
    print(f"  - patch_size: {backbone.patch_size}")
    print(f"  - embed_dim: {backbone.embed_dim}")
    print(f"  - n_blocks: {backbone.n_blocks}")
    print(f"  - num_heads: {backbone.num_heads}")

    return backbone


def build_detector(
    detector_weights_path: Optional[str] = None
) -> Optional[DetectorWithProcessor]:
    """
    构建完整的目标检测器

    该函数将 DINOv3 骨干网络与 Deformable DETR 检测头集成。
    检测器使用两阶段架构，包含区域提议网络和边界框精调。

    参数：
        detector_weights_path: 可选的检测头权重文件路径

    返回：
        DetectorWithProcessor: 完整的目标检测器实例，如果构建失败则返回 None
    """
    print("\n构建目标检测器...")
    print("=" * 60)
    print("检测器架构：Deformable DETR (Plain-DETR)")
    print("=" * 60)

    try:
        from dinov3.hub.detectors import DetectionWeights, dinov3_vit7b16_de
        from dinov3.hub.backbones import Weights as BackboneWeights

        print("尝试加载检测器...")
        
        # 构建检测器，使用本地权重文件和模拟骨干网络
        print(f"加载本地检测器权重: {detector_weights_path}")
        
        # 先构建一个模拟的检测器，然后替换其权重
        detector = dinov3_vit7b16_de(
            pretrained=False,  # 不加载预训练权重
            weights=detector_weights_path,
            backbone_weights=BackboneWeights.LVD1689M,
        )
        
        # 然后手动加载本地权重文件
        import torch
        state_dict = torch.load(detector_weights_path, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        detector.load_state_dict(state_dict, strict=False)
        detector.eval()

        print("检测器构建成功！")
        return detector

    except Exception as e:
        print(f"加载检测器失败: {e}")
        print("\n检测器加载失败原因分析：")
        print("1. 网络连接问题：检测头权重需要从官方服务器下载")
        print("2. 权重文件格式不匹配：需要对应骨干网络架构的检测头")
        print("3. 权重文件路径错误：请检查文件是否存在")
        return None





def preprocess_image(image_path: str, target_size: int = 224) -> torch.Tensor:
    """
    图像预处理函数

    将输入图像转换为 DINOv3 模型所需的张量格式。

    参数：
        image_path: 输入图像的文件路径
        target_size: 目标图像尺寸

    返回：
        torch.Tensor: 预处理后的图像张量 [1, C, H, W]
    """
    print(f"预处理图像: {image_path}")

    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image)

    return tensor.unsqueeze(0)


def load_original_image(image_path: str) -> Image.Image:
    """
    加载原始图像（不进行预处理）

    参数：
        image_path: 图像文件路径

    返回：
        Image.Image: 原始 PIL 图像对象
    """
    return Image.open(image_path).convert("RGB")


def visualize_detections(
    image: Image.Image,
    detections: List[Dict],
    output_path: str,
    score_threshold: float = 0.3
) -> None:
    """
    可视化目标检测结果

    在图像上绘制检测框、类别标签和置信度分数。

    参数：
        image: 原始 PIL 图像
        detections: 检测结果列表
        output_path: 输出图像路径
        score_threshold: 显示检测结果的置信度阈值
    """
    draw = ImageDraw.Draw(image)

    image_width, image_height = image.size

    colors = {
        "person": "#FF6B6B",
        "car": "#4ECDC4",
        "dog": "#45B7D1",
        "cat": "#96CEB4",
        "chair": "#FFEAA7",
        "bottle": "#DDA0DD",
        "potted plant": "#98D8C8",
        "tv": "#F7DC6F",
        "dining table": "#BB8FCE",
        "bed": "#85C1E9",
    }

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()

    detection_count = 0

    for detection in detections:
        scores = detection["scores"]
        labels = detection["labels"]
        boxes = detection["boxes"]

        for i in range(len(scores)):
            score = scores[i].item()
            if score < score_threshold:
                continue

            label_id = labels[i].item()
            x1, y1, x2, y2 = boxes[i].tolist()

            x1 = int(x1 * image_width / 224)
            y1 = int(y1 * image_height / 224)
            x2 = int(x2 * image_width / 224)
            y2 = int(y2 * image_height / 224)

            x1 = max(0, min(x1, image_width - 1))
            y1 = max(0, min(y1, image_height - 1))
            x2 = max(x1 + 1, min(x2, image_width))
            y2 = max(y1 + 1, min(y2, image_height))

            category = COCO_CATEGORY_NAMES.get(label_id, f"class_{label_id}")
            color = colors.get(category, "#FFFFFF")

            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            label_text = f"{category}: {score:.2f}"
            text_bbox = draw.textbbox((x1, y1), label_text, font=font)
            text_bg = [
                text_bbox[0] - 2,
                text_bbox[1] - 2,
                text_bbox[2] + 2,
                text_bbox[3] + 2,
            ]
            draw.rectangle(text_bg, fill=color)
            draw.text((x1, y1), label_text, fill="black", font=font)

            detection_count += 1

    image.save(output_path)
    print(f"检测结果已保存至: {output_path}")
    print(f"检测到的目标数量: {detection_count}")


def run_detection(
    detector,
    image_tensor: torch.Tensor,
    original_image: Image.Image,
    output_dir: str,
    image_name: str = None
) -> List[Dict]:
    """
    执行目标检测并可视化结果

    参数：
        detector: 目标检测器
        image_tensor: 预处理后的图像张量
        original_image: 原始图像
        output_dir: 输出目录
        image_name: 原始图像名称，用于生成结果文件名

    返回：
        List[Dict]: 检测结果列表
    """
    print("\n执行目标检测...")

    start_time = time.time()

    # 使用正确的方式调用检测器
    try:
        from dinov3.eval.detection.util.misc import NestedTensor
        
        # 将图像张量转换为 NestedTensor
        image_tensor_squeezed = image_tensor.squeeze(0)
        mask = torch.ones(image_tensor_squeezed.shape[1:], dtype=torch.bool, device=image_tensor_squeezed.device)
        nested_tensor = NestedTensor(tensors=image_tensor_squeezed, mask=mask)
        
        # 调用检测器
        detections = detector([nested_tensor])
    except Exception as e:
        print(f"  使用 NestedTensor 时出错: {e}")
        print("  尝试使用普通张量格式...")
        # 尝试使用普通张量格式
        detections = detector([image_tensor.squeeze(0)])

    inference_time = time.time() - start_time
    print(f"推理时间: {inference_time:.3f} 秒")

    os.makedirs(output_dir, exist_ok=True)
    
    # 生成输出文件名
    if image_name:
        base_name = os.path.splitext(os.path.basename(image_name))[0]
        output_path = os.path.join(output_dir, f"{base_name}_detection.jpg")
    else:
        output_path = os.path.join(output_dir, "detection_result.jpg")

    visualize_detections(original_image.copy(), detections, output_path, score_threshold=0.3)

    return detections


def print_detection_summary(detections: List[Dict]) -> None:
    """
    打印检测结果摘要

    参数：
        detections: 检测结果列表
    """
    print("\n" + "=" * 60)
    print("检测结果摘要")
    print("=" * 60)

    category_counts = {}

    for detection in detections:
        scores = detection["scores"]
        labels = detection["labels"]

        for i in range(len(scores)):
            score = scores[i].item()
            if score < 0.3:
                continue

            label_id = labels[i].item()
            category = COCO_CATEGORY_NAMES.get(label_id, f"未知类别 ({label_id})")

            if category not in category_counts:
                category_counts[category] = []
            category_counts[category].append(score)

    print(f"{'类别':<20} {'数量':<8} {'最高置信度':<12} {'平均置信度':<12}")
    print("-" * 52)

    for category, scores in sorted(category_counts.items(), key=lambda x: -len(x[1])):
        count = len(scores)
        max_score = max(scores)
        avg_score = sum(scores) / count
        print(f"{category:<20} {count:<8} {max_score:.4f}       {avg_score:.4f}")

    print("-" * 52)
    print(f"{'总计':<20} {sum(len(s) for s in category_counts.values()):<8}")
    print("=" * 60)


def main():
    """
    主函数：演示 DINOv3 目标检测流程

    执行步骤：
    1. 加载 DINOv3 骨干网络权重
    2. 构建目标检测器
    3. 加载并预处理测试图像
    4. 执行目标检测
    5. 可视化并保存检测结果
    """
    print("=" * 60)
    print("DINOv3 目标检测演示")
    print("=" * 60)

    # 配置参数
    detector_weights_path = "/home/wh/fj/dinov3/dinov3_vit7b16_coco_detr_head-b0235ff7.pth"
    image_dir = "/home/wh/fj/Datasets/visdrone/images"
    output_dir = "/home/wh/fj/dinov3/detection_results"
    target_size = 224

    # 构建检测器，使用本地权重文件
    detector = build_detector(detector_weights_path)

    if detector is None:
        print("错误：无法构建目标检测器")
        print("请检查权重文件路径是否正确，或者网络连接是否正常")
        return

    print("\n" + "-" * 60)
    print("目标检测测试")
    print("-" * 60)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取图像目录中的所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    if os.path.exists(image_dir):
        for file in os.listdir(image_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(image_dir, file))
        
        print(f"找到 {len(image_files)} 张图像")
        
        # 对每张图像执行检测
        for i, image_path in enumerate(image_files):
            print(f"\n处理图像 {i+1}/{len(image_files)}: {image_path}")
            
            try:
                # 预处理图像
                image_tensor = preprocess_image(image_path, target_size=target_size)
                print(f"  输入图像形状: {image_tensor.shape}")
                
                # 加载原始图像
                original_image = load_original_image(image_path)
                print(f"  原始图像尺寸: {original_image.size}")
                
                # 执行检测
                detections = run_detection(detector, image_tensor, original_image, output_dir, image_path)
                
                # 打印检测结果摘要
                print_detection_summary(detections)
                
            except Exception as e:
                print(f"  处理图像时出错: {e}")
                continue
    else:
        print(f"图像目录不存在: {image_dir}")

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
