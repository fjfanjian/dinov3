import torch
import torch.nn as nn
from dinov3.hub.backbones import dinov3_vitb16


class DinoV3FeatureExtractor(nn.Module):
    """DINOv3 特征提取器 - 适配目标检测器

    输出多尺度特征图 (C2, C3, C4, C5) 供 FPN 使用
    """
    def __init__(self, backbone_name: str = "vitb16", output_indices: tuple = (-1,)):
        super().__init__()

        # 加载 DINOv3 骨干网络
        if backbone_name == "vitb16":
            self.backbone = dinov3_vitb16(weights="weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
            self.embed_dim = 768
            self.patch_size = 16
            self.depth = 12
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # 冻结骨干网络参数
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.eval()  # 设为评估模式

        # 输出哪些层的特征 (默认输出最后一层)
        self.output_indices = output_indices

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: 输入图像 [B, 3, H, W]
        Returns:
            features: 多尺度特征图列表
        """
        # 转换为 eval 模式以确保 BN/LN 不更新
        self.backbone.eval()

        features = []

        # 方法1: 使用 get_intermediate_layers 获取单层特征
        if len(self.output_indices) == 1:
            output = self.backbone.get_intermediate_layers(
                x,
                n=self.output_indices[0],
                reshape=True
            )[0]
            # output: [B, 768, H', W']
            features.append(output)
        else:
            # 方法2: 获取多层特征
            outputs = self.backbone.get_intermediate_layers(
                x,
                n=max(self.output_indices) + 1,  # 获取足够多的层
                reshape=True
            )
            for idx in self.output_indices:
                features.append(outputs[idx])

        return features


class DinoV3FPNAdapter(nn.Module):
    """FPN 适配器 - 将 DINOv3 输出转换为多尺度特征

    由于 DINOv3 输出 stride 较大 (14)，需要插值或组合得到 P2-P5
    """
    def __init__(self, in_channels: int = 768, out_channels: int = 256):
        super().__init__()

        # 简单的 1x1 卷积适配通道
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1) for _ in range(4)
        ])

        # 输出卷积
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in range(4)
        ])

    def forward(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Args:
            features: DINOv3 特征列表
        Returns:
            fpn_features: FPN 多尺度特征 {P2, P3, P4, P5}
        """
        # DINOv3 输出是单尺度 14x14 (stride 16)
        # 这里的实现根据实际需求调整

        fpn_features = {}
        for i, feat in enumerate(features):
            # 1x1 卷积适配通道
            feat = self.lateral_convs[i](feat)
            fpn_features[f'P{i+2}'] = feat

        return fpn_features


# 测试
if __name__ == "__main__":
    # 1. 特征提取器测试
    extractor = DinoV3FeatureExtractor(backbone_name="vitb16")
    x = torch.randn(2, 3, 640, 640)  # 检测器常用尺寸

    with torch.no_grad():
        feats = extractor(x)
        print("DINOv3 特征输出:")
        for i, f in enumerate(feats):
            print(f"  Feature {i}: {f.shape}")

    # 2. 完整检测头示例 (伪代码)
    print("\n检测器适配示例:")
    print("""
    # 骨干网络 (冻结)
    backbone = DinoV3FeatureExtractor()
    backbone.eval()

    # FPN (可训练)
    fpn = DinoV3FPNAdapter(in_channels=768, out_channels=256)

    # 检测头 (可训练)
    detector_head = YourDetectorHead(in_channels=256)

    # 训练循环
    for images, targets in dataloader:
        with torch.no_grad():
            features = backbone(images)  # [B, 768, 40, 40]

        fpn_features = fpn(features)
        losses = detector_head(fpn_features, targets)

        losses.backward()  # 只更新 fpn 和 detector_head
    """)
