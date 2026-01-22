import torch
from dinov3.hub.backbones import dinov3_vitb16

model = dinov3_vitb16(pretrained=True,
    weights='/home/wh/fj/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
model.eval()

example_input = torch.randn(1, 3, 224, 224)

# 方式1: TorchScript (推荐)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("dinov3_vitb16.pt")

# 方式2: 单独保存 weights
torch.save(model.state_dict(), "dinov3_vitb16_weights.pth")

print("导出完成: dinov3_vitb16.pt")