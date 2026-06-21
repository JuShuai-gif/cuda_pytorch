import torch
import torchvision.models as models

model = models.resnet50()

print(model)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

model_size_mb = total_params * 4 / (1024 * 1024)  # FP32: 4 bytes per param

print(f"总参数量: {total_params:,}")
print(f"可训练参数: {trainable_params:,}")
print(f"模型大小(FP32): {model_size_mb:.2f} MB")


