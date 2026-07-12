# =============================================================================
# tf_codebase 子包初始化文件
#
# 本子包的主要职责：将 PyTorch 训练的 MCUNet 模型转换为 TensorFlow 格式，
# 并最终导出为 TFLite（包含 INT8 量化），以便在 MCU 设备上推理。
#
# TF → TFLite 转换管线概览：
#   1. PyTorch state_dict → TensorFlow 权重字典（permute 调整维度顺序）
#   2. TensorFlow Graph 构建（使用 tf_layers.py 中的层定义）
#   3. 权重注入 + INT8 量化校准（representative dataset）
#   4. TFLite flatbuffer 导出
#
# 从 tf_model_zoo 模块导出所有公开符号，方便外部调用。
# =============================================================================
# utilities to convert model into tf-lite format
from .tf_model_zoo import *
