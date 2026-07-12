# =============================================================================
# mcunet.tinynas 包初始化文件
#
# 本包是 MCUNet 项目中 TinyNAS（Tiny Neural Architecture Search）的核心实现。
# TinyNAS 是一个专门针对微控制器（MCU）等资源受限设备设计的神经架构搜索框架。
#
# tf_codebase 子包提供了将 PyTorch 训练的模型转换为 TensorFlow / TFLite 格式的工具链，
# 使得模型可以部署到 TensorFlow Lite Micro 后端（如 ARM Cortex-M 系列 MCU）上运行。
#
# 主要组件：
#   - tf_layers.py:    TensorFlow 层定义（Conv2D、DepthwiseConv、BN、激活函数等）
#   - tf_modules.py:   高级模块（MBInvertedBlock、ProxylessNASNets 完整网络）
#   - generate_tflite.py:  PyTorch → TFLite 的完整转换 + 量化管线
#   - tf_model_zoo.py:  预训练模型加载工具
# =============================================================================
