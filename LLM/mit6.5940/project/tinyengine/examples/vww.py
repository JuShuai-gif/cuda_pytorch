# ----------------------------------------------------------------------
# Project: TinyEngine
# Title:   vww_to_c.py
#
# Reference papers:
#  - MCUNet: Tiny Deep Learning on IoT Device, NeurIPS 2020
#  - MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, NeurIPS 2021
#  - MCUNetV3: On-Device Training Under 256KB Memory, NeurIPS 2022
# Contact authors:
#  - Wei-Ming Chen, wmchen@mit.edu
#  - Wei-Chen Wang, wweichen@mit.edu
#  - Ji Lin, jilin@mit.edu
#  - Ligeng Zhu, ligeng@mit.edu
#  - Song Han, songhan@mit.edu
#
# Target ISA:  ARMv7E-M
# ----------------------------------------------------------------------

from code_generator.CodegenUtilTFlite import GenerateSourceFilesFromTFlite
from mcunet.mcunet.model_zoo import download_tflite

# 1. 从模型库下载预训练的 VWW（Visual Wake Words）TFLite 模型
# 2. 将模型部署到 MCU 之前，需要先把模型转换为中间表示（IR），并提取权重参数和量化参数
tflite_path = download_tflite(net_id="mcunet-vww1")

# 3. 为 MCU 端上部署生成 C 源代码
peakmem = GenerateSourceFilesFromTFlite(
    tflite_path,
    life_cycle_path="./lifecycle.png",
)
print(f"Peak memory: {peakmem} bytes")
