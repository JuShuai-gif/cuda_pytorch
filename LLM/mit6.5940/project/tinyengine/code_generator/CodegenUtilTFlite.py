# ----------------------------------------------------------------------
# Project: TinyEngine
# Title:   CodegenUtilTFlite.py
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

import os
from tempfile import TemporaryDirectory

# 导入自定义模块：代码生成器、内存调度器、TFLite 转换器
from .CodeGenerator import CodeGenerator
from .GeneralMemoryScheduler import GeneralMemoryScheduler
from .TfliteConvertor import TfliteConvertor


def GenerateSourceFilesFromTFlite(
    tflite_path,
    life_cycle_path=None,
):
    """
    从 TFLite 模型文件生成 C 源码文件（模型推理代码）

    参数:
        tflite_path: TFLite 模型文件路径
        life_cycle_path: 内存生命周期图的保存路径，为 None 时使用临时目录
    """
    use_inplace = True  # 是否启用原地内存复用（in-place memory sharing）

    # 使用临时目录存放中间文件，退出时自动清理
    with TemporaryDirectory() as WORKING_DIR:
        if life_cycle_path is None:
            # 未指定路径时，生命周期图保存到临时目录
            schedule_image_path = os.path.join(WORKING_DIR, "schedule.png")
        else:
            schedule_image_path = life_cycle_path

        # Step 1: 解析 TFLite 模型，提取算子信息
        tf_convertor = TfliteConvertor(tflite_path)
        tf_convertor.parseOperatorInfo()
        layer = tf_convertor.layer
        outTable = []
        VisaulizeTrainable = False  # 代码生成阶段禁用可视化训练模式
        # Step 2: 创建内存调度器，规划激活值的运行时内存布局
        memory_scheduler = GeneralMemoryScheduler(
            layer,
            False,
            False,
            outputTables=outTable,
            inplace=use_inplace,
            mem_visual_path=schedule_image_path,
            VisaulizeTrainable=VisaulizeTrainable,
        )
        memory_scheduler.USE_INPLACE = use_inplace
        memory_scheduler.allocateMemory()  # 执行内存分配 / 调度算法

        # 获取转换器中的输出表（算子参数信息）
        outTable = tf_convertor.outputTables if hasattr(tf_convertor, "outputTables") else []
        # Step 3: 创建代码生成器，生成模型推理的 C 代码
        code_generator = CodeGenerator(
            memsche=memory_scheduler,
            inplace=memory_scheduler.USE_INPLACE,
            unsigned_input=False,   # 输入数据不使用无符号类型
            patch_params=None,      # patch-based 推理参数（MCUNetV2）
            FP_output=False,        # 输出不需要浮点
            profile_mode=False,     # 非性能分析模式
            fp_requantize=True,     # 启用浮点反量化
            tflite_op=False,        # 不使用 TFLite 算子模拟
            dummy_address=False,    # 不使用虚假地址
            outputTables=outTable,
        )
        # 代码生成前设置检测输出（如有）
        code_generator.codeGeneration()

        # 返回输入/输出缓冲区的指针信息
        return memory_scheduler.buffers["input_output"]
