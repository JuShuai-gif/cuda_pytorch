# ----------------------------------------------------------------------
# Project: TinyEngine
# Title:   TfliteConvertor.py
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

import logging

# TFLite 解析器模块：将各种算子（Conv、Add、Pool 等）从 FlatBuffer 格式解析为 TinyEngine 内部 IR
import code_generator.converters.tflite_parser as TF_Parser
from code_generator.converters.tflite_parser.mean1dto2d import MEAN2D
from code_generator.converters.tflite_parser.utils import get_input_tensors, get_output_tensors, getOpCodeStr

from .constant import SKIP_OPs  # 不需要转换、可直接跳过的算子列表
# TFLite FlatBuffer 的 Python 绑定（由 flatc 编译器自动生成）
from .tflite import Model


# 将 TFLite 模型解析并转换为 TinyEngine 内部 IR 格式
class TfliteConvertor(object):
    def __init__(self, filepath):
        self.filepath = filepath  # TFLite 模型文件路径
        # 使用 FlatBuffers 零拷贝读取模型：直接解析二进制 buffer，无需反序列化开销
        self.model = self.loadTFmodel(filepath)
        self.subgraph = self.model.Subgraphs(0)  # TFLite 可以有多个子图，这里取第一个（主图）
        self.layer = []  # 转换后的算子层列表，即 TinyEngine IR
        self.tmpPADIndice = None  # 临时保存 PAD 算子的输入/输出索引，用于后续的 conv+pad 融合
        self.skip_transpose = None  # 临时保存 TRANSPOSE 算子的索引，用于图优化（跳过转置）
        self.average_1D_to_2D_holder = MEAN2D()  # 用于将 1D MEAN 算子合并为 2D MEAN 的辅助对象

    # ========== 公共方法 ==========

    def loadTFmodel(self, filepath):
        """从文件读取 TFLite FlatBuffer 二进制数据，零拷贝解析为 Model 对象"""
        buf = open(filepath, "rb").read()
        return Model.Model.GetRootAsModel(buf, 0)

    def dumpModelInfo(self):
        """打印模型基本信息（调试用）"""
        version = self.model.Version()
        print("Model version:", version)
        description = self.model.Description().decode("utf-8")
        print("Description:", description)
        subgraph_len = self.model.SubgraphsLength()
        print("Subgraph length:", subgraph_len)

        self.dumpLayerInfo()

    def dumpLayerInfo(self):
        """打印每一层的算子类型和输入输出索引（调试用）"""
        print("Layer length:", len(self.layer))

        for i, layer in enumerate(self.layer):
            if self.layer[i]["op"] == "ADD":
                print(
                    "op:",
                    layer["op"],
                    ",input_idx:",
                    layer["input_idx"],
                    ",input2_idx:",
                    layer["input2_idx"],
                    "output_idx:",
                    layer["output_idx"],
                )
            else:
                print(
                    "op:",
                    layer["op"],
                    ",input_idx:",
                    layer["input_idx"],
                    "output_idx:",
                    layer["output_idx"],
                )

    def parseOperatorInfo(self):
        """
        遍历 TFLite 子图中的所有算子，逐一解析并转换为 TinyEngine IR。
        同时检测并融合 SE（Squeeze-and-Excitation）模块的算子模式。
        """
        operators_len = self.subgraph.OperatorsLength()

        skip_next_ops = 0  # 用于跳过已融合到 SE 模块内的后续算子
        for i in range(operators_len):
            if skip_next_ops > 0:
                skip_next_ops -= 1
                continue

            op = self.subgraph.Operators(i)
            # 预取后续两个算子，检测是否为 SE 模块的 ADD->MUL->MUL 三算子模式
            if i + 2 < operators_len - 2:
                next_op = self.subgraph.Operators(i + 1)
                next_next_op = self.subgraph.Operators(i + 2)
                three_op_sequence = [op, next_op, next_next_op]

                if self.checkIfRequireSEelementmult(three_op_sequence):
                    logging.info("found SE block")
                    skip_next_ops = 2  # 跳过后续 2 个已被融合的算子
                    # 下图展示 SE 模块的计算图结构：
                    #         -> MEAN -> MEAN -> PWCONV -> PWCONV -> | ADD -> MUL ->     |
                    #  DWCONV                                        |            -> MUL |
                    #                                                |   融合目标  SEelementmult |
                    SEelementmult_op = TF_Parser.parse_SEelement(three_op_sequence, self.model, self.layer)

                    self.layer.append(SEelementmult_op)
                    continue

            self._handleOperator(op)

    # ========== 单个算子解析（按算子类型分发到对应的 parser） ==========
    def _handleOperator(self, op):
        """根据算子类型，调用对应的 parser 函数，将 TFLite 算子转换为 TinyEngine IR"""
        op_code_str = getOpCodeStr(op, self.model)
        if op_code_str == "CONV_2D":
            # 常规卷积：传入 tmpPADIndice 以支持 pad+conv 融合
            self.layer.append(TF_Parser.parse_conv2d(op, self.model, self.tmpPADIndice))
            self.tmpPADIndice = None  # 融合完成后清空
        elif op_code_str == "ADD":
            self.layer.append(TF_Parser.parse_add(op, self.model))
        elif op_code_str == "AVERAGE_POOL_2D":
            self.layer.append(TF_Parser.parse_avgpool(op, self.model))
        elif op_code_str == "DEPTHWISE_CONV_2D":
            # 深度可分离卷积：与常规卷积共用同一个 parser
            self.layer.append(TF_Parser.parse_conv2d(op, self.model, self.tmpPADIndice))
            self.tmpPADIndice = None
        elif op_code_str == "PAD":
            # PAD 不单独生成层，而是保存索引供后续 conv 融合
            self._convert_PAD(op)
        elif op_code_str == "RESIZE_NEAREST_NEIGHBOR":
            self.layer.append(TF_Parser.parse_upsample(op, self.model))
        elif op_code_str == "MAX_POOL_2D":
            self.layer.append(TF_Parser.parse_maxpool(op, self.model))
        elif op_code_str in "MEAN":
            # MEAN 算子：尝试将连续的 1D MEAN 合并为单个 2D MEAN
            ret_op = TF_Parser.parse_mead1dto2d(op, self.model, self.average_1D_to_2D_holder)
            if ret_op is not None:
                # TODO: 目前只处理特定图模式：TRANSPOSE -> MEAN -> MEANS
                if self.skip_transpose is not None:
                    ret_op.params["input_idx"] = self.skip_transpose.input_idx
                    ret_op.input_tensors[0].graph_idx = self.skip_transpose.input_idx
                self.layer.append(ret_op)
        elif op_code_str == "TRANSPOSE":
            # TRANSPOSE 不单独生成层，保存索引用于后续优化
            self._convert_TRANSPOSE(op)
        elif op_code_str == "FULLY_CONNECTED":
            self.layer.append(TF_Parser.parse_fc(op, self.model))
        elif op_code_str in SKIP_OPs:
            # 跳过不需要转换的算子（如 RESHAPE、IDENTITY 等）
            pass
        else:
            raise NotImplementedError(f"Unsupported {op_code_str}")

    def checkIfRequireSEelementmult(self, three_op_sequence):
        """
        检测三个连续算子是否为 SE（Squeeze-and-Excitation）模块模式：
        ADD -> MUL -> MUL
        下图展示完整的 SE 模块路径（从中途插入）：
                 -> MEAN -> MEAN -> PWCONV -> PWCONV -> | ADD -> MUL ->     |
          DWCONV                                        |            -> MUL |
                                                        |   融合目标         |
        """
        if (
            getOpCodeStr(three_op_sequence[0], self.model) == "ADD"
            and getOpCodeStr(three_op_sequence[1], self.model) == "MUL"
            and getOpCodeStr(three_op_sequence[2], self.model) == "MUL"
        ):
            return True
        return False

    # ========== 算子融合预处理 ==========

    def _convert_PAD(self, op):
        """
        处理 PAD 算子：不单独生成层，而是将 PAD 的输入/输出张量索引保存到 tmpPADIndice，
        以便后续的 Conv 算子将其融合进来（pad 被合并到卷积的 im2col 阶段）。
        """
        input_tensors = get_input_tensors(op, self.model)
        input_tensor = input_tensors[0]

        output_tensors = get_output_tensors(op, self.model)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        self.tmpPADIndice = PAD_tensorIndice(input_tensor.tensor_idx, output_tensor.tensor_idx)

    def _convert_TRANSPOSE(self, op):
        """
        处理 TRANSPOSE 算子：不单独生成层，将索引保存到 skip_transpose，
        用于后续的图优化——将 TRANSPOSE 的输出索引直接替换为输入索引，
        达到消除冗余转置的效果。
        """
        input_tensors = get_input_tensors(op, self.model)
        input_tensor = input_tensors[0]

        output_tensors = get_output_tensors(op, self.model)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        self.skip_transpose = PAD_tensorIndice(input_tensor.tensor_idx, output_tensor.tensor_idx)


# 辅助类：保存算子融合时被跳过的中间张量的输入/输出索引
class PAD_tensorIndice(object):
    def __init__(self, input_idx, output_idx):
        self.input_idx = input_idx
        self.output_idx = output_idx
