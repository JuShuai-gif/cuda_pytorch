# ============================================================================
# eval_det.py —— 行人检测（Person Detection）推理与可视化
#
# 用途：
#   对 MCUNet 的 TFLite 检测模型（person-det）进行推理，并将检测结果
# （边界框）可视化输出到原图上。推理使用 TensorFlow Lite 的 C++ 后端
# （通过 Python API 调用），后处理（NMS + YOLO 解码）在 PyTorch 中完成。
#
# 工作流程：
#   下载 TFLite 模型 → 加载到 TF Lite Interpreter → 读取输入图像
#   → 预处理（归一化到 [-1, 1]）→ TF Lite 推理 → YOLO 输出层解码
#   → MergeNMS 去除重叠框 → 绘制边界框并保存可视化结果
#
# 说明：
#   这里的后处理逻辑（YOLO 解码 + NMS）与 MCU 端 TinyEngine 运行时
#   中的 C 实现逻辑一致，只是这里先用 Python 验证效果。
# ============================================================================

import os
import argparse
import numpy as np

import torch
import tensorflow as tf
from PIL import Image, ImageDraw
from mcunet.utils.det_helper import MergeNMS, Yolo3Output

from mcunet.model_zoo import download_tflite

# 强制 TensorFlow 只在 CPU 上运行（TFLite 的 INT8 量化推理在 CPU 上更快，
# 而且 MCU 目标平台也是 ARM CPU，用 CPU 评测更贴近实际部署场景）
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # use only cpu for tf-lite evaluation

# 关闭 TensorFlow 的日志输出（只显示错误级别以上）
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# 命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument(
    "--net_id", type=str, help="模型在 model_zoo 中的标识符，固定为 person-det"
)
parser.add_argument(
    "--image_path",
    default="assets/sample_images/person_det.jpg",
    help="输入图像的路径（默认使用项目中自带的示例图片）",
)

args = parser.parse_args()


# ============================================================================
# eval_image —— 对单张图像执行 TFLite 推理 + YOLO 后处理 + 可视化
#
# 参数:
#   image (np.ndarray): 形状为 (1, H, W, C) 的预处理后图像数据
#
# 流程:
#   1. 将图像传入 TFLite Interpreter 执行推理
#   2. 从三个输出特征图（对应三个 YOLO 尺度）中获取原始预测
#   3. 通过 YOLO3Output 层将特征图解码为 (class_id, score, bbox)
#   4. 将三个尺度的输出拼接，送入 MergeNMS 去除重叠框
#   5. 在原始图像上绘制置信度 > 0.3 的边界框
#   6. 保存可视化结果到同目录下的 *_vis.jpg 文件
# ============================================================================
def eval_image(image):
    # 将输入数据设置到 TFLite 的输入张量中
    interpreter.set_tensor(input_details[0]["index"], image.reshape(*input_shape))
    # 执行一次推理（invoke 是 TF Lite 的推理入口）
    interpreter.invoke()
    # 从输出张量中读取结果（三个特征图分别对应三个 YOLO 尺度）
    output_data = [
        interpreter.get_tensor(output_details[i]["index"])
        for i in range(len(output_details))
    ]

    # 将 TF Lite 输出从 NumPy 转成 PyTorch Tensor 并调整维度排列
    # TFLite 输出形状: (batch, height, width, channels)
    # PyTorch 期望形状: (batch, channels, height, width)
    # 所以需要 permute(0, 3, 1, 2)
    outputs = [
        torch.from_numpy(d).permute(0, 3, 1, 2).contiguous() for d in output_data
    ]

    # 对每个尺度的特征图执行 YOLO 解码：
    #   - 将网格偏移 + sigmoid(中心偏移) → 中心坐标
    #   - exp(尺度预测) * anchor → 宽高
    #   - sigmoid(物体性) * sigmoid(类别) → 类别置信度
    #   - 输出形状: (batch, num_anchors * H * W, 6) 每行为 [class_id, score, x1, y1, x2, y2]
    outputs = [
        output_layer(output) for output_layer, output in zip(output_layers, outputs)
    ]

    # 将三个尺度的检测结果在维度 1（anchor 数量）上拼接
    # 这样所有尺度的候选框在一起进行 NMS
    outputs = torch.cat(outputs, dim=1)

    # 执行 MergeNMS 非极大值抑制，去除重叠框
    # 返回: ids (类别ID), scores (置信度), bboxes (边界框 [x1,y1,x2,y2])
    ids, scores, bboxes = nms_layer(outputs)

    # ========== 可视化部分 ==========
    # 只保留置信度阈值大于 0.3 的检测结果
    threshold = 0.3
    n_positive = (scores > threshold).sum()
    ids = ids[0, :n_positive, 0].numpy()  # 取第一张图的类别 ID
    bboxes = bboxes[0, :n_positive].numpy()  # 取第一张图的边界框坐标

    # 重新加载原始图像（用于绘制框线）
    pil_image = load_example_image(resolution[::-1])
    image_draw = ImageDraw.Draw(pil_image)

    # 遍历每个检测到的目标，绘制红色边界框并打印坐标
    for cls, bbox in zip(ids, bboxes):
        image_draw.rectangle(bbox, outline="red")
        print(cls, [round(_) for _ in bbox])

    # 保存可视化结果到 *_vis.jpg
    filename, file_extension = os.path.splitext(args.image_path)
    vis_image_dir = filename + "_vis" + file_extension
    pil_image.save(vis_image_dir)


# ============================================================================
# load_example_image —— 从磁盘加载图像并缩放到指定分辨率
#
# 参数:
#   resolution (tuple): (宽度, 高度) 目标分辨率
# 返回:
#   PIL.Image 对象
# ============================================================================
def load_example_image(resolution):
    # 打开图像并转为 RGB 三通道
    image = Image.open(args.image_path).convert("RGB")
    # 缩放到模型输入要求的分辨率
    image = image.resize(resolution)
    return image


# ============================================================================
# preprocess_image —— 图像预处理
#
# 将 PIL Image 转换为模型输入所需的格式：
#   1. 转为 NumPy 数组，添加 batch 维度
#   2. 归一化到 [-1, 1]：pixel = (pixel / 255) * 2 - 1
#
# 注意这里使用 float32 类型，因为 TFLite 模型的输入层有量化器节点，
# 它会自动将 float32 输入量化为 INT8，所以我们输入 float 即可。
# ============================================================================
def preprocess_image(image):
    image_np = np.array(image)[None, ...]  # 添加 batch 维度 (1, H, W, C)
    image_np = (image_np / 255) * 2 - 1  # 归一化到 [-1, 1]
    return image_np.astype("float32")


# ============================================================================
# build_det_helper —— 构建检测后处理所需的组件
#
# 构建两个关键组件：
#   1. MergeNMS：合并模式的 NMS（对重叠框做加权平均融合，而非直接丢弃）
#   2. YOLO3Output：三个尺度的 YOLO 输出解码层
#
# 三个尺度的 anchor 配置对应 YOLOv3 的设计：
#   - stride=32 (大尺度)：检测大目标，anchor 大
#   - stride=16 (中尺度)：检测中目标，anchor 中等
#   - stride=8  (小尺度)：检测小目标，anchor 小
#
# 返回:
#   (nms_layer, output_layers) 元组
# ============================================================================
def build_det_helper():
    # 构建 MergeNMS 配置（合并模式会在 IoU 超过阈值时对框做加权平均融合，
    # 而不是直接丢弃低分框，这对行人检测这种目标部分重叠的场景更友好）
    nms = MergeNMS.build_from_config(
        {
            "nms_name": "merge",
            "nms_valid_thres": 0.01,  # 置信度过滤阈值（低于该值的框直接被忽略）
            "nms_thres": 0.45,  # NMS 的 IoU 阈值（高于该值的重叠框被合并）
            "nms_topk": 400,  # NMS 前保留的最高分框数量限制
            "post_nms": 100,  # NMS 后保留的最大框数
            "pad_val": -1,  # 填充值（用于将输出填充到固定维度）
        }
    )
    # YOLOv3 的三个尺度输出层配置
    # 每个尺度包含: num_class(类别数), anchors(anchor 宽高对),
    # stride(特征图相对于输入的下采样倍数), alloc_size(预设网格大小)
    output_configs = [
        {
            "num_class": 1,
            "anchors": [116, 90, 156, 198, 373, 326],
            "stride": 32,
            "alloc_size": [128, 128],
        },
        {
            "num_class": 1,
            "anchors": [30, 61, 62, 45, 59, 119],
            "stride": 16,
            "alloc_size": None,
        },
        {
            "num_class": 1,
            "anchors": [10, 13, 16, 30, 33, 23],
            "stride": 8,
            "alloc_size": None,
        },
    ]
    # 构建三个 YOLO 输出层，并设置为 eval 模式（禁用训练时的额外输出）
    outputs = [Yolo3Output(**cfg).eval() for cfg in output_configs]
    return nms, outputs


# ============================================================================
# 主入口
# ============================================================================
if __name__ == "__main__":
    # 第一步：下载 person-det 检测模型的 TFLite 文件
    tflite_path = download_tflite(net_id="person-det")

    # 第二步：创建 TFLite Interpreter 并分配张量内存
    # TF Lite Interpreter 负责加载模型、管理张量内存和执行推理
    interpreter = tf.lite.Interpreter(tflite_path)
    interpreter.allocate_tensors()

    # 第三步：构建检测后处理组件（YOLO 解码 + NMS）
    nms_layer, output_layers = build_det_helper()

    # 第四步：获取模型的输入/输出张量信息
    # input_details 包含: index(张量索引), shape(形状), dtype(数据类型), name(名称)
    # output_details 同理
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 模型的输入形状通常是 (1, height, width, 3)，提取分辨率
    input_shape = input_details[0]["shape"]
    resolution = input_shape[1:3]  # 注意该模型使用非正方形输入

    # 第五步：加载示例图像并预处理
    sample_image = load_example_image(resolution[::-1])  # PIL 要求 (width, height)
    sample_image = preprocess_image(sample_image)

    # 第六步：执行推理 + 后处理 + 可视化
    eval_image(sample_image)
