# =============================================================================
# generate_tflite.py — PyTorch → TFLite 转换管线
#
# 本文件是 MCUNet 模型部署管线的核心：
#   1. 读取 PyTorch 训练的模型权重
#   2. 转换为 TensorFlow 权重格式（维度 permute + key 替换）
#   3. 使用 tf_layers.py/tf_modules.py 构建等价的 TF 计算图
#   4. 通过 TFLiteConverter 导出为 INT8 量化的 TFLite 模型
#
# 关键设计：
#   - PyTorch 权重到 TF 的维度 permute 映射（因为 channel 顺序不同）
#   - 使用 representative dataset 做 int8 量化校准
#   - 将全精度模型 INT8 量化，以适应 MCU 的有限算力
# =============================================================================

import tensorflow as tf
import torch
import functools
import numpy as np


def generate_tflite_with_weight(
    pt_model, resolution, tflite_fname, calib_loader, n_calibrate_sample=500
):
    """将 PyTorch MCUNet 模型转换为 INT8 量化的 TFLite 模型

    这是 PyTorch → TFLite 完整转换管线的入口函数。

    参数说明:
        pt_model:           PyTorch 模型实例（必须是 ProxylessNASNets 类型）
        resolution:         输入图像分辨率（如 224、192、160 等）
        tflite_fname:       输出的 TFLite 文件路径
        calib_loader:       校准数据加载器（用于 int8 量化范围确定）
        n_calibrate_sample: 校准使用的样本数量（默认 500）

    工作流程:
        第 1 步 — 权重格式转换:
            PyTorch 的 conv weight shape: [out_c, in_c, k, k] (NCHW)
            TF 的 conv weight shape:      [k, k, in_c, out_c] (HWIO)
            需要 permute / transpose 调整维度顺序

        第 2 步 — TF 计算图构建:
            使用 tf_modules.ProxylessNASNets 构建等价的 TF 网络
            传入已转换的权重字典，注入到 TF 变量中

        第 3 步 — TFLite 导出 + INT8 量化:
            - 使用 representative_dataset 做校准
            - 输入和输出都是 int8 类型
            - 启用默认优化 (Optimize.DEFAULT)
            - 设置支持的算子为 TFLITE_BUILTINS_INT8

    关于量化的重要说明:
        - INT8 量化将浮点权重和激活映射到 [-128, 127] 范围内
        - 量化过程需要校准数据来确定每层激活的数值范围
        - 校准本质上是统计每层输出的 min/max 或分布
        - TensorRT / TFLite 使用不同的校准策略（本实现使用每批次推理）
    """

    # ========== 第 1 步：将 PyTorch state_dict 转换为 TF 权重格式 ==========

    # 获取 PyTorch 模型的 state_dict（有序字典，包含所有可学习参数）
    pt_sd = pt_model.state_dict()

    # 初始化 TF 权重字典
    # key 将使用 '/' 替换 '.' 来适配 TF 的 variable_scope 命名
    tf_sd = {}
    for key, v in pt_sd.items():
        # --- PyTorch → TF 的维度 permute 映射规则 ---
        # PyTorch Conv2d weight: [out_c, in_c, k_h, k_w] (NCHW)
        # TF    Conv2d weight:   [k_h, k_w, in_c, out_c] (HWIO)
        # 所以在 torch 中需要 permute(2, 3, 1, 0) 或 permute(2, 3, 0, 1)

        if key.endswith("depth_conv.conv.weight"):
            # Depthwise conv:
            #   PyTorch shape: [out_c=in_c, in_c=1, k, k]（multiplicative 因子）
            #   TF shape:      [k, k, in_c, 1]
            # permute(2, 3, 0, 1): 把 k,k 放到前两维，out_c 放到第 3 维，in_c=1 放到第 4 维
            # 但实际上 Depthwise conv 在 PyTorch 中的 shape 是 [in_c, 1, k, k]
            # 需要 permute(2, 3, 0, 1) → [k, k, in_c, 1]
            v = v.permute(2, 3, 0, 1)
        elif key.endswith("conv.weight"):
            # 普通 Conv2d:
            #   PyTorch shape: [out_c, in_c, k, k]
            #   TF shape:      [k, k, in_c, out_c]
            # permute(2, 3, 1, 0): k,k → 0,1; in_c → 2; out_c → 3
            v = v.permute(2, 3, 1, 0)
        elif key == "classifier.linear.weight":
            # 全连接层权重（分类器）:
            #   PyTorch shape: [out_units, in_features]
            #   TF shape:      [in_features, out_units]（matmul 约定不同）
            # permute(1, 0): 转置
            v = v.permute(1, 0)

        # 将 PyTorch 的 '.' 分隔符替换为 TF 的 '/' 分隔符
        # 例如: 'features.0.conv.weight' → 'features/0/conv/weight'
        # 这样就能匹配 TF variable_scope 中使用的 init_key 格式
        tf_sd[key.replace(".", "/")] = v.numpy()

    # ========== 第 2 步：构建 TF 网络并注入权重 ==========

    weight_decay = 0.0

    # 创建一个新的 TF 计算图（避免与默认图冲突）
    with tf.Graph().as_default() as graph:
        # 在每个 session 内构建图
        with tf.Session() as sess:
            # 定义网络映射函数，将输入 TF 占位符映射到输出 logits
            def network_map(images):
                """将输入 images 通过网络前向传播得到 logits

                参数:
                    images: TF 占位符，shape [1, h, w, 3]

                返回值:
                    (logits, auxiliary_endpoints_dict)
                """
                # 从 PyTorch 模型配置中获取网络架构配置
                net_config = pt_model.config
                # 动态导入 tf_modules（避免循环依赖）
                from .tf_modules import ProxylessNASNets

                # 创建 TF 版本的 ProxylessNASNets
                # 传入 net_config（架构配置）和 tf_sd（权重字典）
                # is_training=False 表示推理模式（BN 使用 moving statistics）
                net_tf = ProxylessNASNets(
                    net_config=net_config,
                    net_weights=tf_sd,
                    n_classes=pt_model.classifier.linear.out_features,
                    graph=graph,
                    sess=sess,
                    is_training=False,
                    images=images,
                    img_size=resolution,
                )
                logits = net_tf.logits
                return logits, {}  # 第二个返回值保留给 auxiliary endpoints

            # 定义 arg_scope（此处没有使用特殊的 arg_scope）
            def arg_scopes_map(weight_decay=0.0):
                arg_scope = tf.contrib.framework.arg_scope
                with arg_scope([]) as sc:
                    return sc

            slim = tf.contrib.slim

            # 使用 functools.wraps 保留 network_map 的签名信息
            @functools.wraps(network_map)
            def network_fn(images):
                arg_scope = arg_scopes_map(weight_decay=weight_decay)
                with slim.arg_scope(arg_scope):
                    return network_map(images)

            # 创建输入占位符（NHWC 格式）
            # shape = [1, resolution, resolution, 3]，batch 固定为 1
            # 注意：TFLite 通常使用固定 batch size（1）
            input_shape = [1, resolution, resolution, 3]
            placeholder = tf.placeholder(
                name="input", dtype=tf.float32, shape=input_shape
            )

            # 构建完整计算图并执行一次前向传播（此时会初始化变量）
            out, _ = network_fn(placeholder)

            # ========== 第 3 步：转换为 TFLite（INT8 量化） ==========

            # 从 session 创建 TFLiteConverter
            # 需要指定输入和输出张量
            converter = tf.lite.TFLiteConverter.from_session(sess, [placeholder], [out])

            # 启用默认优化（将触发 int8 量化）
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            # 设置推理输入输出类型为 int8
            # 这意味着模型将使用纯 int8 运算（而非 float16 或混合精度）
            converter.inference_output_type = tf.int8
            converter.inference_input_type = tf.int8

            # 定义代表性数据集生成器（用于量化校准）
            def representative_dataset_gen():
                """生成代表性样本用于 INT8 量化校准

                校准目标：确定每层激活值的量化范围（scale 和 zero_point）。
                方法：遍历校准数据集的子集，让 TFLite 观察每层的数值分布。

                Yields:
                    每个元素是 [input_tensor] 的列表，input_tensor 已经是 NHWC 格式
                """
                for i_b, (data, _) in enumerate(calib_loader):
                    # 达到校准样本数量上限后停止
                    if i_b == n_calibrate_sample:
                        break
                    # PyTorch 的数据格式是 NCHW，需要转换为 NHWC
                    # transpose(0, 2, 3, 1): [N, C, H, W] → [N, H, W, C]
                    # 同时转换为 float32 以确保与 TF 图兼容
                    data = data.numpy().transpose(0, 2, 3, 1).astype(np.float32)
                    yield [data]

            # 设置代表性数据集
            converter.representative_dataset = representative_dataset_gen

            # 限制仅使用 INT8 算子（硬件兼容性约束）
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

            # 执行转换：生成 TFLite flatbuffer
            tflite_buffer = converter.convert()

            # 写入文件
            tf.gfile.GFile(tflite_fname, "wb").write(tflite_buffer)


if __name__ == "__main__":
    # ========== 命令行入口 ==========
    # 使用方法: python generate_tflite.py <cfg_path> <ckpt_path> <tflite_path>
    #
    # 参数说明:
    #   cfg_path:    模型架构配置文件（JSON 格式）
    #   ckpt_path:   PyTorch 模型检查点路径（如果为 None 则随机初始化）
    #   tflite_path: 输出的 TFLite 文件路径

    # a simple script to convert the model to
    import sys

    sys.path.append("")
    import json

    # 解析命令行参数
    cfg_path = sys.argv[1]  # 架构配置文件路径
    ckpt_path = sys.argv[2]  # 模型权重检查点路径
    tflite_path = sys.argv[3]  # 输出 TFLite 文件路径

    # 导入并构建 PyTorch 模型
    from mcunet.tinynas.nn import ProxylessNASNets

    # 从 JSON 配置文件加载网络架构
    cfg = json.load(open(cfg_path))
    model = ProxylessNASNets.build_from_config(cfg)

    # 加载预训练权重（如果提供了检查点）
    if ckpt_path != "None":
        sd = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(sd["state_dict"])

    # ========== 准备量化校准数据 ==========
    # 使用 ImageNet 训练集的子集作为校准数据
    # calibrate the model for quantization
    from torchvision import datasets, transforms

    train_dataset = datasets.ImageFolder(
        "/dataset/imagenet/train",
        transform=transforms.Compose(
            [
                # transforms.Resize(int(resolution * 256 / 224)),
                # transforms.CenterCrop(resolution),
                transforms.RandomResizedCrop(cfg["resolution"]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        ),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=4
    )

    # 执行转换
    generate_tflite_with_weight(
        model, cfg["resolution"], tflite_path, train_loader, n_calibrate_sample=500
    )
