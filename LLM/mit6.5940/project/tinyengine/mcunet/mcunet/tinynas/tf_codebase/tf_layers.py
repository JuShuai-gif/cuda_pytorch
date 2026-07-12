# =============================================================================
# tf_layers.py — TensorFlow 基础层定义（用于 MCUNet 模型转换）
#
# 本文件是 PyTorch → TFLite 转换管线的核心构件之一。
# 它用原生 TensorFlow ops 实现了 MobileNetV2 / ProxylessNAS 风格的算子，
# 包括：Conv2D、DepthwiseConv2D、BatchNorm、激活函数、全连接层等。
#
# 设计目标：
#   - 与 PyTorch 的 padding 行为对齐（USE_TORCH_PADDING = True）
#   - 支持从 PyTorch 导出的权重字典注入（param_initializer 机制）
#   - 所有层都使用 NHWC 数据格式（TFLite 要求）
#
# 关键设计决策：
#   USE_TORCH_PADDING = True: 手动 pad 输入而非依赖 TF 的 'SAME' padding，
#   因为 PyTorch 的 Conv2d 默认使用 floor((k-1)/2) 的对称 padding，
#   而 TF 的 'SAME' 行为在偶数 kernel size 时与 PyTorch 不一致。
# =============================================================================

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import avg_pool2d

# ---------------------------------------------------------------------------
# USE_TORCH_PADDING:
#   全局标志，控制是否使用 "PyTorch 风格" 的 padding。
#   True  = 手动调用 tf.pad 做对称 padding，然后使用 'VALID' 卷积
#   False = 直接使用 TF 的 'SAME' padding 模式
#
# 为何需要此标志？
#   PyTorch 与 TensorFlow 对 "SAME" padding 的实现细节不同：
#   - PyTorch: 总是在两边均匀地 pad，pad = kernel_size // 2
#   - TF:      当 kernel_size 为偶数时，右侧/底部会多 pad 1 个像素
#   这种差异会导致导出的 TFLite 模型精度不一致。
#   因此设置为 True，通过手动 pad + 'VALID' 来保证与 PyTorch 完全对齐。
# ---------------------------------------------------------------------------
USE_TORCH_PADDING = True


def conv2d(
    _input,
    out_features,
    kernel_size,
    stride=1,
    padding="SAME",
    param_initializer=None,
    scope_name="conv",
    use_bias=False,
):
    """TensorFlow 2D 卷积层（支持权重注入和 PyTorch 风格 padding）

    参数说明:
        _input:             TF 张量，shape 为 [batch, height, width, channels] (NHWC)
        out_features:       输出通道数（卷积核数量）
        kernel_size:        卷积核边长（正方形卷积核）
        stride:             卷积步长
        padding:            填充模式，'SAME' 或 'VALID'
        param_initializer:  权重初始化器字典。key 是变量作用域路径，
                            例如 'conv/weight'。用于从 PyTorch state_dict 注入权重。
        scope_name:         TF variable_scope 名称，用于变量命名空间管理
        use_bias:           是否包含偏置项

    返回值:
        卷积后的 TF 张量，shape 为 [batch, new_h, new_w, out_features]

    工作流程:
        1. 从输入张量获取输入通道数 in_features
        2. 在 'scope_name' 作用域下创建 weight 变量
        3. 根据 padding 模式决定是否手动 pad（USE_TORCH_PADDING 时）
        4. 执行 tf.nn.conv2d
        5. 若 use_bias，创建 bias 变量并加到输出上
    """
    # 从输入张量的最后一个维度（NHWC 的 channels）获取输入通道数
    in_features = int(_input.get_shape()[3])

    # 如果未提供 param_initializer，初始化为空字典，避免后续 dict.get 出错
    if not param_initializer:
        param_initializer = {}
    output = _input
    with tf.variable_scope(scope_name):
        # 构建 weight 变量的初始化器查找 key
        # key 格式为 "conv/weight"（对应 PyTorch 的 "conv.weight" 经过 "/" 替换）
        init_key = "%s/weight" % tf.get_variable_scope().name
        # 如果在 param_initializer 中找到了对应的 key，使用该初始化器（即注入 PyTorch 权重）
        # 否则使用 He 初始化（variance_scaling_initializer）
        initializer = param_initializer.get(
            init_key, tf.contrib.layers.variance_scaling_initializer()
        )
        # 创建 weight 变量：shape = [kernel_size, kernel_size, in_features, out_features]
        weight = tf.get_variable(
            name="weight",
            shape=[kernel_size, kernel_size, in_features, out_features],
            initializer=initializer,
        )

        # 断言：padding 为 'SAME' 时必须配合 BN，否则必须使用 bias
        # 这是因为 'SAME' + 手动 pad 的路径没有 bias，需要靠 BN 提供可训练的偏移
        assert padding == "SAME" or use_bias

        if padding == "SAME":
            if USE_TORCH_PADDING:
                # ---- PyTorch 风格 padding ----
                # 在输入张量的上下左右各 pad kernel_size // 2 个像素
                pad = kernel_size // 2
                paddings = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
                output = tf.pad(output, paddings, "CONSTANT")

                # 使用 'VALID' 模式卷积（因为已经手动 pad 过了）
                output = tf.nn.conv2d(
                    output, weight, [1, stride, stride, 1], "VALID", data_format="NHWC"
                )
            else:
                # ---- TensorFlow 原生 'SAME' padding ----
                output = tf.nn.conv2d(
                    output, weight, [1, stride, stride, 1], "SAME", data_format="NHWC"
                )
        else:
            # ---- 'VALID' padding（不填充） ----
            output = tf.nn.conv2d(
                output, weight, [1, stride, stride, 1], "VALID", data_format="NHWC"
            )

        # 如果 use_bias，添加偏置项
        # 注意：MCUNet 中通常 use_bias=False，因为 BN 已经包含了可训练偏移
        if use_bias:
            init_key = "%s/bias" % tf.get_variable_scope().name
            initializer = param_initializer.get(
                init_key, tf.constant_initializer([0.0] * out_features)
            )
            bias = tf.get_variable(
                name="bias", shape=[out_features], initializer=initializer
            )
            output = output + bias
    return output


def depthwise_conv2d(
    _input, kernel_size, stride=1, padding="SAME", param_initializer=None
):
    """TensorFlow Depthwise 卷积（逐通道卷积）

    在 MobileNetV2 / MCUNet 中，depthwise conv 是 inverted bottleneck 的关键组件。
    它对每个输入通道独立执行卷积（groups = in_features），然后输出相同数量的通道。

    参数说明:
        _input:             TF 张量，NHWC 格式 [batch, h, w, in_features]
        kernel_size:        卷积核边长
        stride:             卷积步长
        padding:            填充模式
        param_initializer:  权重初始化器字典

    与普通卷积的关键区别:
        - weight shape: [kernel_size, kernel_size, in_features, 1]
        - 每个输入通道只有一个卷积核（而不是 out_features 个）
        - 输出通道数 = 输入通道数

    工作流程:
        1. 获取输入通道数
        2. 创建 depthwise 权重变量（输出通道维度为 1）
        3. 手动 pad（USE_TORCH_PADDING 时）后执行 tf.nn.depthwise_conv2d
    """
    # 获取输入通道数
    in_features = int(_input.get_shape()[3])

    if not param_initializer:
        param_initializer = {}
    output = _input
    with tf.variable_scope("conv"):
        # 解析权重初始化器，逻辑与 conv2d 相同
        init_key = "%s/weight" % tf.get_variable_scope().name
        initializer = param_initializer.get(
            init_key, tf.contrib.layers.variance_scaling_initializer()
        )
        # Depthwise conv 的权重 shape: [k, k, in_features, 1]
        # 第四个维度为 1 表示每个输入通道只有一个卷积核
        weight = tf.get_variable(
            name="weight",
            shape=[kernel_size, kernel_size, in_features, 1],
            initializer=initializer,
        )
        assert padding == "SAME"
        if USE_TORCH_PADDING:
            # PyTorch 风格 padding
            if padding == "SAME":
                pad = kernel_size // 2
                paddings = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
                output = tf.pad(output, paddings, "CONSTANT")

            # 使用 'VALID' 模式的 depthwise conv
            output = tf.nn.depthwise_conv2d(
                output, weight, [1, stride, stride, 1], "VALID", data_format="NHWC"
            )
        else:
            # 直接使用 TF 的 'SAME' 模式
            output = tf.nn.depthwise_conv2d(
                output, weight, [1, stride, stride, 1], "SAME", data_format="NHWC"
            )
    return output


def avg_pool(_input, k=2, s=2):
    """全局/局部平均池化

    在 MCUNet 中，avg_pool 通常用于全局平均池化（GAP），
    即在特征图的空间维度上取均值，将 [b, h, w, c] 变为 [b, 1, 1, c]。

    参数说明:
        _input:  TF 张量，NHWC 格式
        k:       池化窗口大小
        s:       池化步长

    注意:
        - 使用 'VALID' padding（窗口不超出边界）
        - 断言输入高度 == k == s，这通常是全局池化的情形
        - 底层调用 tf.contrib.layers.avg_pool2d
    """
    padding = "VALID"
    # 断言输入尺寸等于池化窗口大小——这是全局平均池化的特征
    # 如果条件不成立，说明不是全局池化，可能需要改用其他实现
    assert int(_input.get_shape()[1]) == k == s
    output = avg_pool2d(
        _input, kernel_size=[k, k], stride=[s, s], padding=padding, data_format="NHWC"
    )
    return output


def fc_layer(_input, out_units, use_bias=False, param_initializer=None):
    """全连接层（Fully Connected Layer）

    在 MCUNet 中，全连接层通常作为分类器使用，将特征向量映射到类别分数。

    参数说明:
        _input:             TF 张量，shape 为 [batch, features]
        out_units:          输出维度（分类头则为类别数）
        use_bias:           是否包含偏置项
        param_initializer:  权重初始化器字典

    工作流程:
        1. 获取输入特征维度 features_total
        2. 创建 weight 变量，shape = [features_total, out_units]
        3. 执行矩阵乘法 tf.matmul
        4. 可选地添加偏置项

    与 conv2d 转换的区别:
        - 当卷积核大小为 1x1 且输入/输出为 2D 时，fc 等价于 1x1 conv
        - TFLite 对全连接层有特定的算子优化
    """
    # 获取输入最后一个维度作为特征数
    features_total = int(_input.get_shape()[-1])
    if not param_initializer:
        param_initializer = {}
    with tf.variable_scope("linear"):
        # 使用 xavier_initializer（Glorot 初始化），与 fc 层的常见实践一致
        init_key = "%s/weight" % tf.get_variable_scope().name
        initializer = param_initializer.get(
            init_key, tf.contrib.layers.xavier_initializer()
        )
        weight = tf.get_variable(
            name="weight", shape=[features_total, out_units], initializer=initializer
        )
        # 矩阵乘法: output = input @ weight^T 不成立，这里是 input @ weight
        # 注意：tf.matmul(_input, weight) 中 _input 在左边，weight 在右边
        # 所以 weight 的 shape 必须是 [features_total, out_units]
        output = tf.matmul(_input, weight)
        if use_bias:
            init_key = "%s/bias" % tf.get_variable_scope().name
            initializer = param_initializer.get(
                init_key, tf.constant_initializer([0.0] * out_units)
            )
            bias = tf.get_variable(
                name="bias", shape=[out_units], initializer=initializer
            )
            output = output + bias
    return output


def batch_norm(_input, is_training, epsilon=1e-3, decay=0.9, param_initializer=None):
    """批归一化层（Batch Normalization）

    在推理阶段（is_training=False），BN 执行以下计算：
        output = gamma * (input - moving_mean) / sqrt(moving_var + epsilon) + beta

    参数说明:
        _input:             TF 张量
        is_training:        是否为训练模式。推理时传入 False
        epsilon:            防止除零的小常数
        decay:              moving_mean/moving_var 的指数衰减系数
        param_initializer:  参数初始化器字典

    权重映射（PyTorch → TF）:
        PyTorch BN 参数名    →    TF 参数名
        bn.weight           →    bn/gamma  （缩放因子）
        bn.bias             →    bn/beta   （偏移量）
        bn.running_mean     →    bn/moving_mean
        bn.running_var      →    bn/moving_variance

    重要说明:
        - scale=True: 使用可训练缩放因子 gamma
        - updates_collections=None: 立即更新 moving_mean/var，不放入 collections
        - 训练模式与推理模式下 BN 行为不同，is_training 控制此行为
    """
    with tf.variable_scope("bn"):
        scope = tf.get_variable_scope().name
        if param_initializer is not None:
            # 将 PyTorch 的 BN 参数名映射到 TF 的 BN 参数名
            bn_init = {
                "beta": param_initializer["%s/bias" % scope],
                "gamma": param_initializer["%s/weight" % scope],
                "moving_mean": param_initializer["%s/running_mean" % scope],
                "moving_variance": param_initializer["%s/running_var" % scope],
            }
        else:
            bn_init = None
        # 使用 tf.contrib.layers.batch_norm 实现 BN
        # data_format='NHWC' 表示 BN 在通道维度上操作
        output = tf.contrib.layers.batch_norm(
            _input,
            scale=True,  # 使用可训练的 gamma 缩放因子
            is_training=is_training,
            param_initializers=bn_init,
            updates_collections=None,  # 立即更新统计量
            epsilon=epsilon,
            decay=decay,
            data_format="NHWC",  # BN 作用于 NHWC 的最后一个维度
        )
    return output


def activation(x, activation="relu6"):
    """激活函数层

    MCUNet 主要使用 ReLU6 激活函数，而非标准的 ReLU。
    选择 ReLU6 的原因：
        - 输出上限为 6，适合低精度推理（INT8 量化时不易溢出）
        - 在量化到 uint8 时，ReLU6 比 ReLU 更容易校准
        - MobileNetV2 论文中广泛使用

    参数说明:
        x:          输入的 TF 张量
        activation: 激活函数类型，目前仅支持 'relu6'

    未来扩展预留:
        - HardSwish（EfficientNet 中使用）
        - Swish / SiLU
    """
    if activation == "relu6":
        # ReLU6: min(max(x, 0), 6)
        # 与 ReLU 的差别是上限为 6 而非无穷大
        # 以下被注释的代码展示了 HardSwish 的可能实现方式
        # return HardSwish()(_input)
        # return tf.nn.swish(_input)
        # return x * tf.nn.relu6(x + np.float32(3)) * np.float32(1. / 6.)
        return tf.nn.relu6(x)
    else:
        raise ValueError("Do not support %s" % activation)


def flatten(_input):
    """展平层：将多维特征图展平为 2D [batch, features]

    参数说明:
        _input: 任意维度的 TF 张量

    返回值:
        如果输入已经是 2D，直接返回；否则 reshape 为 [batch, -1]

    注意:
        - np.prod(input_shape[1:]) 计算特征维度总数
        - -1 表示自动推断 batch 维度
    """
    input_shape = _input.shape.as_list()
    if len(input_shape) != 2:
        # 展平除了 batch 维度外的所有维度
        return tf.reshape(_input, [-1, np.prod(input_shape[1:])])
    else:
        return _input


def dropout(_input, keep_prob, is_training):
    """Dropout 层

    注意：当前实现为恒等映射（即禁用 Dropout）。
    这是 MCUNet 的设计决定——在 MCU 部署时不需要 Dropout。

    未来如果需要启用:
        - 在训练模式下以 keep_prob 概率保留神经元
        - 在推理模式下直接返回输入
        - 使用 tf.cond 实现条件执行

    被注释的代码展示了标准 Dropout 实现。
    """
    # TODO: this modification may be problematic
    output = _input
    return output
    # 以下是被注释的标准 Dropout 实现：
    # if keep_prob < 1:
    #     output = tf.cond(
    #         tf.cast(is_training, tf.bool),
    #         lambda: tf.nn.dropout(_input, keep_prob),
    #         lambda: _input
    #     )
    # else:
    #     output = _input
    # return output


def _make_divisible(v, divisor, min_value=None):
    """将数值调整为给定除数的整数倍（用于通道数对齐）

    在 MobileNetV2 / MCUNet 中，为了硬件加速效率，通道数通常设为 8 或 16 的倍数。
    此函数确保调整后的值不会比原值小超过 10%。

    参数说明:
        v:          原始值（如输入通道数乘以扩展系数后的结果）
        divisor:    除数（对齐基数，通常为 8、16 或 32）
        min_value:  最小值下限，默认等于 divisor

    返回值:
        对齐后的值（divisor 的整数倍）

    工作流程:
        1. new_v = round(v / divisor) * divisor
        2. 如果 new_v < 0.9 * v（下调超过 10%），再加一个 divisor 补偿
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MBInvertedConvLayer:
    """Mobile Inverted Bottleneck 卷积层（MobileNetV2 核心组件）

    这是 MCUNet 中最重要的构建块。它由三个子层组成：
        1. inverted_bottleneck: 1x1 卷积扩展通道数（expand ratio > 1 时）
        2. depth_conv:          k×k depthwise 卷积（空间特征提取）
        3. point_linear:        1x1 卷积压缩回目标通道数

    架构设计原理:
        - "Inverted" 与传统 bottleneck 相反：先扩展再压缩
        - 这种设计将空间卷积和通道变换解耦，计算效率更高
        - Depthwise conv 的计算量远小于普通 conv，特别适合 MCU

    参数说明:
        _id:           层标识符（用于 TF variable_scope）
        filter_num:    输出通道数
        kernel_size:   depthwise 卷积核大小
        stride:        depthwise 步长
        expand_ratio:  扩展系数（中间通道数 = in_features * expand_ratio）
    """

    def __init__(self, _id, filter_num, kernel_size=3, stride=1, expand_ratio=6):
        """初始化 MBInvertedConvLayer

        参数:
            _id:           层的唯一标识符，如 'mobile_inverted_conv'
            filter_num:    最终输出通道数
            kernel_size:   depthwise 卷积核大小，默认 3
            stride:        depthwise 步长，默认 1
            expand_ratio:  通道扩展系数，默认 6（MobileNetV2 标准值）
        """
        self.id = _id
        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio

    def build(self, _input, net, init=None):
        """构建 MBInvertedConvLayer 的计算图

        参数说明:
            _input: 输入 TF 张量，NHWC 格式
            net:    父网络对象（提供 is_training、bn_eps、bn_decay 等属性）
            init:   权重初始化器字典

        工作流程:
            1. 如果 expand_ratio > 1:
               a. 1x1 conv: in_features → in_features * expand_ratio
               b. BN + ReLU6
            2. Depthwise conv（k×k，groups = 中间通道数）
               a. BN + ReLU6
            3. 1x1 conv: 中间通道数 → filter_num
               a. BN（无激活函数——这是 "linear" bottleneck 的关键）

        设计决策说明:
            - 第 3 步没有激活函数（linear bottleneck），这是 MobileNetV2 的关键创新
            - 保留信息在低维空间时，非线性激活会破坏信息流
            - 第 1 步的扩展可以只做 round 而非 _make_divisible，
              这是因为在 MCUNet 中 expand_ratio 已经过 NAS 搜索优化
        """
        output = _input
        in_features = int(_input.get_shape()[3])
        with tf.variable_scope(self.id):
            # ---- 第 1 步：Inverted Bottleneck（扩展通道） ----
            # 只有当 expand_ratio > 1 时才需要扩展
            # expand_ratio = 1 时意味着跳过扩展阶段
            if self.expand_ratio > 1:
                feature_dim = round(in_features * self.expand_ratio)
                # 以下注释展示了 _make_divisible 的可选做法
                # feature_dim = _make_divisible(in_features * self.expand_ratio, 32)
                with tf.variable_scope("inverted_bottleneck"):
                    # 1x1 卷积：扩展通道维度
                    output = conv2d(output, feature_dim, 1, 1, param_initializer=init)
                    output = batch_norm(
                        output,
                        net.is_training,
                        epsilon=net.bn_eps,
                        decay=net.bn_decay,
                        param_initializer=init,
                    )
                    output = activation(output, "relu6")

            # ---- 第 2 步：Depthwise 卷积（空间特征提取） ----
            with tf.variable_scope("depth_conv"):
                output = depthwise_conv2d(
                    output, self.kernel_size, self.stride, param_initializer=init
                )
                output = batch_norm(
                    output,
                    net.is_training,
                    epsilon=net.bn_eps,
                    decay=net.bn_decay,
                    param_initializer=init,
                )
                output = activation(output, "relu6")

            # ---- 第 3 步：Pointwise Linear 投影（压缩通道） ----
            # 注意：这里没有激活函数，形成 "linear bottleneck"
            with tf.variable_scope("point_linear"):
                output = conv2d(output, self.filter_num, 1, 1, param_initializer=init)
                output = batch_norm(
                    output,
                    net.is_training,
                    epsilon=net.bn_eps,
                    decay=net.bn_decay,
                    param_initializer=init,
                )
        return output


class ConvLayer:
    """标准卷积层（Conv2D + BN + ReLU6）

    简单的 Conv2D → BN → ReLU6 三层组合。
    在 MCUNet 中用于：
        - 输入层（first_conv）：将 RGB 图像映射到初始特征空间
        - 特征混合层（feature_mix_layer）：在 backbone 后的 1x1 融合层

    参数说明:
        _id:         层标识符
        filter_num:  输出通道数
        kernel_size: 卷积核大小
        stride:      步长
    """

    def __init__(self, _id, filter_num, kernel_size=3, stride=1):
        """初始化 ConvLayer"""
        self.id = _id
        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.stride = stride

    def build(self, _input, net, init=None):
        """构建 ConvLayer 计算图

        工作流程:
            1. Conv2D（不带 bias，因为后续有 BN）
            2. BN（提供可训练的偏移和缩放）
            3. ReLU6 激活
        """
        output = _input
        with tf.variable_scope(self.id):
            output = conv2d(
                output,
                self.filter_num,
                self.kernel_size,
                self.stride,
                param_initializer=init,
            )

            output = batch_norm(
                output,
                net.is_training,
                epsilon=net.bn_eps,
                decay=net.bn_decay,
                param_initializer=init,
            )

            output = activation(output, "relu6")
        return output


class ConvLayer_fc:
    """1x1 卷积形式的全连接层（用于分类头）

    在 MCUNet 中，分类器使用 1x1 卷积而非传统全连接层。
    原因：在全局平均池化后，特征图 shape 为 [b, 1, 1, c]，
    1x1 卷积等价于全连接，但可以复用已有的卷积优化路径。

    与 ConvLayer 的区别:
        - 使用 'VALID' padding（输入已经是 1x1，无需填充）
        - 使用 use_bias=True（分类器通常不需要 BN）
        - scope_name='linear'（与 fc_layer 命名一致）

    参数说明:
        _id:         层标识符
        filter_num:  输出维度（类别数）
        kernel_size: 卷积核大小（默认 3，但实际使用 1）
        stride:      步长
    """

    def __init__(self, _id, filter_num, kernel_size=3, stride=1):
        """初始化 ConvLayer_fc"""
        self.id = _id
        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.stride = stride

    def build(self, _input, net, init=None):
        """构建 1x1 卷积分类器

        工作流程:
            1. 使用 'VALID' padding 的 1x1 卷积
            2. 使用偏置（无 BN）
            3. scope_name 为 'linear'，与 fc_layer 对齐
        """
        output = _input
        with tf.variable_scope(self.id):
            output = conv2d(
                output,
                self.filter_num,
                self.kernel_size,
                self.stride,
                padding="VALID",
                # scope_name 设为 'linear' 以便与 PyTorch 的 'classifier.linear' 对齐
                param_initializer=init,
                scope_name="linear",
                use_bias=True,
            )
        return output


class LinearLayer:
    """标准全连接层（带可选的 Dropout）

    在 MCUNet 中，此实现被 ConvLayer_fc 取代（因为 ConvLayer_fc 更高效）。
    保留此处作为备用实现，但当前并未使用。

    参数说明:
        _id:       层标识符
        n_units:   输出维度
        drop_rate: Dropout 比率（当前被禁用）
    """

    def __init__(self, _id, n_units, drop_rate=0):
        """初始化 LinearLayer"""
        self.id = _id
        self.n_units = n_units
        self.drop_rate = drop_rate

    def build(self, _input, net, init=None):
        """构建全连接层计算图

        工作流程（当前实现为恒等映射）:
            1. Dropout（当前被 glob 禁用，直接返回输入）
            2. 全连接层（use_bias=True）
        """
        output = _input
        with tf.variable_scope(self.id):
            if self.drop_rate > 0:
                output = dropout(output, 1 - self.drop_rate, net.is_training)
            output = fc_layer(
                output, self.n_units, use_bias=True, param_initializer=init
            )
        return output
