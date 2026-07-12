# =============================================================================
# tf_modules.py — MCUNet TensorFlow 高级模块实现
#
# 本文件在 tf_layers.py 的基础层之上构建了两个核心组件：
#   1. MobileInvertedResidualBlock — 带残差连接的反向瓶颈块
#   2. ProxylessNASNets — 完整的 MCUNet / ProxylessNAS 网络
#
# 这两个组件共同构成了 TF 计算图的核心骨架，
# 是 PyTorch → TFLite 转换管线的关键环节。
# =============================================================================

from .tf_layers import *


class MobileInvertedResidualBlock:
    """Mobile Inverted Residual Block（移动端反向残差块）

    这是 MobileNetV2 提出的基础构建块，在 MCUNet 中被广泛使用。
    它的结构是：MBInvertedConvLayer + 可选的残差连接。

    残差连接的条件（has_residual）：
        - 当 stride = 1 且输入输出通道数相同时启用
        - stride ≠ 1 时禁用（空间尺寸变化，无法直接相加）
        - 通道数变化时可能通过 shortcut 层（1x1 conv）调整

    参数说明:
        _id:                   块标识符
        mobile_inverted_conv:  MBInvertedConvLayer 实例
        has_residual:          是否使用残差连接（shortcut）
    """

    def __init__(self, _id, mobile_inverted_conv, has_residual):
        """初始化 MobileInvertedResidualBlock

        参数:
            _id:                   块标识符，如 'blocks/0'
            mobile_inverted_conv:  MBInvertedConvLayer 实例（处理主路径）
            has_residual:          是否添加残差连接
        """
        self.id = _id
        self.mobile_inverted_conv = mobile_inverted_conv
        self.has_residual = has_residual

    def build(self, _input, net, init=None):
        """构建残差块的计算图

        工作流程:
            1. 输入通过 MBInvertedConvLayer 主路径
            2. 如果启用了残差连接，将输出与输入相加（element-wise add）
            3. 返回最终输出

        TF 中的残差连接实现：
            - output = F(x) + x（当 has_residual = True 时）
            - 这是一个 element-wise addition，TFLite 会将其融合为 ADD 算子
            - 这种融合对 MCU 推理效率至关重要
        """
        output = _input
        with tf.variable_scope(self.id):
            # 执行主路径：MBInvertedConvLayer
            output = self.mobile_inverted_conv.build(output, net, init)

            # 如果启用残差连接，添加 element-wise add
            # 注意：这要求 _input 和 output 的 shape 完全一致
            if self.has_residual:
                output = output + _input
        return output


class ProxylessNASNets:
    """完整的 ProxylessNAS / MCUNet 网络模型（TensorFlow 实现）

    此类是整个 TF 转换管线的核心。它：
        1. 接收网络配置字典（net_config），定义完整的网络架构
        2. 构建 TF 计算图（包括输入占位符、loss、accuracy 等）
        3. 支持从 PyTorch 导出的权重字典（net_weights）注入

    网络架构（由 net_config 描述）:
        - first_conv:        初始 3x3 卷积（stride 2）
        - blocks:            N 个 MobileInvertedResidualBlock 序列
        - feature_mix_layer: 可选的 1x1 特征混合卷积
        - global_avg_pool:   全局平均池化
        - classifier:        1x1 卷积形式的分类器

    参数说明:
        net_config:   网络架构配置字典（JSON 格式）
        net_weights:  PyTorch 导出的权重初始化器字典（可空）
        graph:        外部 TF Graph（可空，为空则新建）
        sess:         外部 TF Session（可空，为空则新建）
        is_training:  是否为训练模式（推理时为 False）
        images:       外部输入图像张量（可空，为空则创建占位符）
        img_size:     输入图像尺寸（int 或 [h, w]）
        n_classes:    分类数（如 ImageNet 为 1000）
    """

    def __init__(
        self,
        net_config,
        net_weights=None,
        graph=None,
        sess=None,
        is_training=True,
        images=None,
        img_size=None,
        n_classes=1001,
    ):
        """初始化 ProxylessNASNets 网络

        构建流程:
            1. 设置计算图（外部传入或新建）
            2. 定义输入占位符（或使用外部传入的 images）
            3. 构建网络前向计算图
            4. 定义 loss 和 accuracy 计算
            5. 初始化 session
        """
        # ---- 计算图设置 ----
        if graph is not None:
            # 使用外部传入的 graph（如 generate_tflite.py 中的场景）
            self.graph = graph
            slim = True  # 使用 is_training 的固定值而非占位符
        else:
            # 新建一个空的 TF 计算图
            self.graph = tf.Graph()
            slim = False  # 使用占位符控制 is_training

        self.net_config = net_config
        self.n_classes = n_classes

        # ---- 在计算图上下文中构建网络 ----
        with self.graph.as_default():
            # 定义输入（images、labels、is_training）
            self._define_inputs(
                slim=slim, is_training=is_training, images=images, img_size=img_size
            )

            # 构建前向传播图，返回 logits
            logits = self.build(init=net_weights)
            self.logits = logits

            # Softmax 概率输出（用于推理时的分类结果）
            soft_logit = tf.nn.softmax(logits, dim=1)

            # 预测结果 = logits（argmax 在外部计算）
            prediction = logits

            # ---- 定义训练相关 ops（即使推理时也会定义） ----
            # 交叉熵损失
            # losses
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=self.labels
                )
            )
            self.cross_entropy = cross_entropy

            # 准确率计算
            correct_prediction = tf.equal(
                tf.argmax(prediction, 1), tf.argmax(self.labels, 1)
            )
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # 全局变量初始化器
            self.global_variables_initializer = tf.global_variables_initializer()

        # 初始化 session（创建并运行初始化器）
        self._initialize_session(sess)

    # ---- 属性访问器 ----

    @property
    def bn_eps(self):
        """BN 的 epsilon 参数

        从 net_config 中读取：net_config['bn']['eps']
        默认值通常为 1e-3
        """
        return self.net_config["bn"]["eps"]

    @property
    def bn_decay(self):
        """BN 的指数衰减系数

        从 net_config 中的 momentum 转换而来：
            bn_decay = 1 - momentum
        这是因为 TF 使用 decay（滑动平均衰减率），
        而 PyTorch 使用 momentum（当前 batch 的权重）。

        例如：PyTorch momentum=0.9 → TF decay=0.1
        """
        return 1 - self.net_config["bn"]["momentum"]

    # ---- Session 管理 ----

    def _initialize_session(self, sess):
        """初始化 TF session 并运行变量初始化器

        参数:
            sess: 外部传入的 session，为 None 则新建

        GPU 配置说明:
            默认不限制 GPU 内存使用（allow_growth = False）
            这是因为在 TFLite 转换场景中可能没有 GPU
        """
        # Initialize session, variables
        config = (
            tf.ConfigProto()
        )  # allow_soft_placement=True, log_device_placement=False
        # 限制模型 GPU 内存使用为按需增长（注释掉以兼容无 GPU 环境）
        # restrict model GPU memory utilization to min required
        # config.gpu_options.allow_growth = True
        if sess is None:
            # 创建新的 TF session
            self.sess = tf.Session(graph=self.graph, config=config)
        else:
            # 使用外部传入的 session
            self.sess = sess
        # 运行全局变量初始化器
        # 注意：如果使用了 net_weights（constant_initializer），
        # 在此步骤会将 PyTorch 权重赋值给 TF 变量
        self.sess.run(self.global_variables_initializer)

    # ---- 输入定义 ----

    def _define_inputs(self, slim=False, is_training=True, images=None, img_size=None):
        """定义网络的输入占位符

        创建三个输入占位符：
            1. images:      输入图像 [batch, h, w, 3]
            2. labels:      one-hot 标签 [batch, n_classes]
            3. is_training: 训练/推理模式标志（slim=False 时需要）

        参数:
            slim:       如果为 True，is_training 使用固定值而非占位符
            is_training:训练模式标志（slim=True 时使用此固定值）
            images:     外部传入的图像张量（slim=True 时使用）
            img_size:   输入图像尺寸
        """
        # ---- 图像输入 ----
        if isinstance(img_size, list) or isinstance(img_size, tuple):
            # 宽高不同时使用 tuple
            assert len(img_size) == 2
            shape = [None, img_size[0], img_size[1], 3]
        else:
            # 正方形输入
            shape = [None, img_size, img_size, 3]

        if images is not None:
            # 使用外部传入的 images 张量（转换场景，由 generate_tflite.py 传入）
            self.images = images
        else:
            # 创建 placeholder，运行时通过 feed_dict 传入数据
            self.images = tf.placeholder(tf.float32, shape=shape, name="input_images")

        # ---- 标签输入 ----
        # one-hot 格式的标签，shape [batch, n_classes]
        self.labels = tf.placeholder(
            tf.float32, shape=[None, self.n_classes], name="labels"
        )

        # ---- 训练模式标志 ----
        if slim:
            # slim 模式下，is_training 是 Python 常量而非 Tensor
            # 适用于推理/转换场景（generate_tflite.py 中 is_training=False）
            self.is_training = is_training
        else:
            # 非 slim 模式，创建 bool 型 placeholder
            # 训练时通过 feed_dict 控制
            self.is_training = tf.placeholder(tf.bool, shape=[], name="is_training")

    # ---- 辅助方法 ----

    @staticmethod
    def labels_to_one_hot(n_classes, labels):
        """将类别索引标签转换为 one-hot 格式

        参数:
            n_classes: 类别总数
            labels:    类别索引数组，shape [batch]

        返回值:
            one-hot 编码数组，shape [batch, n_classes]

        示例:
            若 n_classes=5, labels=[0, 2, 4]
            返回: [[1,0,0,0,0], [0,0,1,0,0], [0,0,0,0,1]]
        """
        new_labels = np.zeros((labels.shape[0], n_classes), dtype=np.float32)
        new_labels[range(labels.shape[0]), labels] = np.ones(labels.shape)
        return new_labels

    # ---- 网络构建 ----

    def build(self, init=None):
        """构建整个网络的前向计算图

        这是模型架构定义的核心方法。它按照 net_config 中的描述，
        逐层构建 TF 计算图。

        参数:
            init: 权重初始化器字典（将 PyTorch 权重转换为 TF constant_initializer）

        返回值:
            logits: 网络的输出张量，shape [batch, n_classes]

        网络结构:
            input [b, h, w, 3]
              │
              ├─ first_conv (3x3, stride 2) + BN + ReLU6
              │
              ├─ blocks[0..N-1] (MobileInvertedResidualBlock)
              │    └─ 每个 block 包含：
              │         ├─ MBInvertedConvLayer (expand + depthwise + project)
              │         └─ optional residual addition
              │
              ├─ feature_mix_layer (1x1 conv + BN + ReLU6, optional)
              │
              ├─ global_avg_pool → [b, 1, 1, c]
              │
              └─ classifier (1x1 conv, no BN, use_bias=True) → [b, n_classes]

        关于 blocks 的命名约定:
            - blocks[i] 的 TF scope 为 'blocks/i'
            - 每个 block 内的 MBInvertedConvLayer 固定名为 'mobile_inverted_conv'
            - shortcut（残差连接）不使用单独的变量作用域
        """
        output = self.images

        # ---- 权重初始化器转换 ----
        # 将 NumPy 数组包装为 TF constant_initializer
        # 这样在 session 初始化时会将权重值赋给对应的 TF 变量
        if init is not None:
            for key in init:
                init[key] = tf.constant_initializer(init[key])

        # ---- 第 1 层：first_conv（初始卷积层） ----
        # 将 3 通道 RGB 输入映射到初始特征空间
        first_conv = ConvLayer(
            "first_conv",
            self.net_config["first_conv"]["out_channels"],  # 输出通道数
            3,  # kernel_size = 3
            2,
        )  # stride = 2（快速降采样）
        output = first_conv.build(output, self, init)

        # ---- 主体 blocks（N 个 MobileInvertedResidualBlock） ----
        for i, block_config in enumerate(self.net_config["blocks"]):
            # 跳过 ZeroLayer（NAS 搜索中被剪枝掉的层）
            if block_config["mobile_inverted_conv"]["name"] == "ZeroLayer":
                continue

            # 创建 MBInvertedConvLayer 实例
            mobile_inverted_conv = MBInvertedConvLayer(
                "mobile_inverted_conv",
                block_config["mobile_inverted_conv"]["out_channels"],
                block_config["mobile_inverted_conv"]["kernel_size"],
                block_config["mobile_inverted_conv"]["stride"],
                block_config["mobile_inverted_conv"]["expand_ratio"],
            )

            # 判断是否有残差连接
            # shortcut 为 None 或 ZeroLayer 时表示无残差连接
            if (
                block_config["shortcut"] is None
                or block_config["shortcut"]["name"] == "ZeroLayer"
            ):
                has_residual = False
            else:
                has_residual = True

            # 创建 MobileInvertedResidualBlock（包含 MBInvertedConvLayer + 残差）
            block = MobileInvertedResidualBlock(
                "blocks/%d" % i, mobile_inverted_conv, has_residual
            )
            output = block.build(output, self, init)

        # ---- 特征混合层（可选） ----
        # 在 backbone 的最后加一个 1x1 卷积进行通道融合
        # 这个层在 MCUNet 的某些变体中存在（如 MCUNet-ViT）
        if self.net_config["feature_mix_layer"] is not None:
            feature_mix_layer = ConvLayer(
                "feature_mix_layer",
                self.net_config["feature_mix_layer"]["out_channels"],
                1,  # kernel_size = 1（点卷积）
                1,
            )  # stride = 1
            output = feature_mix_layer.build(output, self, init)

        # ---- 全局平均池化 ----
        # 将 [b, h, w, c] 降采样到 [b, 1, 1, c]
        # 池化窗口大小等于特征图尺寸
        output = avg_pool(output, output.get_shape()[1], output.get_shape()[2])

        # ---- 分类器 ----
        # 使用 1x1 卷积（ConvLayer_fc）替代传统全连接层
        # 这样在 TFLite 中只有卷积算子，算子类型更少，兼容性更好
        classifier = ConvLayer_fc(
            "classifier",
            self.n_classes,
            1,  # kernel_size = 1
            1,
        )  # stride = 1
        output = classifier.build(output, self, init)

        # 将 [b, 1, 1, n_classes] reshape 为 [b, n_classes]
        output = tf.reshape(output, shape=[-1, self.n_classes])
        return output

    # ---- 被注释掉的旧版 build 实现 ----
    # 以下是被 MNASNet 风格命名（expanded_conv、expanded_conv_0 等）的旧版实现
    # 与当前版本的主要区别在于 variable_scope 的命名方式不同
    # def build(self, init=None):
    #     output = self.images
    #     if init is not None:
    #         for key in init:
    #             init[key] = tf.constant_initializer(init[key])
    #
    #     # first conv
    #     first_conv = ConvLayer(
    #         'Conv',
    #         self.net_config['first_conv']['out_channels'],
    #         3,
    #         2)
    #     output = first_conv.build(output, self, init)
    #
    #     for i, block_config in enumerate(self.net_config['blocks']):
    #         if block_config['mobile_inverted_conv']['name'] == 'ZeroLayer':
    #             continue
    #         mobile_inverted_conv = MBInvertedConvLayer(
    #             '',
    #             block_config['mobile_inverted_conv']['out_channels'],
    #             block_config['mobile_inverted_conv']['kernel_size'],
    #             block_config['mobile_inverted_conv']['stride'],
    #             block_config['mobile_inverted_conv']['expand_ratio'],
    #         )
    #         if block_config['shortcut'] is None or block_config['shortcut']['name'] == 'ZeroLayer':
    #             has_residual = False
    #         else:
    #             has_residual = True
    #         if i == 0:
    #             block = MobileInvertedResidualBlock(
    #                 'expanded_conv'
    #                 , mobile_inverted_conv, has_residual)
    #         elif i <= 3:
    #             block = MobileInvertedResidualBlock(
    #                 'expanded_conv_%d' %
    #                 i, mobile_inverted_conv, has_residual)
    #         else:
    #             block = MobileInvertedResidualBlock(
    #                 'expanded_conv_%d' %
    #                 (i-2), mobile_inverted_conv, has_residual)
    #         output = block.build(output, self, init)
    #
    #     # feature mix layer
    #     feature_mix_layer = ConvLayer(
    #         'Conv_1',
    #         self.net_config['feature_mix_layer']['out_channels'],
    #         1,
    #         1)
    #     output = feature_mix_layer.build(output, self, init)
    #
    #     output = avg_pool(output, 7, 7)
    #     output = flatten(output)
    #     classifier = LinearLayer(
    #         'Logits/Conv2d_1c_1x1',
    #         self.n_classes,
    #         self.net_config['classifier']['dropout_rate'])
    #     output = classifier.build(output, self, init)
    #     return output
