# # **MIT 6.5940 EfficientML.ai 实验 3：神经架构搜索**
# 作者：MIT HAN Lab

# ## 简介

# 本 Colab notebook 为实验 3（神经架构搜索）提供了代码和框架。在本实验中，你将学习如何搜索出一个能在微控制器上高效运行的微型神经网络。你可以在这里完成你的解答。

"""
长期以来，研究人员都是手工设计神经网络架构。神经网络架构的设计空间非常庞大：它包括层数、通道宽度、分支数、卷积核大小和输入分辨率。因此，手工调节这些设计旋钮出了名地困难。而**神经架构搜索**（**NAS**）则可以帮助研究人员在各类效率和精度约束下自动调节这些设计旋钮。它大幅节省了神经网络设计的工程成本，并有助于推动 AI 的普及。在本实验中，我们将带你从零开始实现神经架构搜索。

"""

"""
早期的 NAS 方法会在设计空间中对候选网络进行穷举式训练，并使用基于 **RNN 的控制器**配合强化学习来优化采样策略。代表性方法包括 [Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578)、[NASNet](https://arxiv.org/abs/1707.07012) 和 [MNASNet](https://arxiv.org/abs/1807.11626)。这些方法的计算开销通常很大，因为每个候选网络都必须从头训练，基于 RNN 的控制器才能获得奖励信号（即候选网络的精度）。

后来，研究人员开发出了**可微分 NAS** 方法，例如 [DARTS](https://arxiv.org/abs/1806.09055)、[ProxylessNAS](https://arxiv.org/abs/1812.00332) 和 [FBNet](https://arxiv.org/abs/1812.03443)，大幅降低了训练候选网络的总成本。DARTS 将每一层的输出建模为不同候选操作输出的加权平均，而 ProxylessNAS 则通过在内存中只保留两条路径（而非全部路径）进一步降低了 DARTS 的内存开销。此后的**单次（one-shot）**方法，如 [Single Path One Shot](https://arxiv.org/abs/1904.00420)，进一步发现训练过程中每次只保留一条路径也是可行的。

尽管可微分 NAS 和单次 NAS 比基于控制器的方法高效得多，但每当我们要设计一个新的神经网络时，仍然需要重新跑完整的训练、搜索和微调流程。考虑到边缘设备数量庞大（例如截至 2018 年已有 [超过 200 亿台 IoT 设备](https://www.statista.com/statistics/471264/iot-number-of-connected-devices-worldwide/)），这给模型定制带来了高昂的成本（在 ImageNet 数据集上通常需要 200-300 GPU 小时）。

"""

"""
因此，在本实验中我们采用 [Once for All](https://arxiv.org/abs/1908.09791)（OFA），这是一种能大幅降低为不同设备定制神经网络架构成本的方法。OFA 训练一个包含设计空间内所有**子网络（sub-networks）**的大型**超网络（super network）**。如果我们直接从超网络中提取子网络，它们能达到与从头训练相近的精度。因此，OFA 支持**无需重新训练**的直接部署。

此外，OFA 还引入了**精度预测器和效率预测器**，以进一步降低架构搜索过程中的评估成本。直观来看，要得到一个子网络的精度，需要在整个保留验证集上跑一次推理，这在 ImageNet 上大约需要 1 分钟。OFA 的做法则是预先收集大量的（架构，精度）数据对，并训练一个回归模型来在搜索时**预测**精度。这将获取每个子网络精度反馈的成本从 1 分钟大幅降低到不足 1 秒。类似的思路也可以用于效率预测器——由于必须多次运行候选网络的前向传播，**延迟（latency）**的评估通常非常缓慢。

"""

# 在本实验中，你将研究如何借助 **OFA** 和**预测器**，为资源极度受限的微控制器搜索出可高效运行的网络。微控制器是一种低成本、低功耗的硬件。它们部署广泛，应用场景丰富。

# 但紧张的内存预算（比 GPU 小 50,000 倍）使得深度学习的部署变得困难。

# 本实验主要分为两个部分：**精度与效率预测器**以及**架构搜索**。

# - 预测器部分共有 ***4*** 道题。其中一道题（5 分）在 **Getting Started** 部分，另外三道题（30 分）在 **Predictors** 部分。
# - 架构搜索部分共有 ***6*** 道题。

# 首先，安装所需的软件包，并下载本实验将使用的 [**Visual Wake Words** 数据集](https://arxiv.org/abs/1906.05721)。
# 在 Colab 中运行：!sudo apt-get install graphviz; !wget ...; !pip install thop onnx

import argparse
import json
from PIL import Image
from tqdm import tqdm
import copy
import math
import numpy as np
import os
import random
import torch
from torch import nn
from torchvision import datasets, transforms
from mcunet.tinynas.search.accuracy_predictor import (
    AccuracyDataset,
    MCUNetArchEncoder,
)

from mcunet.tinynas.elastic_nn.networks.ofa_mcunets import OFAMCUNets
from mcunet.utils.mcunet_eval_helper import calib_bn, validate
from mcunet.utils.arch_visualization_helper import draw_arch
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ## **Getting Started：超网络与 VWW 数据集（1 道题，5 分）**

"""
在本实验中，我们将使用以 **once-for-all（OFA）** 方式训练的 **[MCUNetV2](https://arxiv.org/abs/2110.15352)** *超网络*。回想一下，*超网络*是一个随机化的大型神经网络，它包含设计空间内所有的候选子网络。我们可以直接从超网络中提取子网络并评估它们的精度。该精度可以进一步用作反馈信号来指导神经网络设计。OFA 超网络的优势在于，直接提取出的子网络能够达到与从头训练相当（甚至更好）的性能。

MCUNetV2 是一系列专为资源受限微控制器量身打造的高效神经网络。它采用了基于 patch 的推理、感受野重分配以及系统-神经网络协同设计，大幅改善了 [MCUNet](https://arxiv.org/abs/2007.10319) 的精度-效率权衡。

"""

"""
我们先来可视化 VWW 数据集中的一些样本。这是一个二分类图像分类数据集（判断图像中是否有人），从 [Microsoft COCO](https://arxiv.org/abs/1405.0312) 中子采样得到。我们首先定义一个函数，用于在验证集上构建一个 dataloader。

注意：函数 `build_val_data_loader` 有一个参数 `split`。我们用 `split = 0`（默认值）表示验证集（不能直接用于架构搜索），而 `split = 1` 将用作保留的 minival 集（用于生成精度数据集并校准 BN 参数）。

"""


# 加载数据集
def build_val_data_loader(data_dir, resolution, batch_size=128, split=0):
    # split = 0: real val set, split = 1: holdout validation set
    assert split in [0, 1]
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    kwargs = {"num_workers": min(8, os.cpu_count()), "pin_memory": False}

    val_transform = transforms.Compose(
        [
            transforms.Resize(
                (resolution, resolution)
            ),  # if center crop, the person might be excluded
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_dataset = datasets.ImageFolder(data_dir, transform=val_transform)

    val_dataset = torch.utils.data.Subset(
        val_dataset, list(range(len(val_dataset)))[split::2]
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **kwargs
    )
    return val_loader


# 借助这个 dataloader 构建器，我们就能浏览 VWW 验证集。你可以多次运行下面的单元格，查看数据集中不同的图像。

data_dir = "data/vww-s256/val"

val_data_loader = build_val_data_loader(data_dir, resolution=128, batch_size=1)

vis_x, vis_y = 2, 3
fig, axs = plt.subplots(vis_x, vis_y)

num_images = 0
for data, label in val_data_loader:
    img = np.array((((data + 1) / 2) * 255).numpy(), dtype=np.uint8)
    img = img[0].transpose(1, 2, 0)
    if label.item() == 0:
        label_text = "No person"
    else:
        label_text = "Person"
    axs[num_images // vis_y][num_images % vis_y].imshow(img)
    axs[num_images // vis_y][num_images % vis_y].set_title(f"Label: {label_text}")
    axs[num_images // vis_y][num_images % vis_y].set_xticks([])
    axs[num_images // vis_y][num_images % vis_y].set_yticks([])
    num_images += 1
    if num_images > vis_x * vis_y - 1:
        break

plt.show()

"""
很好，现在你对这个数据集已经有了基本的认识。接下来我们来构建 OFA 超网络！`OFAMCUNets` 超网络由 MCUNetV2 设计空间中 $>10^{19}$ 个子网络组成。这些子网络由不同卷积核大小（3、5、7）和扩展比（3、4、6）的[反向 MobileNet block](https://arxiv.org/abs/1801.04381) 构成。OFA 超网络还允许所有网络阶段使用弹性深度（从 base depth 到 base_depth + 2）。最后，超网络支持以 0.5$\times$、0.75$\times$ 或 1.0$\times$ 进行全局通道缩放（由 `width_mult_list` 指定）。

"""

device = "cuda:0"
ofa_network = OFAMCUNets(
    n_classes=2,
    bn_param=(0.1, 1e-3),
    dropout_rate=0.0,
    base_stage_width="mcunet384",
    width_mult_list=[0.5, 0.75, 1.0],
    ks_list=[3, 5, 7],
    expand_ratio_list=[3, 4, 6],
    depth_list=[0, 1, 2],
    base_depth=[1, 2, 2, 2, 2],
    fuse_blk1=True,
    se_stages=[False, [False, True, True, True], True, True, True, False],
)

ofa_network.load_state_dict(
    torch.load("vww_supernet.pth", map_location="cpu")["state_dict"], strict=True
)

ofa_network = ofa_network.to(device)

"""
接下来我们验证 checkpoint 是否被正确加载。我们将在 MCUNetV2 设计空间中采样一些网络，并在 VWW 数据集上评估其精度。评估用时不到一分钟，你应该会看到大约 83.6-88.7% 的精度。可以看到，我们能够直接从设计空间中提取这些子网络，并在**无需训练**的情况下快速得到它们的精度。这正是 once-for-all（OFA）超网络带来的独特优势。

我们先定义一个辅助函数 `evaluate_sub_network`，用于直接测试从超网络中提取出的子网络的精度。

"""

from mcunet.utils.pytorch_utils import (
    count_peak_activation_size,
    count_net_flops,
    count_parameters,
)


def evaluate_sub_network(ofa_network, cfg, image_size=None):
    if "image_size" in cfg:
        image_size = cfg["image_size"]
    batch_size = 128
    # step 1. sample the active subnet with the given config.
    ofa_network.set_active_subnet(**cfg)
    # step 2. extract the subnet with corresponding weights.
    subnet = ofa_network.get_active_subnet().to(device)
    # step 3. calculate the efficiency stats of the subnet.
    peak_memory = count_peak_activation_size(subnet, (1, 3, image_size, image_size))
    macs = count_net_flops(subnet, (1, 3, image_size, image_size))
    params = count_parameters(subnet)
    # step 4. perform BN parameter re-calibration.
    calib_bn(subnet, data_dir, batch_size, image_size)
    # step 5. define the validation dataloader.
    val_loader = build_val_data_loader(data_dir, image_size, batch_size)
    # step 6. validate the accuracy.
    acc = validate(subnet, val_loader)
    return acc, peak_memory, macs, params


# 我们还提供了一个方便的辅助函数，用于可视化子网络的架构。该函数接收子网络的配置，并返回一张表示其架构的图像。


def visualize_subnet(cfg):
    draw_arch(cfg["ks"], cfg["e"], cfg["d"], cfg["image_size"], out_name="viz/subnet")
    im = Image.open("viz/subnet.png")
    im = im.rotate(90, expand=1)
    fig = plt.figure(figsize=(im.size[0] / 250, im.size[1] / 250))
    plt.axis("off")
    plt.imshow(im)
    plt.show()


"""
现在，我们来可视化一些子网络，并在 VWW 数据集上对它们进行评估！我们提供了一个示例，从设计空间中随机采样一个子网络，并得到它在 VWW 数据集上的精度、MACs 和参数量。我们还会使用 `visualize_subnet` 来可视化其架构。

在架构可视化中，每个 block 的图例 `MBConv{e}-{k}x{k}` 表示当前 block 是一个扩展比为 `e`、深度可分离卷积层卷积核大小为 `k` 的反向移动 block。block 的不同颜色表示不同的卷积核大小，灰色 block 是网络阶段的分隔符。block 的不同宽度表示不同的扩展比。我们还在每个 block 旁标注了输出分辨率。

注意，我们假设图像分辨率固定为 96。你可以随意在下面新增一个单元格，尝试不同的输入分辨率。

提示：你可以修改 `sample_active_subnet` 方法的 `sample_function` 参数来控制采样过程。

"""

image_size = 96

cfg = ofa_network.sample_active_subnet(
    sample_function=random.choice, image_size=image_size
)
acc, _, _, params = evaluate_sub_network(ofa_network, cfg)
visualize_subnet(cfg)
print(
    f"The accuracy of the sampled subnet: #params={params / 1e6: .1f}M, accuracy={acc: .1f}%."
)

largest_cfg = ofa_network.sample_active_subnet(
    sample_function=max, image_size=image_size
)
acc, _, _, params = evaluate_sub_network(ofa_network, largest_cfg)
visualize_subnet(largest_cfg)
print(f"The largest subnet: #params={params / 1e6: .1f}M, accuracy={acc: .1f}%.")

smallest_cfg = ofa_network.sample_active_subnet(
    sample_function=min, image_size=image_size
)
acc, peak_memory, macs, params = evaluate_sub_network(ofa_network, smallest_cfg)
visualize_subnet(smallest_cfg)
print(f"The smallest subnet: #params={params / 1e6: .1f}M, accuracy={acc: .1f}%.")

# ### 问题 1（5 分）：设计空间探索。

# 尝试多次运行上面的单元格，手动采样不同的子网络。你也可以改变输入分辨率。谈谈你的发现。

# 提示：哪个维度对精度起的作用最大？

# **答案：**（请填写）

# ## **第 1 部分：预测器（3 道题，30 分）**

# 神经架构搜索需要从 OFA 超网络中采样大量子网络，并评估这些子网络的性能。这样的性能评估非常耗时。

# 在本实验中，我们借助**效率预测器**和**精度预测器**来探索极快速的神经网络搜索。

"""
### 问题 2（10 分）：实现效率预测器。

对于效率预测器，我们使用一个基于 hook 的解析模型来统计给定网络的 #MACs 和峰值内存占用。我们使用提供的 API 从零开始构建它。

具体来说，我们定义一个名为 `AnalyticalEfficiencyPredictor` 的类。该类有两个主要函数：`get_efficiency` 和 `satisfy_constraint`。

函数 `get_efficiency` 接收子网络配置，并返回该子网络的 #MACs 和峰值内存。这里，我们假设 #MACs 的单位是百万（million），峰值内存占用的单位是 KB。

提示：看看上面的 `evaluate_sub_network` 函数。我们使用 `count_net_flops` 来获取网络的 MACs，使用 `count_peak_activation_size` 来获取网络的激活大小。

"""


class AnalyticalEfficiencyPredictor:
    def __init__(self, net):
        self.net = net

    def get_efficiency(self, spec: dict):
        self.net.set_active_subnet(**spec)
        subnet = self.net.get_active_subnet()
        if torch.cuda.is_available():
            subnet = subnet.cuda()
        ############### YOUR CODE STARTS HERE ###############
        # Hint: take a look at the `evaluate_sub_network` function above.
        # Hint: the data shape is (batch_size, input_channel, image_size, image_size)
        data_shape = (1, 3, spec["image_size"], spec["image_size"])
        macs = count_net_flops(subnet, data_shape)
        peak_memory = count_peak_activation_size(subnet, data_shape)
        ################ YOUR CODE ENDS HERE ################

        return dict(millionMACs=macs / 1e6, KBPeakMemory=peak_memory / 1024)

    def satisfy_constraint(self, measured: dict, target: dict):
        for key in measured:
            # if the constraint is not specified, we just continue
            if key not in target:
                continue
            # if we exceed the constraint, just return false.
            if measured[key] > target[key]:
                return False
        # no constraint violated, return true.
        return True


# 我们来测试一下你实现的解析式效率预测器：检查它对我们刚才评估过的最小子网络和最大子网络返回的值。效率预测器的结果应当与之前的结果一致。

efficiency_predictor = AnalyticalEfficiencyPredictor(ofa_network)

image_size = 96
# Print out the efficiency of the smallest subnet.
smallest_cfg = ofa_network.sample_active_subnet(
    sample_function=min, image_size=image_size
)
eff_smallest = efficiency_predictor.get_efficiency(smallest_cfg)

# Print out the efficiency of the largest subnet.
largest_cfg = ofa_network.sample_active_subnet(
    sample_function=max, image_size=image_size
)
eff_largest = efficiency_predictor.get_efficiency(largest_cfg)

print("Efficiency stats of the smallest subnet:", eff_smallest)
print("Efficiency stats of the largest subnet:", eff_largest)

"""
### 问题 3（10 分）：实现精度预测器。

精度预测器用于预测给定子网络在 VWW 数据集上的分类精度，这样我们在架构搜索过程中每遇到一个新子网络时，就**无需**每次都运行代价高昂的推理。这样的精度预测器是一个在用 OFA 网络构建的精度数据集上训练的 MLP（多层感知机）模型。MLP 网络的推理只需几毫秒，因此精度预测器可以将搜索过程加速**几个数量级**。

"""

# 精度预测器接收一个子网络的架构，并预测它在 VWW 数据集上的精度。由于它是一个 MLP 网络，子网络必须被编码成一个**向量**。在本实验中，我们提供了一个 `MCUNetArchEncoder` 类，用于完成从**子网络架构**到**二值向量**的转换。

image_size_list = [96, 112, 128, 144, 160]
arch_encoder = MCUNetArchEncoder(
    image_size_list=image_size_list,
    base_depth=ofa_network.base_depth,
    depth_list=ofa_network.depth_list,
    expand_list=ofa_network.expand_ratio_list,
    width_mult_list=ofa_network.width_mult_list,
)

"""
我们预先生成了一个精度数据集，它是一组 `[architecture, accuracy]` 数据对，存放在 `acc_datasets` 文件夹下。

有了架构编码器，现在请你定义精度预测器，它是一个每个中间层有 400 个通道的多层感知机（MLP）网络。为简单起见，我们将层数固定为 **3**。请在下面的单元格中实现这个 MLP 网络。

"""


class AccuracyPredictor(nn.Module):
    def __init__(
        self,
        arch_encoder,
        hidden_size=400,
        n_layers=3,
        checkpoint_path=None,
        device="cuda:0",
    ):
        super(AccuracyPredictor, self).__init__()
        self.arch_encoder = arch_encoder
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device

        layers = []

        ############### YOUR CODE STARTS HERE ###############
        # Let's build an MLP with n_layers layers.
        # Each layer (nn.Linear) has hidden_size channels and
        # uses nn.ReLU as the activation function.
        # Hint: You can assume that n_layers is fixed to be 3, for simplicity.
        # Hint: the input dimension of the first layer is not hidden_size.
        #       use self.arch_encoder.n_dim to get the input dimension
        for i in range(self.n_layers):
            in_dim = self.arch_encoder.n_dim if i == 0 else self.hidden_size
            layers.append(nn.Sequential(nn.Linear(in_dim, self.hidden_size), nn.ReLU()))
        ################ YOUR CODE ENDS HERE ################
        layers.append(nn.Linear(self.hidden_size, 1, bias=False))
        self.layers = nn.Sequential(*layers)
        self.base_acc = nn.Parameter(
            torch.zeros(1, device=self.device), requires_grad=False
        )

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            self.load_state_dict(checkpoint)
            print("Loaded checkpoint from %s" % checkpoint_path)

        self.layers = self.layers.to(self.device)

    def forward(self, x):
        y = self.layers(x).squeeze()
        return y + self.base_acc

    def predict_acc(self, arch_dict_list):
        X = [self.arch_encoder.arch2feature(arch_dict) for arch_dict in arch_dict_list]
        X = torch.tensor(np.array(X)).float().to(self.device)
        return self.forward(X)


# 我们来打印一下你刚刚定义的 `AccuracyPredictor` 的架构。

os.makedirs("pretrained", exist_ok=True)
acc_pred_checkpoint_path = (
    f"pretrained/{ofa_network.__class__.__name__}_acc_predictor.pth"
)
acc_predictor = AccuracyPredictor(
    arch_encoder,
    hidden_size=400,
    n_layers=3,
    checkpoint_path=None,
    device=device,
)
print(acc_predictor)

"""
我们先在下面的单元格中可视化精度数据集中的一些样本。

精度数据集由 50,000 个 `[architecture, accuracy]` 数据对组成，其中 40,000 个用作训练集，其余 10,000 个用作验证集。

对于**精度（accuracy）**，我们计算精度数据集中所有 `[architecture, accuracy]` 数据对的平均精度，并将其定义为 `base_acc`。对于精度预测器，我们并不直接回归每个架构的精度，而是将其训练目标设为 `accuracy - base_acc`。由于 `accuracy - base_acc` 通常比 `accuracy` 本身小得多，这能让训练更容易。

对于**架构（architecture）**，设计空间内的每个子网络都由一个二值向量唯一表示。该二值向量是全局参数（*例如*输入分辨率、宽度乘子）以及每个反向 MobileNet block 参数（*例如*卷积核大小和扩展比）的**独热（one-hot）表示**拼接而成。注意，我们更倾向于使用**独热（one-hot）**表示而非**数值（numerical）**表示，因为所有设计超参数都是**离散**值。

例如，我们的设计空间支持

```python kernel_size = [3, 5, 7] expand_ratio = [3, 4, 6] ```

那么，我们将 `kernel_size=3` 表示为 `[1, 0, 0]`，`kernel_size=5` 表示为 `[0, 1, 0]`，`kernel_size=7` 表示为 `[0, 0, 1]`。类似地，`expand_ratio=3` 写作 `[1, 0, 0]`，`expand_ratio=4` 写作 `[0, 1, 0]`，`expand_ratio=6` 写作 `[0, 0, 1]`。每个反向 MobileNet block 的表示通过将卷积核大小嵌入与扩展比嵌入拼接而得到。注意，对于被跳过的 block，我们用 `[0, 0, 0]` 来表示它们的卷积核大小和扩展比。运行下面的单元格后，你将看到关于架构-嵌入对应关系的详细说明。

"""

acc_dataset = AccuracyDataset("acc_datasets")
train_loader, valid_loader, base_acc = acc_dataset.build_acc_data_loader(
    arch_encoder=arch_encoder
)

print(
    f"The basic accuracy (mean accuracy of all subnets within the dataset is: {(base_acc * 100): .1f}%."
)

# Let's print one sample in the training set
sampled = 0
for data, label in train_loader:
    data = data.to(device)
    label = label.to(device)
    print("=" * 100)
    # dummy pass to print the divided encoding
    arch_encoding = arch_encoder.feature2arch(
        data[0].int().cpu().numpy(), verbose=False
    )
    # print out the architecture encoding process in detail
    arch_encoding = arch_encoder.feature2arch(data[0].int().cpu().numpy(), verbose=True)
    visualize_subnet(arch_encoding)
    print(
        f"The accuracy of this subnet on the holdout validation set is: {(label[0] * 100): .1f}%."
    )
    sampled += 1
    if sampled == 1:
        break

# ### 问题 4（10 分）：补全精度预测器训练的代码。

# 现在，我们用提供的数据集来训练精度预测器！在这一部分，你需要负责实现精度预测器的训练与验证。训练过程大约需要 1-2 分钟。

# 提示：关于如何用 PyTorch 训练神经网络，你可以参考 Tutorial 2。

criterion = torch.nn.L1Loss().to(device)
optimizer = torch.optim.Adam(acc_predictor.parameters())
# the default value is zero
acc_predictor.base_acc.data += base_acc
for epoch in tqdm(range(10)):
    acc_predictor.train()
    for data, label in tqdm(
        train_loader, desc="Epoch%d" % (epoch + 1), position=0, leave=True
    ):
        # step 1. Move the data and labels to device (cuda:0).
        data = data.to(device)
        label = label.to(device)
        ############### YOUR CODE STARTS HERE ###############
        # step 2. Run forward pass.
        pred = acc_predictor(data)
        # step 3. Calculate the loss.
        loss = criterion(pred, label)
        # step 4. Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ################ YOUR CODE ENDS HERE ################

    acc_predictor.eval()
    with torch.no_grad():
        with tqdm(total=len(valid_loader), desc="Val", position=0, leave=True) as t:
            for data, label in valid_loader:
                # step 1. Move the data and labels to device (cuda:0).
                data = data.to(device)
                label = label.to(device)
                ############### YOUR CODE STARTS HERE ###############
                # step 2. Run forward pass.
                pred = acc_predictor(data)
                # step 3. Calculate the loss.
                loss = criterion(pred, label)
                ############### YOUR CODE ENDS HERE ###############
                t.set_postfix({"loss": loss.item()})
                t.update(1)

if not os.path.exists(acc_pred_checkpoint_path):
    torch.save(acc_predictor.cpu().state_dict(), acc_pred_checkpoint_path)

# 现在，我们来绘制预测精度与真实精度的相关性图，确保我们的预测器是可靠的。要拿到满分，你应当在这一部分看到线性相关关系。

predicted_accuracies = []
ground_truth_accuracies = []
acc_predictor = acc_predictor.to("cuda:0")
acc_predictor.eval()
with torch.no_grad():
    with tqdm(total=len(valid_loader), desc="Val") as t:
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)
            pred = acc_predictor(data)
            predicted_accuracies += pred.cpu().numpy().tolist()
            ground_truth_accuracies += label.cpu().numpy().tolist()
            if len(predicted_accuracies) > 200:
                break
plt.scatter(predicted_accuracies, ground_truth_accuracies)
# draw y = x
min_acc, max_acc = min(predicted_accuracies), max(predicted_accuracies)
plt.plot([min_acc, max_acc], [min_acc, max_acc], c="red", linewidth=2)
plt.xlabel("Predicted accuracy")
plt.ylabel("Measured accuracy")
plt.title("Correlation between predicted accuracy and real accuracy")

# ## **第 2 部分：神经架构搜索（6 道题，65 分 + 10 分附加分）**

"""
到目前为止，我们已经定义好了效率预测器和精度预测器。让我们用这两个强大的预测器来开始快速的模型定制吧！

在这一部分，你需要实现两种典型的搜索算法：**随机搜索（random search）**和**进化搜索（evolutionary search）**。搜索算法的目标是找到在满足效率约束（例如 MACs、峰值内存）的同时提供最佳精度的模型架构。

"""

# ### 问题 5（5 分）：补全下面的随机搜索智能体。


class RandomSearcher:
    def __init__(self, efficiency_predictor, accuracy_predictor):
        self.efficiency_predictor = efficiency_predictor
        self.accuracy_predictor = accuracy_predictor

    def random_valid_sample(self, constraint):
        # randomly sample subnets until finding one that satisfies the constraint
        while True:
            sample = self.accuracy_predictor.arch_encoder.random_sample_arch()
            efficiency = self.efficiency_predictor.get_efficiency(sample)
            if self.efficiency_predictor.satisfy_constraint(efficiency, constraint):
                return sample, efficiency

    def run_search(self, constraint, n_subnets=100):
        subnet_pool = []
        # sample subnets
        for _ in tqdm(range(n_subnets)):
            sample, efficiency = self.random_valid_sample(constraint)
            subnet_pool.append(sample)
        # predict the accuracy of subnets
        accs = self.accuracy_predictor.predict_acc(subnet_pool)
        ############### YOUR CODE STARTS HERE ###############
        # hint: one line of code
        # get the index of the best subnet
        best_idx = torch.argmax(accs).item()
        ############### YOUR CODE ENDS HERE #################
        # return the best subnet
        return accs[best_idx], subnet_pool[best_idx]


# ### 问题 6（5 分）：补全下面的函数。


def search_and_measure_acc(agent, constraint, **kwargs):
    ############### YOUR CODE STARTS HERE ###############
    # hint: call the search function
    best_info = agent.run_search(constraint, **kwargs)
    ############### YOUR CODE ENDS HERE #################
    # get searched subnet
    ofa_network.set_active_subnet(**best_info[1])
    subnet = ofa_network.get_active_subnet().to(device)
    # calibrate bn
    calib_bn(subnet, data_dir, 128, best_info[1]["image_size"])
    # build val loader
    val_loader = build_val_data_loader(data_dir, best_info[1]["image_size"], 128)
    # measure accuracy
    acc = validate(subnet, val_loader)
    # print best_info
    print(f"Accuracy of the selected subnet: {acc}")
    # visualize model architecture
    visualize_subnet(best_info[1])
    return acc, subnet


random.seed(1)
np.random.seed(1)
nas_agent = RandomSearcher(efficiency_predictor, acc_predictor)
# MACs-constrained search
subnets_rs_macs = {}
for millonMACs in [50, 100]:
    search_constraint = dict(millonMACs=millonMACs)
    print(f"Random search with constraint: MACs <= {millonMACs}M")
    subnets_rs_macs[millonMACs] = search_and_measure_acc(
        nas_agent, search_constraint, n_subnets=300
    )

# memory-constrained search
subnets_rs_memory = {}
for KBPeakMemory in [256, 512]:
    search_constraint = dict(KBPeakMemory=KBPeakMemory)
    print(f"Random search with constraint: Peak memory <= {KBPeakMemory}KB")
    subnets_rs_memory[KBPeakMemory] = search_and_measure_acc(
        nas_agent, search_constraint, n_subnets=300
    )

# ### 问题 7（20 分）：补全下面的进化搜索智能体。

"""
现在你已经成功实现了随机搜索算法。在这一部分，我们将实现一种采样效率更高的搜索算法——进化搜索。进化搜索的灵感来自进化算法（或称遗传算法）。首先从设计空间中采样出一个由子网络组成的**种群（population）**。然后，在每一**代（generation）**中，我们执行随机变异和交叉操作，如上图所示。精度最高的子网络将被保留，这个过程会不断重复，直到代数达到 `max_time_budget`。与随机搜索类似，在整个搜索过程中，所有无法满足效率约束的子网络都将被丢弃。

"""


class EvolutionSearcher:
    def __init__(self, efficiency_predictor, accuracy_predictor, **kwargs):
        self.efficiency_predictor = efficiency_predictor
        self.accuracy_predictor = accuracy_predictor

        # evolution hyper-parameters
        self.arch_mutate_prob = kwargs.get("arch_mutate_prob", 0.1)
        self.resolution_mutate_prob = kwargs.get("resolution_mutate_prob", 0.5)
        self.population_size = kwargs.get("population_size", 100)
        self.max_time_budget = kwargs.get("max_time_budget", 500)
        self.parent_ratio = kwargs.get("parent_ratio", 0.25)
        self.mutation_ratio = kwargs.get("mutation_ratio", 0.5)

    def update_hyper_params(self, new_param_dict):
        self.__dict__.update(new_param_dict)

    def random_valid_sample(self, constraint):
        # randomly sample subnets until finding one that satisfies the constraint
        while True:
            sample = self.accuracy_predictor.arch_encoder.random_sample_arch()
            efficiency = self.efficiency_predictor.get_efficiency(sample)
            if self.efficiency_predictor.satisfy_constraint(efficiency, constraint):
                return sample, efficiency

    def mutate_sample(self, sample, constraint):
        while True:
            new_sample = copy.deepcopy(sample)

            self.accuracy_predictor.arch_encoder.mutate_resolution(
                new_sample, self.resolution_mutate_prob
            )
            self.accuracy_predictor.arch_encoder.mutate_width(
                new_sample, self.arch_mutate_prob
            )
            self.accuracy_predictor.arch_encoder.mutate_arch(
                new_sample, self.arch_mutate_prob
            )

            efficiency = self.efficiency_predictor.get_efficiency(new_sample)
            if self.efficiency_predictor.satisfy_constraint(efficiency, constraint):
                return new_sample, efficiency

    def crossover_sample(self, sample1, sample2, constraint):
        while True:
            new_sample = copy.deepcopy(sample1)
            for key in new_sample.keys():
                if not isinstance(new_sample[key], list):
                    ############### YOUR CODE STARTS HERE ###############
                    # hint: randomly choose the value from sample1[key] and sample2[key], random.choice
                    new_sample[key] = random.choice([sample1[key], sample2[key]])
                    ############### YOUR CODE ENDS HERE #################
                else:
                    for i in range(len(new_sample[key])):
                        ############### YOUR CODE STARTS HERE ###############
                        new_sample[key][i] = random.choice(
                            [sample1[key][i], sample2[key][i]]
                        )
                        ############### YOUR CODE ENDS HERE #################

            efficiency = self.efficiency_predictor.get_efficiency(new_sample)
            if self.efficiency_predictor.satisfy_constraint(efficiency, constraint):
                return new_sample, efficiency

    def run_search(self, constraint, **kwargs):
        self.update_hyper_params(kwargs)

        mutation_numbers = int(round(self.mutation_ratio * self.population_size))
        parents_size = int(round(self.parent_ratio * self.population_size))

        best_valids = [-100]
        population = []  # (acc, sample) tuples
        child_pool = []
        best_info = None
        # generate random population
        for _ in range(self.population_size):
            sample, efficiency = self.random_valid_sample(constraint)
            child_pool.append(sample)

        accs = self.accuracy_predictor.predict_acc(child_pool)
        for i in range(self.population_size):
            population.append((accs[i].item(), child_pool[i]))

        # evolving the population
        with tqdm(total=self.max_time_budget) as t:
            for i in range(self.max_time_budget):
                ############### YOUR CODE STARTS HERE ###############
                # hint: sort the population according to the acc (descending order)
                population = sorted(population, key=lambda x: x[0], reverse=True)
                ############### YOUR CODE ENDS HERE #################

                ############### YOUR CODE STARTS HERE ###############
                # hint: keep topK samples in the population, K = parents_size
                # the others are discarded.
                population = population[:parents_size]
                ############### YOUR CODE ENDS HERE #################

                # update best info
                acc = population[0][0]
                if acc > best_valids[-1]:
                    best_valids.append(acc)
                    best_info = population[0]
                else:
                    best_valids.append(best_valids[-1])

                child_pool = []
                for j in range(mutation_numbers):
                    # randomly choose a sample
                    par_sample = population[np.random.randint(parents_size)][1]
                    # mutate this sample
                    new_sample, efficiency = self.mutate_sample(par_sample, constraint)
                    child_pool.append(new_sample)

                for j in range(self.population_size - mutation_numbers):
                    # randomly choose two samples
                    par_sample1 = population[np.random.randint(parents_size)][1]
                    par_sample2 = population[np.random.randint(parents_size)][1]
                    # crossover
                    new_sample, efficiency = self.crossover_sample(
                        par_sample1, par_sample2, constraint
                    )
                    child_pool.append(new_sample)
                # predict accuracy with the accuracy predictor
                accs = self.accuracy_predictor.predict_acc(child_pool)
                for j in range(self.population_size):
                    population.append((accs[j].item(), child_pool[j]))

                t.update(1)

        return best_info


# ### 问题 8（10 分）：运行进化搜索，并调节 evo_params 以优化结果。描述你的发现。

random.seed(1)
np.random.seed(1)

# hint: tune hyper-parameters below
evo_params = {
    "arch_mutate_prob": 0.1,  # The probability of architecture mutation in evolutionary search
    "resolution_mutate_prob": 0.1,  # The probability of resolution mutation in evolutionary search
    "population_size": 10,  # The size of the population
    "max_time_budget": 10,
    "parent_ratio": 0.1,
    "mutation_ratio": 0.1,
}

nas_agent = EvolutionSearcher(efficiency_predictor, acc_predictor, **evo_params)
# MACs-constrained search
subnets_evo_macs = {}
for millonMACs in [50, 100]:
    search_constraint = dict(millionMACs=millonMACs)
    print(f"Evolutionary search with constraint: MACs <= {millonMACs}M")
    subnets_evo_macs[millonMACs] = search_and_measure_acc(nas_agent, search_constraint)

# memory-constrained search
subnets_evo_memory = {}
for KBPeakMemory in [256, 512]:
    search_constraint = dict(KBPeakMemory=KBPeakMemory)
    print(f"Evolutionary search with constraint: Peak memory <= {KBPeakMemory}KB")
    subnets_evo_memory[KBPeakMemory] = search_and_measure_acc(
        nas_agent, search_constraint
    )

"""
### 问题 9（15 分 + 10 分附加分）：在真实世界的约束下运行进化搜索。

在实际应用中，我们可能会有多个效率约束：https://blog.tensorflow.org/2019/10/visual-wake-words-with-tensorflow-lite_30.html 。 使用进化搜索来寻找满足以下约束的模型： - [15 分] 250 KB，60M MACs（精度 >= 92.5% 可得满分） - [10 分，**附加分**] 200KB，30M MACs（精度 >= 90% 可得满分）

提示：这两个任务不必使用相同的 `evo_params`。

"""

random.seed(1)
np.random.seed(1)
# hint: tune hyper-parameters below
evo_params = {
    "arch_mutate_prob": 0.1,  # The probability of architecture mutation in evolutionary search
    "resolution_mutate_prob": 0.1,  # The probability of resolution mutation in evolutionary search
    "population_size": 10,  # The size of the population
    "max_time_budget": 10,
    "parent_ratio": 0.1,
    "mutation_ratio": 0.1,
}

nas_agent = EvolutionSearcher(efficiency_predictor, acc_predictor, **evo_params)

(millionMACs, KBPeakMemory) = [60, 250]
print(
    f"Evolution search with constraint: MACs <= {millionMACs}M, peak memory <= {KBPeakMemory}KB"
)
search_and_measure_acc(
    nas_agent, dict(millionMACs=millionMACs, KBPeakMemory=KBPeakMemory)
)
print("Evolution search finished!")

random.seed(1)
np.random.seed(1)
# hint: tune hyper-parameters below
evo_params = {
    "arch_mutate_prob": 0.1,  # The probability of architecture mutation in evolutionary search
    "resolution_mutate_prob": 0.1,  # The probability of resolution mutation in evolutionary search
    "population_size": 10,  # The size of the population
    "max_time_budget": 10,
    "parent_ratio": 0.1,
    "mutation_ratio": 0.1,
}

nas_agent = EvolutionSearcher(efficiency_predictor, acc_predictor, **evo_params)

(millionMACs, KBPeakMemory) = [30, 200]
print(
    f"Evolution search with constraint: MACs <= {millionMACs}M, peak memory <= {KBPeakMemory}KB"
)
search_and_measure_acc(
    nas_agent, dict(millionMACs=millionMACs, KBPeakMemory=KBPeakMemory)
)
print("Evolution search finished!")

# ### 问题 10（10 分）：在当前设计空间中，是否有可能找到满足以下效率约束的子网络？
# - A：子网络的激活大小**至多 256KB**，且子网络的 MACs **至多 15M**。
# - B：子网络的激活大小**至多 64 KB**。
