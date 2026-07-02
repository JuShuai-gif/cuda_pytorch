"""
目标：
- 理解量化的基本概念
- 实现并应用 k-means 量化
- 实现并应用 k-means 量化的量化感知训练
- 实现并应用线性量化
- 实现并应用线性量化的纯整数推理
- 初步了解量化带来的性能提升(如加速效果)
- 理解这些量化方法之间的差异与权衡
"""

import copy
import math
import random
from collections import OrderedDict, defaultdict

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchprofile import profile_macs
from torchvision.datasets import *
from torchvision.transforms import *

from torchprofile import profile_macs

assert torch.cuda.is_available(), (
    "The current runtime does not have CUDA support."
    "Please go to menu bar (Runtime - Change runtime type) and select GPU"
)


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def download_url(url, model_dir=".", overwrite=False):
    import os, sys
    from urllib.request import urlretrieve

    target_dir = url.split("/")[-1]
    model_dir = os.path.expanduser(model_dir)
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_dir = os.path.join(model_dir, target_dir)
        cached_file = model_dir
        if not os.path.exists(cached_file) or overwrite:
            sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
            urlretrieve(url, cached_file)
        return cached_file
    except Exception as e:
        # remove lock file so download can be executed next time.
        os.remove(os.path.join(model_dir, "download.lock"))
        sys.stderr.write("Failed to download from url %s" % url + "\n" + str(e) + "\n")
        return None


class VGG(nn.Module):
    ARCH = [64, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

    def __init__(self) -> None:
        super().__init__()

        layers = []
        counts = defaultdict(int)

        def add(name: str, layer: nn.Module) -> None:
            layers.append((f"{name}{counts[name]}", layer))
            counts[name] += 1

        in_channels = 3
        for x in self.ARCH:
            if x != "M":
                # conv-bn-relu
                add("conv", nn.Conv2d(in_channels, x, 3, padding=1, bias=False))
                add("bn", nn.BatchNorm2d(x))
                add("relu", nn.ReLU(True))
                in_channels = x
            else:
                # maxpool
                add("pool", nn.MaxPool2d(2))
        add("avgpool", nn.AvgPool2d(2))
        self.backbone = nn.Sequential(OrderedDict(layers))
        self.classifier = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # backbone: [N, 3, 32, 32] => [N, 512, 2, 2]
        x = self.backbone(x)

        # avgpool: [N, 512, 2, 2] => [N, 512]
        # x = x.mean([2, 3])
        x = x.view(x.shape[0], -1)

        # classifier: [N, 512] => [N, 10]
        x = self.classifier(x)
        return x


def train(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    callbacks=None,
) -> None:
    model.train()

    for inputs, targets in tqdm(dataloader, desc="train", leave=False):
        # Move the data from CPU to GPU
        inputs = inputs.cuda()
        targets = targets.cuda()

        # Reset the gradients (from the last iteration)
        optimizer.zero_grad()

        # Forward inference
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward propagation
        loss.backward()

        # Update optimizer and LR scheduler
        optimizer.step()
        scheduler.step()

        if callbacks is not None:
            for callback in callbacks:
                callback()


@torch.inference_mode()
def evaluate(model: nn.Module, dataloader: DataLoader, extra_preprocess=None) -> float:
    model.eval()

    num_samples = 0
    num_correct = 0

    for inputs, targets in tqdm(dataloader, desc="eval", leave=False):
        # Move the data from CPU to GPU
        inputs = inputs.cuda()
        if extra_preprocess is not None:
            for preprocess in extra_preprocess:
                inputs = preprocess(inputs)

        targets = targets.cuda()

        # Inference
        outputs = model(inputs)

        # Convert logits to class indices
        outputs = outputs.argmax(dim=1)

        # Update metrics
        num_samples += targets.size(0)
        num_correct += (outputs == targets).sum()

    return (num_correct / num_samples * 100).item()


def get_model_flops(model, inputs):
    num_macs = profile_macs(model, inputs)
    return num_macs


def get_model_size(model: nn.Module, data_width=32):
    """
    calculate the model size in bits
    :param data_width: #bits per element
    """
    num_elements = 0
    for param in model.parameters():
        num_elements += param.numel()
    return num_elements * data_width


Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB


def test_k_means_quantize(
    test_tensor=torch.tensor(
        [
            [-0.3747, 0.0874, 0.3200, -0.4868, 0.4404],
            [-0.0402, 0.2322, -0.2024, -0.4986, 0.1814],
            [0.3102, -0.3942, -0.2030, 0.0883, -0.4741],
            [-0.1592, -0.0777, -0.3946, -0.2128, 0.2675],
            [0.0611, -0.1933, -0.4350, 0.2928, -0.1087],
        ]
    ),
    bitwidth=2,
):
    def plot_matrix(tensor, ax, title, cmap=ListedColormap(["white"])):
        ax.imshow(tensor.cpu().numpy(), vmin=-0.5, vmax=0.5, cmap=cmap)
        ax.set_title(title)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        for i in range(tensor.shape[1]):
            for j in range(tensor.shape[0]):
                text = ax.text(
                    j,
                    i,
                    f"{tensor[i, j].item():.2f}",
                    ha="center",
                    va="center",
                    color="k",
                )

    fig, axes = plt.subplots(1, 2, figsize=(8, 12))
    ax_left, ax_right = axes.ravel()

    print(test_tensor)
    plot_matrix(test_tensor, ax_left, "original tensor")

    num_unique_values_before_quantization = test_tensor.unique().numel()
    k_means_quantize(test_tensor, bitwidth=bitwidth)
    num_unique_values_after_quantization = test_tensor.unique().numel()
    print("* Test k_means_quantize()")
    print(f"    target bitwidth: {bitwidth} bits")
    print(
        f"        num unique values before k-means quantization: {num_unique_values_before_quantization}"
    )
    print(
        f"        num unique values after  k-means quantization: {num_unique_values_after_quantization}"
    )
    assert num_unique_values_after_quantization == min(
        (1 << bitwidth), num_unique_values_before_quantization
    )
    print("* Test passed.")

    plot_matrix(
        test_tensor, ax_right, f"{bitwidth}-bit k-means quantized tensor", cmap="tab20c"
    )
    fig.tight_layout()
    plt.show()


def test_linear_quantize(
    test_tensor=torch.tensor(
        [
            [0.0523, 0.6364, -0.0968, -0.0020, 0.1940],
            [0.7500, 0.5507, 0.6188, -0.1734, 0.4677],
            [-0.0669, 0.3836, 0.4297, 0.6267, -0.0695],
            [0.1536, -0.0038, 0.6075, 0.6817, 0.0601],
            [0.6446, -0.2500, 0.5376, -0.2226, 0.2333],
        ]
    ),
    quantized_test_tensor=torch.tensor(
        [
            [-1, 1, -1, -1, 0],
            [1, 1, 1, -2, 0],
            [-1, 0, 0, 1, -1],
            [-1, -1, 1, 1, -1],
            [1, -2, 1, -2, 0],
        ],
        dtype=torch.int8,
    ),
    real_min=-0.25,
    real_max=0.75,
    bitwidth=2,
    scale=1 / 3,
    zero_point=-1,
):
    def plot_matrix(tensor, ax, title, vmin=0, vmax=1, cmap=ListedColormap(["white"])):
        ax.imshow(tensor.cpu().numpy(), vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(title)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                datum = tensor[i, j].item()
                if isinstance(datum, float):
                    text = ax.text(
                        j, i, f"{datum:.2f}", ha="center", va="center", color="k"
                    )
                else:
                    text = ax.text(
                        j, i, f"{datum}", ha="center", va="center", color="k"
                    )

    quantized_min, quantized_max = get_quantized_range(bitwidth)
    fig, axes = plt.subplots(1, 3, figsize=(10, 32))
    plot_matrix(test_tensor, axes[0], "original tensor", vmin=real_min, vmax=real_max)
    _quantized_test_tensor = linear_quantize(
        test_tensor, bitwidth=bitwidth, scale=scale, zero_point=zero_point
    )
    _reconstructed_test_tensor = scale * (_quantized_test_tensor.float() - zero_point)
    print("* Test linear_quantize()")
    print(f"    target bitwidth: {bitwidth} bits")
    print(f"        scale: {scale}")
    print(f"        zero point: {zero_point}")
    assert _quantized_test_tensor.equal(quantized_test_tensor)
    print("* Test passed.")
    plot_matrix(
        _quantized_test_tensor,
        axes[1],
        f"2-bit linear quantized tensor",
        vmin=quantized_min,
        vmax=quantized_max,
        cmap="tab20c",
    )
    plot_matrix(
        _reconstructed_test_tensor,
        axes[2],
        f"reconstructed tensor",
        vmin=real_min,
        vmax=real_max,
        cmap="tab20c",
    )
    fig.tight_layout()
    plt.show()


def test_quantized_fc(
    input=torch.tensor(
        [
            [0.6118, 0.7288, 0.8511, 0.2849, 0.8427, 0.7435, 0.4014, 0.2794],
            [0.3676, 0.2426, 0.1612, 0.7684, 0.6038, 0.0400, 0.2240, 0.4237],
            [0.6565, 0.6878, 0.4670, 0.3470, 0.2281, 0.8074, 0.0178, 0.3999],
            [0.1863, 0.3567, 0.6104, 0.0497, 0.0577, 0.2990, 0.6687, 0.8626],
        ]
    ),
    weight=torch.tensor(
        [
            [
                1.2626e-01,
                -1.4752e-01,
                8.1910e-02,
                2.4982e-01,
                -1.0495e-01,
                -1.9227e-01,
                -1.8550e-01,
                -1.5700e-01,
            ],
            [
                2.7624e-01,
                -4.3835e-01,
                5.1010e-02,
                -1.2020e-01,
                -2.0344e-01,
                1.0202e-01,
                -2.0799e-01,
                2.4112e-01,
            ],
            [
                -3.8216e-01,
                -2.8047e-01,
                8.5238e-02,
                -4.2504e-01,
                -2.0952e-01,
                3.2018e-01,
                -3.3619e-01,
                2.0219e-01,
            ],
            [
                8.9233e-02,
                -1.0124e-01,
                1.1467e-01,
                2.0091e-01,
                1.1438e-01,
                -4.2427e-01,
                1.0178e-01,
                -3.0941e-04,
            ],
            [
                -1.8837e-02,
                -2.1256e-01,
                -4.5285e-01,
                2.0949e-01,
                -3.8684e-01,
                -1.7100e-01,
                -4.5331e-01,
                -2.0433e-01,
            ],
            [
                -2.0038e-01,
                -5.3757e-02,
                1.8997e-01,
                -3.6866e-01,
                5.5484e-02,
                1.5643e-01,
                -2.3538e-01,
                2.1103e-01,
            ],
            [
                -2.6875e-01,
                2.4984e-01,
                -2.3514e-01,
                2.5527e-01,
                2.0322e-01,
                3.7675e-01,
                6.1563e-02,
                1.7201e-01,
            ],
            [
                3.3541e-01,
                -3.3555e-01,
                -4.3349e-01,
                4.3043e-01,
                -2.0498e-01,
                -1.8366e-01,
                -9.1553e-02,
                -4.1168e-01,
            ],
        ]
    ),
    bias=torch.tensor(
        [0.1954, -0.2756, 0.3113, 0.1149, 0.4274, 0.2429, -0.1721, -0.2502]
    ),
    quantized_bias=torch.tensor([3, -2, 3, 1, 3, 2, -2, -2], dtype=torch.int32),
    shifted_quantized_bias=torch.tensor(
        [-1, 0, -3, -1, -3, 0, 2, -4], dtype=torch.int32
    ),
    calc_quantized_output=torch.tensor(
        [
            [0, -1, 0, -1, -1, 0, 1, -2],
            [0, 0, -1, 0, 0, 0, 0, -1],
            [0, 0, 0, -1, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 1, -1, -2],
        ],
        dtype=torch.int8,
    ),
    bitwidth=2,
    batch_size=4,
    in_channels=8,
    out_channels=8,
):
    def plot_matrix(tensor, ax, title, vmin=0, vmax=1, cmap=ListedColormap(["white"])):
        ax.imshow(tensor.cpu().numpy(), vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(title)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                datum = tensor[i, j].item()
                if isinstance(datum, float):
                    text = ax.text(
                        j, i, f"{datum:.2f}", ha="center", va="center", color="k"
                    )
                else:
                    text = ax.text(
                        j, i, f"{datum}", ha="center", va="center", color="k"
                    )

    output = torch.nn.functional.linear(input, weight, bias)

    quantized_weight, weight_scale, weight_zero_point = (
        linear_quantize_weight_per_channel(weight, bitwidth)
    )
    quantized_input, input_scale, input_zero_point = linear_quantize_feature(
        input, bitwidth
    )
    _quantized_bias, bias_scale, bias_zero_point = (
        linear_quantize_bias_per_output_channel(bias, weight_scale, input_scale)
    )
    assert _quantized_bias.equal(_quantized_bias)
    _shifted_quantized_bias = shift_quantized_linear_bias(
        quantized_bias, quantized_weight, input_zero_point
    )
    assert _shifted_quantized_bias.equal(shifted_quantized_bias)
    quantized_output, output_scale, output_zero_point = linear_quantize_feature(
        output, bitwidth
    )

    _calc_quantized_output = quantized_linear(
        quantized_input,
        quantized_weight,
        shifted_quantized_bias,
        bitwidth,
        bitwidth,
        input_zero_point,
        output_zero_point,
        input_scale,
        weight_scale,
        output_scale,
    )
    assert _calc_quantized_output.equal(calc_quantized_output)

    reconstructed_weight = weight_scale * (quantized_weight.float() - weight_zero_point)
    reconstructed_input = input_scale * (quantized_input.float() - input_zero_point)
    reconstructed_bias = bias_scale * (quantized_bias.float() - bias_zero_point)
    reconstructed_calc_output = output_scale * (
        calc_quantized_output.float() - output_zero_point
    )

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    quantized_min, quantized_max = get_quantized_range(bitwidth)
    plot_matrix(weight, axes[0, 0], "original weight", vmin=-0.5, vmax=0.5)
    plot_matrix(input.t(), axes[1, 0], "original input", vmin=0, vmax=1)
    plot_matrix(output.t(), axes[2, 0], "original output", vmin=-1.5, vmax=1.5)
    plot_matrix(
        quantized_weight,
        axes[0, 1],
        f"{bitwidth}-bit linear quantized weight",
        vmin=quantized_min,
        vmax=quantized_max,
        cmap="tab20c",
    )
    plot_matrix(
        quantized_input.t(),
        axes[1, 1],
        f"{bitwidth}-bit linear quantized input",
        vmin=quantized_min,
        vmax=quantized_max,
        cmap="tab20c",
    )
    plot_matrix(
        calc_quantized_output.t(),
        axes[2, 1],
        f"quantized output from quantized_linear()",
        vmin=quantized_min,
        vmax=quantized_max,
        cmap="tab20c",
    )
    plot_matrix(
        reconstructed_weight,
        axes[0, 2],
        f"reconstructed weight",
        vmin=-0.5,
        vmax=0.5,
        cmap="tab20c",
    )
    plot_matrix(
        reconstructed_input.t(),
        axes[1, 2],
        f"reconstructed input",
        vmin=0,
        vmax=1,
        cmap="tab20c",
    )
    plot_matrix(
        reconstructed_calc_output.t(),
        axes[2, 2],
        f"reconstructed output",
        vmin=-1.5,
        vmax=1.5,
        cmap="tab20c",
    )

    print("* Test quantized_fc()")
    print(f"    target bitwidth: {bitwidth} bits")
    print(f"      batch size: {batch_size}")
    print(f"      input channels: {in_channels}")
    print(f"      output channels: {out_channels}")
    print("* Test passed.")
    fig.tight_layout()
    plt.show()


checkpoint_path = "/home/ghr/ghr_code/data/vgg.cifar.pretrained.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
model = VGG().cuda()
print(f"=> loading checkpoint '{checkpoint_path}'")
model.load_state_dict(checkpoint["state_dict"])
recover_model = lambda: model.load_state_dict(checkpoint["state_dict"])

image_size = 32
transforms = {
    "train": Compose(
        [
            RandomCrop(image_size, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
        ]
    ),
    "test": ToTensor(),
}
dataset = {}
for split in ["train", "test"]:
    dataset[split] = CIFAR10(
        root="data/cifar10",
        train=(split == "train"),
        download=True,
        transform=transforms[split],
    )
dataloader = {}
for split in ["train", "test"]:
    dataloader[split] = DataLoader(
        dataset[split],
        batch_size=512,
        shuffle=(split == "train"),
        num_workers=0,
        pin_memory=True,
    )

# 首先评估 FP32 模型的准确率和模型大小
fp32_model_accuracy = evaluate(model, dataloader=["test"])
fp32_model_size = get_model_size(model)
print(f"fp32 model has accuracy={fp32_model_accuracy:.2f}%")
print(f"fp32 model has size={fp32_model_size / MiB:.2f} MiB")

"""
https://arxiv.org/pdf/1510.00149

网络量化通过减少表示深度网络所需的每权重比特数来压缩网络。量化后的网络在硬件支持下可以获得更快的推理速度

一个 n 比特的 k-means 量化会将突出分为 2^n 个聚类，同一聚类中的突触将共享相同的权重值

因此，k-means量化会创建一个码本，包含：
- centroids:2^n 个 fp32 聚类中心
- labels:一个 n 比特的整数张量，其元素数量与原始 fp32 权重张量相同。每个整数表示该元素所属的聚类

在推理过程中，根据码本生成一个 fp32 张量用于推理：

quantized_weight = codebook.centroids[codebook.labels].view_as(weight)

"""
from collections import namedtuple

Codebook = namedtuple("Codebook", ["centroids", "labels"])

# 问题 1: 实现 k-means 量化
from fast_pytorch_kmeans import KMeans


def k_means_quantize(fp32_tensor: torch.Tensor, bitwidth=4, codebook=None):
    """
    quantize tensor using k-means clustering
    :param fp32_tensor:
    :param bitwidth: [int] quantization bit width, default=4
    :param codebook: [Codebook] (the cluster centroids, the cluster label tensor)
    :return:
        [Codebook = (centroids, labels)]
            centroids: [torch.(cuda.)FloatTensor] the cluster centroids
            labels: [torch.(cuda.)LongTensor] cluster label tensor
    """
    if codebook is None:
        ############### YOUR CODE STARTS HERE ###############
        # get number of clusters based on the quantization precision
        # hint: one line of code
        # bitwidth 位数能表示多少个不同的值？
        # 比如 bitwidth = 4 -> 2^4 = 16 个不同的聚类
        n_clusters = 2**bitwidth
        ############### YOUR CODE ENDS HERE #################
        # use k-means to get the quantization centroids
        kmeans = KMeans(n_clusters=n_clusters, mode="euclidean", verbose=0)
        labels = kmeans.fit_predict(fp32_tensor.view(-1, 1)).to(torch.long)
        centroids = kmeans.centroids.to(torch.float).view(-1)
        codebook = Codebook(centroids, labels)
    ############### YOUR CODE STARTS HERE ###############
    # decode the codebook into k-means quantized tensor for inference
    # hint: one line of code
    quantized_tensor = codebook.centroids[codebook.labels]
    ############### YOUR CODE ENDS HERE #################
    fp32_tensor.set_(quantized_tensor.view_as(fp32_tensor))
    return codebook


# 通过对一个虚拟张量应用 k-means 量化来测试 k-means 量化函数
test_k_means_quantize()

# 问题 2
"""
上一个代码单元执行了 2 比特 k-means 量化，并绘制了量化前后的张量。
每个聚类以不同的颜色表示。量化后的张量中显示了 4 种不同的颜色

问题 2.1 如果执行 4 比特 k-means 量化，量化后的张量中会显示多少种不同的颜色？为什么？
答：如果执行 4 比特 k-means 量化，量化后的张量中会显示 16 种不同的颜色。因为 4 比特可以表示 2^4 = 16 个不同的聚类，每个聚类对应一个唯一的颜色。

问题 2.2 如果执行 1 比特 k-means 量化，量化后的张量中会显示多少种不同的颜色？为什么？
答：如果执行 1 比特 k-means 量化，量化后的张量中会显示 2 种不同的颜色。因为 1 比特只能表示 2^1 = 2 个不同的聚类，每个聚类对应一个唯一的颜色。

问题 2.3 如果执行 8 比特 k-means 量化，量化后的张量中会显示多少种不同的颜色？为什么？
答：如果执行 8 比特 k-means 量化，量化后的张量中会显示 256 种不同的颜色。因为 8 比特可以表示 2^8 = 256 个不同的聚类，每个聚类对应一个唯一的颜色。

问题 2.4 如果执行 16 比特 k-means 量化，量化后的张量中会显示多少种不同的颜色？为什么？
答：如果执行 16 比特 k-means 量化，量化后的张量中会显示 65536 种不同的颜色。因为 16 比特可以表示 2^16 = 65536 个不同的聚类，每个聚类对应一个唯一的颜色。

问题 2.5 如果执行 32 比特 k-means 量化，量化后的张量中会显示多少种不同的颜色？为什么？
答：如果执行 32 比特 k-means 量化，量化后的张量中会显示 4294967296 种不同的颜色。因为 32 比特可以表示 2^32 = 4294967296 个不同的聚类，每个聚类对应一个唯一的颜色。

问题 2.6 如果执行 n 比特 k-means 量化，量化后的张量中会显示多少种不同的颜色？为什么？
答：如果执行 n 比特 k-means 量化，量化后的张量中会显示 2^n 种不同的颜色。因为 n 比特可以表示 2^n 个不同的聚类，每个聚类对应一个唯一的颜色。
"""

# 对整个模型进行 K-Means 量化
"""
将 k-means 量化函数封装到一个类中，用于量化整个模型。在 KMeansQuantizer 类中,我们需要记录码本(即 centroids和labels),
以便在模型权重变化时能够应用或更新码本
"""
from torch.nn import parameter


class KMeansQuantizer:
    """
    K-Means 量化器: 用聚类算法将模型权重压缩到 n 比特。

    核心思想:
      将权重矩阵中的所有浮点值聚成 2^bitwidth 个类,
      每个元素只存储它所属的类别编号 (n-bit 整数),
      实际推理时用类别对应的聚类中心 (fp32 centroids) 来重建权重。

    例如 bitwidth=4 → 16 个聚类中心 → 每个元素只需 4 bits 存储, 压缩比 ≈ 8x (vs fp32)
    """

    def __init__(self, model: nn.Module, bitwidth=4):
        # 初始化时对整个模型做一次 k-means 量化, 生成初始码本
        # 码本 = {参数名: Codebook(centroids, labels)}
        #   centroids: 2^bitwidth 个浮点聚类中心
        #   labels:    每个权重元素属于哪个聚类的整数索引
        self.codebook = KMeansQuantizer.quantize(model, bitwidth)

    @torch.no_grad()
    def apply(self, model, update_centroids):
        """
        将码本解码回权重, 原地替换模型参数 (param.set_()).

        :param model: 待量化的模型
        :param update_centroids: 是否在应用前重新聚类。
               训练/微调后权重分布变了 → 需要重新跑 k-means 更新聚类中心。
               推理时保持 False → 直接用已有码本解码。
        """
        for name, param in model.named_parameters():
            if name in self.codebook:  # 只处理在码本中的参数 (bias / LayerNorm 不在)
                if update_centroids:
                    # 权重的值经过训练已经变了, 重新聚类更准确
                    # update_codebook 会对新权重重新算聚类中心 (取每个聚类内权重的均值)
                    update_codebook(param, codebook=self.codebook[name])
                # 用(更新后的)码本重建量化权重:
                #   centroids[labels] → 每个元素用所属聚类的中心值替换
                #   k_means_quantize 内部调用 param.set_() 原地修改权重存储
                self.codebook[name] = k_means_quantize(
                    param, codebook=self.codebook[name]
                )

    @staticmethod
    @torch.no_grad()
    def quantize(model: nn.Module, bitwidth=4):
        """
        对模型的所有 2D+ 权重矩阵做 k-means 量化。

        :param bitwidth: int → 全局统一位宽 (如 4-bit)
                         dict → 逐层指定 {"fc1.weight": 8, "fc2.weight": 4}
        :return: {参数名: Codebook(centroids, labels)} 码本字典
        """
        codebook = dict()

        if isinstance(bitwidth, dict):
            # ── 逐层精度 ──
            # 不同层对精度的敏感度不同, 允许差异化配置
            # 例如: 第一层 conv 和最后一层 fc 用 8-bit, 中间层用 4-bit
            for name, param in model.named_parameters():
                if name in bitwidth:
                    codebook[name] = k_means_quantize(param, bitwidth=bitwidth[name])

        else:
            # ── 统一精度 ──
            # param.dim() > 1: 只量化权重矩阵 (Conv/Linear 的 weight)
            # 跳过 dim ≤ 1 的参数 (bias / LayerNorm weight):
            #   这些参数元素少, 量化的存储收益极低, 且对精度影响大
            for name, param in model.named_parameters():
                if param.dim() > 1:
                    codebook[name] = k_means_quantize(param, bitwidth=bitwidth)

        return codebook


# 使用 K-Means量化将模型量化为 8 比特、4比特和 2 比特。注意，在计算模型大小是，我们忽略码本的存储开销
print("Note that the storage for codebooks is ignored when calculating the model size.")
quantizers = dict()
for bitwidth in [8, 4, 2]:
    recover_model()
    print(f"k-means quantizing model into {bitwidth} bits")
    quantizer = KMeansQuantizer(model, bitwidth)
    quantized_model_size = get_model_size(model, bitwidth)
    print(
        f"    {bitwidth}-bit k-means quantized model has size={quantized_model_size / MiB:.2f} MiB"
    )
    quantized_model_accuracy = evaluate(model, dataloader["test"])
    print(
        f"    {bitwidth}-bit k-means quantized model has accuracy={quantized_model_accuracy:.2f}%"
    )
    quantizers[bitwidth] = quantizer

# 训练 K-Means 量化模型
"""
从上一个单元的结果可以看出，当将模型量化为更低比特时，准确率会显著下降。
因此，我们需要进行量化感知训练来恢复准确率。

在 k-means量化感知训练过程中，聚类中心也会被更新。
这一方法在：https://arxiv.org/pdf/1510.00149
"""


"""
为什么这样做:
  训练/微调过程中, optimizer 会更新模型权重(fp32_tensor)。
  权重变了 → 原来聚类所用的 centroids 就不再是最优代表值了 → 需要基于新权重重新计算。

为什么取均值:
  k-means 的优化目标是「最小化聚类内平方误差」:
    对每个聚类 k: 找一个值 c_k, 使得 Σ(该类元素 - c_k)² 最小
  这个凸优化问题的闭式解就是: c_k = 该类所有权重的算术平均值。
  (对误差函数求导 = 0 → c_k = mean(该类元素))

这样做的好处:
  对比「重新跑完整 k-means」:
  ┌──────────────────┬────────────────────┬─────────────────────────┐
  │                  │ update_codebook     │ 重新跑完整 k-means        │
  ├──────────────────┼────────────────────┼─────────────────────────┤
  │ 时间复杂度        │ O(N) 一次遍历       │ O(N × K × iter) 多轮迭代 │
  │ labels 是否变化   │ 不变(保持原聚类边界) │ 会变(可能打乱聚类结构)    │
  │ 前提假设          │ 权重变化不大时有效   │ 无需假设                 │
  │ 适合场景          │ QAT 每个 training    │ 初次量化(从零开始)        │
  │                  │ step(微调时缓慢变化)  │                         │
  └──────────────────┴────────────────────┴─────────────────────────┘
  QAT 训练中每个 step 都会调用此函数(见 callbacks 参数),
  如果每个 step 都重跑完整 k-means, 训练速度会慢几十倍。
  O(N) 的取均值让 QAT 变得实际可行。
"""


# 问题 3
# 上述更新聚类中心的方程本质上是对同一聚类中的权重取均值作为更新后的聚类中心值
def update_codebook(fp32_tensor: torch.Tensor, codebook: Codebook):
    """
    update the centroids in the codebook using updated fp32_tensor
    :param fp32_tensor: [torch.(cuda.)Tensor] 经过 optimizer 更新后的权重
    :param codebook: [Codebook] (the cluster centroids, the cluster label tensor)
    :原理:
       对每个聚类 k, 找到所有 label == k 的元素(即属于第 k 类的所有权重值),
       取它们的算术平均作为新的聚类中心。
       数学依据: 均值是「最小化聚类内平方误差」的最优解(凸优化闭式解)。
    """
    n_clusters = codebook.centroids.numel()
    fp32_tensor = fp32_tensor.view(-1)  # 展平为一维, 便于按 labels 索引
    for k in range(n_clusters):
        ############### YOUR CODE STARTS HERE ###############
        # hint: one line of code
        # 对第 k 个聚类: 找出该类所有元素, 取均值作为新的聚类中心
        # codebook.labels == k → bool 掩码, 标记哪些元素属于该类
        # fp32_tensor[mask].mean() → 该类元素均值 = 最优聚类中心(凸优化闭式解)
        codebook.centroids[k] = fp32_tensor[codebook.labels == k].mean()
    ############### YOUR CODE ENDS HERE #################


# ======================================================================
# QAT (量化感知训练) 微调循环
# ======================================================================
# 目标: 通过微调恢复量化后的精度损失。
#
# 核心思路:
#   量化(压缩权重) → 精度下降 → 用训练数据微调 → 模型学会在量化约束下工作
#
# 关键设计: 每个 training step 之后都重新量化权重
#   1. normal forward  → 用 fp32 权重计算(精确)
#   2. backward + step → optimizer 更新了 fp32 权重
#   3. callback         → quantizer.apply(model, update_centroids=True)
#        a. 用新 fp32 权重重新计算聚类中心(取均值)
#        b. 用新 centroids 重建量化权重 → param.set_() 原地替换
#     这样下一个 step 的 forward 用的是「更新后的量化权重」,
#     模型在训练中学会补偿量化误差。
#
# 如果准确率下降小于 0.5%，我们将停止微调
accuracy_drop_threshold = 0.5
quantizers_before_finetune = copy.deepcopy(quantizers)
quantizers_after_finetune = quantizers

for bitwidth in [8, 4, 2]:
    # Step 1: 恢复 fp32 模型权重 (从预训练 checkpoint)
    recover_model()

    # Step 2: 获取该 bitwidth 对应的量化器 (包含之前算好的码本)
    quantizer = quantizers[bitwidth]
    print(f"k-means quantizing model into {bitwidth} bits")

    # Step 3: 应用量化 — 用码本重建权重, 但不更新聚类中心
    #         (因为权重刚从 checkpoint 恢复, 和初次 quantize 时一样)
    #         param.set_() → 现在模型权重已被替换为量化后的值
    quantizer.apply(model, update_centroids=False)

    # Step 4: 查看量化后模型大小和准确率
    quantized_model_size = get_model_size(model, bitwidth)
    print(
        f"    {bitwidth}-bit k-means quantized model has size={quantized_model_size / MiB:.2f} MiB"
    )
    quantized_model_accuracy = evaluate(model, dataloader["test"])
    print(
        f"    {bitwidth}-bit k-means quantized model has accuracy={quantized_model_accuracy:.2f}% before quantization-aware training "
    )

    # Step 5: 计算精度下降量 — 决定是否需要 QAT 微调
    accuracy_drop = fp32_model_accuracy - quantized_model_accuracy

    if accuracy_drop > accuracy_drop_threshold:
        # ── 精度下降超过阈值 → 需要 QAT 微调 ──
        print(
            f"        Quantization-aware training due to accuracy drop={accuracy_drop:.2f}% is larger than threshold={accuracy_drop_threshold:.2f}%"
        )
        num_finetune_epochs = 5
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            num_finetune_epochs,  # 余弦退火: lr 从 0.01 逐渐降到 ~0
        )
        criterion = nn.CrossEntropyLoss()
        best_accuracy = 0
        epoch = num_finetune_epochs

        # 持续微调直到精度恢复(下降 < 阈值) 或 epoch 耗尽
        while accuracy_drop > accuracy_drop_threshold and epoch > 0:
            # ── 一个 epoch 的训练 ──
            # callbacks 中的 lambda 在每个 training step 后执行:
            #   quantizer.apply(model, update_centroids=True)
            #   → 1. 用当前(被 optimizer 更新过的) fp32 权重重新聚类
            #   → 2. 用新 centroids 重建量化权重, 原地替换 model 参数
            #   这样下一轮 forward 就基于更新后的量化权重计算
            train(
                model,
                dataloader["train"],
                criterion,
                optimizer,
                scheduler,
                callbacks=[lambda: quantizer.apply(model, update_centroids=True)],
            )

            # ── 验证: 用测试集评估当前量化模型的准确率 ──
            model_accuracy = evaluate(model, dataloader["test"])
            is_best = model_accuracy > best_accuracy
            best_accuracy = max(model_accuracy, best_accuracy)  # 追踪最佳精度
            print(
                f"        Epoch {num_finetune_epochs - epoch} Accuracy {model_accuracy:.2f}% / Best Accuracy: {best_accuracy:.2f}%"
            )

            # 用最佳精度重新计算下降量(避免某个 epoch 波动导致过早停止)
            accuracy_drop = fp32_model_accuracy - best_accuracy
            epoch -= 1

    else:
        # ── 精度损失在可接受范围内 → 跳过微调 ──
        print(
            f"        No need for quantization-aware training since accuracy drop={accuracy_drop:.2f}% is smaller than threshold={accuracy_drop_threshold:.2f}%"
        )


# 线性量化
"""
在本节中，我们将实现并执行线性量化。

线性量化在经过范围截断和缩放后，直接将浮点值舍入到最接近的量化整数

线性量化可以表示为：

r = S(q - Z)

其中 r 是浮点实数，q是n比特整数，Z是n比特整数，S是浮点实数。

Z是量化零点，S是量化缩放因子。常量 Z 和 S 都是量化参数
"""

"""
n 比特整数

一个 n 比特有符号整数通常以二进制补码表示

一个 n 比特有符号整数的范围为 [-2^(n-1), 2^(n-1)-1]，其中 n 是整数的位宽。

例如一个 8 比特有符号整数的范围为 [-128, 127]，一个 4 比特有符号整数的范围为 [-8, 7]。
"""


def get_quantized_range(bitwidth):
    quantized_max = (1 << (bitwidth - 1)) - 1
    quantized_min = -(1 << (bitwidth - 1))
    return quantized_min, quantized_max


"""
问题 4：

请完成以下线性量化函数。

提示：

- 由 r = S(q - Z) 可得 q = r/S + Z
- r和 S都是浮点数，因此我们不能直接将整数Z加到 r/S 上。所以 q = int(round(r/S + Z))，其中 round() 是四舍五入函数，int() 是将浮点数转换为整数的函数
- 要将 torch.FloatTensor 转换为 torch.IntTensor,我们可以使用 torch.round()、torch.Tensor.round()、torch.Tensor.round_()
  先将所有值转换为浮点数，然后使用 torch.Tensor.to(torch.int8) 将数据类型从 torch.float 转换为 torch.int8
"""


def linear_quantize(
    fp_tensor, bitwidth, scale, zero_point, dtype=torch.int8
) -> torch.Tensor:
    """
    linear quantization for single fp_tensor
      from
        fp_tensor = (quantized_tensor - zero_point) * scale
      we have,
        quantized_tensor = int(round(fp_tensor / scale)) + zero_point
    :param tensor: [torch.(cuda.)FloatTensor] floating tensor to be quantized
    :param bitwidth: [int] quantization bit width
    :param scale: [torch.(cuda.)FloatTensor] scaling factor
    :param zero_point: [torch.(cuda.)IntTensor] the desired centroid of tensor values
    :return:
        [torch.(cuda.)FloatTensor] quantized tensor whose values are integers
    """
    assert fp_tensor.dtype == torch.float
    assert isinstance(scale, float) or (
        scale.dtype == torch.float and scale.dim() == fp_tensor.dim()
    )
    assert isinstance(zero_point, int) or (
        zero_point.dtype == dtype and zero_point.dim() == fp_tensor.dim()
    )

    ############### YOUR CODE STARTS HERE ###############
    # Step 1: scale the fp_tensor
    scaled_tensor = fp_tensor / scale
    # Step 2: round the floating value to integer value
    rounded_tensor = torch.round(scaled_tensor)
    ############### YOUR CODE ENDS HERE #################

    rounded_tensor = rounded_tensor.to(dtype)

    ############### YOUR CODE STARTS HERE ###############
    # Step 3: shift the rounded_tensor to make zero_point 0
    shifted_tensor = rounded_tensor + zero_point
    ############### YOUR CODE ENDS HERE #################

    # Step 4: clamp the shifted_tensor to lie in bitwidth-bit range
    quantized_min, quantized_max = get_quantized_range(bitwidth)
    quantized_tensor = shifted_tensor.clamp_(quantized_min, quantized_max)
    return quantized_tensor


# 验证
test_linear_quantize()

"""
问题 5

现在我们需要确定线性量化的缩放因子 S和零点 Z

回顾一下，线性量化可以表示为：
r = S(q - Z)

缩放因子
线性量化将浮点范围[fp_min,fp_max]映射到整数范围[quantized_min, quantized_max]，其中 fp_min 和 fp_max 是浮点张量的最小值和最大值，quantized_min 和 quantized_max 是 n 比特有符号整数的最小值和最大值。
也就是说：
r_max = S(q_max - Z)
r_min = S(q_min - Z)

将这两个方程相减，我们得到：

问题 5.1 
请在下一个文本单元中选择正确的答案并删除错误的答案。

S = (r_max - r_min) / (q_max - q_min)

确定浮点张量 fp_tensor 的 r_min 和 r_max有不同方法
- 最常用的方法是直接使用 fp_tensor 的最大值和最小值
- 另一种广泛使用的方法是最小化 KL 散度来确定 fp_max

零点

一旦确定了缩放因子 S，我们可以直接利用 r_min 和 q_min 之间的关系来计算零点 Z

问题 5.2
请在下一个文本单元中选择正确的答案并删除错误的答案。
Z = int(round(q_min - r_min / S))


"""


# 问题 5.3
# 完成以下函数，该函数从浮点张量r计算缩放因子 S 和零点Z
def get_quantization_scale_and_zero_point(fp_tensor, bitwidth):
    """
    get quantization scale for single tensor
    :param fp_tensor: [torch.(cuda.)Tensor] floating tensor to be quantized
    :param bitwidth: [int] quantization bit width
    :return:
        [float] scale
        [int] zero_point
    """
    quantized_min, quantized_max = get_quantized_range(bitwidth)
    fp_max = fp_tensor.max().item()
    fp_min = fp_tensor.min().item()

    ############### YOUR CODE STARTS HERE ###############
    # hint: one line of code for calculating scale
    scale = (fp_max - fp_min) / (quantized_max - quantized_min)
    # hint: one line of code for calculating zero_point
    zero_point = round(quantized_min - fp_min / scale)
    ############### YOUR CODE ENDS HERE #################

    # clip the zero_point to fall in [quantized_min, quantized_max]
    if zero_point < quantized_min:
        zero_point = quantized_min
    elif zero_point > quantized_max:
        zero_point = quantized_max
    else:  # convert from float to int using round()
        zero_point = round(zero_point)
    return scale, int(zero_point)


# 将 linear_quantize() 和 get_quantization_scale_and_zero_point() 函数结合起来，完成线性量化的整个过程
def linear_quantize_feature(fp_tensor, bitwidth):
    """
    linear quantization for feature tensor
    :param fp_tensor: [torch.(cuda.)Tensor] floating feature to be quantized
    :param bitwidth: [int] quantization bit width
    :return:
        [torch.(cuda.)Tensor] quantized tensor
        [float] scale tensor
        [int] zero point
    """
    scale, zero_point = get_quantization_scale_and_zero_point(fp_tensor, bitwidth)
    quantized_tensor = linear_quantize(fp_tensor, bitwidth, scale, zero_point)
    return quantized_tensor, scale, zero_point


# 特殊情况：权重张量的线性量化
# 先看一下权重值的分布
def plot_weight_distribution(model, bitwidth=32):
    # bins = (1 << bitwidth) if bitwidth <= 8 else 256
    if bitwidth <= 8:
        qmin, qmax = get_quantized_range(bitwidth)
        bins = np.arange(qmin, qmax + 2)
        align = "left"
    else:
        bins = 256
        align = "mid"
    fig, axes = plt.subplots(3, 3, figsize=(10, 6))
    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_parameters():
        if param.dim() > 1:
            ax = axes[plot_index]
            ax.hist(
                param.detach().view(-1).cpu(),
                bins=bins,
                density=True,
                align=align,
                color="blue",
                alpha=0.5,
                edgecolor="black" if bitwidth <= 4 else None,
            )
            if bitwidth <= 4:
                quantized_min, quantized_max = get_quantized_range(bitwidth)
                ax.set_xticks(np.arange(start=quantized_min, stop=quantized_max + 1))
            ax.set_xlabel(name)
            ax.set_ylabel("density")
            plot_index += 1
    fig.suptitle(f"Histogram of Weights (bitwidth={bitwidth} bits)")
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.show()


recover_model()
plot_weight_distribution(model)

# 从上面的直方图可以看出，权重值的分布关于 0 几乎是对称的(本例中分类器除外)。
# 因此，在对权重进行量化时，我们通常令零点 Z = 0
r"""
由 r = S(q - Z) 可得 r_max = S \cdot q_max

进而 S = r_max / q_max

我们直接使用权重的最大绝对值作为 r_max
"""


def get_quantization_scale_for_weight(weight, bitwidth):
    """
    get quantization scale for single tensor of weight
    :param weight: [torch.(cuda.)Tensor] floating weight to be quantized
    :param bitwidth: [integer] quantization bit width
    :return:
        [floating scalar] scale
    """
    # we just assume values in weight are symmetric
    # we also always make zero_point 0 for weight
    fp_max = max(weight.abs().max().item(), 5e-7)
    _, quantized_max = get_quantized_range(bitwidth)
    return fp_max / quantized_max


"""
逐通道线性量化

============================================================
什么是"不同通道"？
============================================================

以 Conv2d 权重 [out_channels, in_channels, kH, kW] 为例:

  out_ch 0: kernel[0, :, :, :] 对输入卷积 → 输出 feature map channel 0
  out_ch 1: kernel[1, :, :, :] 对输入卷积 → 输出 feature map channel 1
  ...
  out_ch C: kernel[C, :, :, :] 对输入卷积 → 输出 feature map channel C

每个 out_ch 是一个独立的卷积核, 检测不同的特征(边缘/纹理/颜色等),
因此不同通道的权重值分布差异可能很大:
  channel 0 的值在 [-0.1, 0.1]
  channel 3 的值在 [-0.5, 0.5]  ← 范围大 5 倍!

如果所有通道共用一个 scale:
  小范围通道会被"挤压"到接近 0 的少数几个整数值 → 精度损失巨大。

逐通道量化的做法:
  每个 out_ch 独立计算自己的 scale (取它自己的 abs_max / q_max),
  这样每个通道都能充分利用整数范围, 精度更好。

对线性层 Linear [out_features, in_features] 同理:
  out_ch = out_features, 每个输出神经元有自己的权重向量。

============================================================

回想一下，对于 2D 卷积，权重张量是一个形状为(num_output_channels,num_input_channels,kernel_height,kernel_width)的四维张量。

大量实验表明，对不同输出通道使用不同的缩放因子S 和 零点 Z 效果更好。因此，
我们需要独立地为每个输出通道计算缩放因子 S 和 零点 Z，并对每个输出通道的权重进行量化。
"""


def linear_quantize_weight_per_channel(tensor, bitwidth):
    """
    linear quantization for weight tensor
        using different scales and zero_points for different output channels
    :param tensor: [torch.(cuda.)Tensor] floating weight to be quantized
    :param bitwidth: [int] quantization bit width
    :return:
        [torch.(cuda.)Tensor] quantized tensor
        [torch.(cuda.)Tensor] scale tensor
        [int] zero point (which is always 0)
    """
    dim_output_channels = 0
    num_output_channels = tensor.shape[dim_output_channels]
    scale = torch.zeros(num_output_channels, device=tensor.device)
    for oc in range(num_output_channels):
        _subtensor = tensor.select(dim_output_channels, oc)
        _scale = get_quantization_scale_for_weight(_subtensor, bitwidth)
        scale[oc] = _scale
    scale_shape = [1] * tensor.dim()
    scale_shape[dim_output_channels] = -1
    scale = scale.view(scale_shape)
    quantized_tensor = linear_quantize(tensor, bitwidth, scale, zero_point=0)
    return quantized_tensor, scale, 0


"""
权重线性量化速览

看一下 以不同比特位宽对权重进行线性量化时的权重分布和模型大小
"""


@torch.no_grad()
def peek_linear_quantization():
    for bitwidth in [4, 2]:
        for name, param in model.named_parameters():
            if param.dim() > 1:
                quantized_param, scale, zero_point = linear_quantize_weight_per_channel(
                    param, bitwidth
                )
                param.copy_(quantized_param)
        plot_weight_distribution(model, bitwidth)
        recover_model()


peek_linear_quantization()

# 量化推理
# 量化之后，卷积层和全连接层的推理也会发生变化
"""
问题 6
请完成以下用于对偏置进行线性量化的函数。注意，偏置的量化是基于输入和权重的缩放因子 S 和 零点 Z 进行的。

提示：
从上面的推导中，我们知道

Z_bias = 0
S_bias = S_input * S_weight
"""


def linear_quantize_bias_per_output_channel(bias, weight_scale, input_scale):
    """
    linear quantization for single bias tensor
        quantized_bias = fp_bias / bias_scale
    :param bias: [torch.FloatTensor] bias weight to be quantized
    :param weight_scale: [float or torch.FloatTensor] weight scale tensor
    :param input_scale: [float] input scale
    :return:
        [torch.IntTensor] quantized bias tensor
    """
    assert bias.dim() == 1
    assert bias.dtype == torch.float
    assert isinstance(input_scale, float)
    if isinstance(weight_scale, torch.Tensor):
        assert weight_scale.dtype == torch.float
        weight_scale = weight_scale.view(-1)
        assert bias.numel() == weight_scale.numel()

    ############### YOUR CODE STARTS HERE ###############
    # hint: one line of code
    bias_scale = weight_scale * input_scale
    ############### YOUR CODE ENDS HERE #################

    quantized_bias = linear_quantize(
        bias, 32, bias_scale, zero_point=0, dtype=torch.int32
    )
    return quantized_bias, bias_scale, 0


# 量化全连接层
# 对于量化全连接层，我们首先预计算 Q_bias
# 回顾 Q_bias = q_bias - Linear[Z_input,q_weight]
def shift_quantized_linear_bias(quantized_bias, quantized_weight, input_zero_point):
    """
    shift quantized bias to incorporate input_zero_point for nn.Linear
        shifted_quantized_bias = quantized_bias - Linear(input_zero_point, quantized_weight)
    :param quantized_bias: [torch.IntTensor] quantized bias (torch.int32)
    :param quantized_weight: [torch.CharTensor] quantized weight (torch.int8)
    :param input_zero_point: [int] input zero point
    :return:
        [torch.IntTensor] shifted quantized bias tensor
    """
    assert quantized_bias.dtype == torch.int32
    assert isinstance(input_zero_point, int)
    return quantized_bias - quantized_weight.sum(1).to(torch.int32) * input_zero_point


# 问题 7
# 请完成以下量化全连接层推理函数

"""
提示：

q_output = (Linear[q_input,q_weight] + q_bias) * S_input * S_weight / S_output + Z_output
"""


def quantized_linear(
    input,
    weight,
    bias,
    feature_bitwidth,
    weight_bitwidth,
    input_zero_point,
    output_zero_point,
    input_scale,
    weight_scale,
    output_scale,
):
    """
    quantized fully-connected layer
    :param input: [torch.CharTensor] quantized input (torch.int8)
    :param weight: [torch.CharTensor] quantized weight (torch.int8)
    :param bias: [torch.IntTensor] shifted quantized bias or None (torch.int32)
    :param feature_bitwidth: [int] quantization bit width of input and output
    :param weight_bitwidth: [int] quantization bit width of weight
    :param input_zero_point: [int] input zero point
    :param output_zero_point: [int] output zero point
    :param input_scale: [float] input feature scale
    :param weight_scale: [torch.FloatTensor] weight per-channel scale
    :param output_scale: [float] output feature scale
    :return:
        [torch.CharIntTensor] quantized output feature (torch.int8)
    """
    assert input.dtype == torch.int8
    assert weight.dtype == input.dtype
    assert bias is None or bias.dtype == torch.int32
    assert isinstance(input_zero_point, int)
    assert isinstance(output_zero_point, int)
    assert isinstance(input_scale, float)
    assert isinstance(output_scale, float)
    assert weight_scale.dtype == torch.float

    # Step 1: integer-based fully-connected (8-bit multiplication with 32-bit accumulation)
    if "cpu" in input.device.type:
        # use 32-b MAC for simplicity
        output = torch.nn.functional.linear(
            input.to(torch.int32), weight.to(torch.int32), bias
        )
    else:
        # current version pytorch does not yet support integer-based linear() on GPUs
        output = torch.nn.functional.linear(input.float(), weight.float(), bias.float())

    ############### YOUR CODE STARTS HERE ###############
    # Step 2: scale the output
    #         hint: 1. scales are floating numbers, we need to convert output to float as well
    #               2. the shape of weight scale is [oc, 1, 1, 1] while the shape of output is [batch_size, oc]
    # Step 2: 缩放 — 将整数域结果映射回浮点域的 output scale
    # output.float() → 转为浮点才能做乘除
    # * input_scale * weight_scale → 反量化到 fp32 域
    # / output_scale → 缩放到输出的 scale
    output = output.float() * (input_scale * weight_scale / output_scale)

    # Step 3: shift output by output_zero_point
    # Step 3: 加零点 — 补回输出零点偏移
    #         hint: one line of code
    output = output + output_zero_point
    ############### YOUR CODE ENDS HERE #################

    # Make sure all value lies in the bitwidth-bit range
    output = output.round().clamp(*get_quantized_range(feature_bitwidth)).to(torch.int8)
    return output


# 量化卷积
# 对于量化卷积层，我们首先预计算 Q_bias
# 回顾 Q_bias = q_bias - Conv[Z_input,q_weight]
def shift_quantized_conv2d_bias(quantized_bias, quantized_weight, input_zero_point):
    """
    shift quantized bias to incorporate input_zero_point for nn.Conv2d
        shifted_quantized_bias = quantized_bias - Conv(input_zero_point, quantized_weight)
    :param quantized_bias: [torch.IntTensor] quantized bias (torch.int32)
    :param quantized_weight: [torch.CharTensor] quantized weight (torch.int8)
    :param input_zero_point: [int] input zero point
    :return:
        [torch.IntTensor] shifted quantized bias tensor
    """
    assert quantized_bias.dtype == torch.int32
    assert isinstance(input_zero_point, int)
    return (
        quantized_bias
        - quantized_weight.sum((1, 2, 3)).to(torch.int32) * input_zero_point
    )


"""
问题 8

请完成以下量化卷积函数

提示：

q_output = (Conv[q_input,q_weight] + q_bias) * S_input * S_weight / S_output + Z_output
"""


def quantized_conv2d(
    input,
    weight,
    bias,
    feature_bitwidth,
    weight_bitwidth,
    input_zero_point,
    output_zero_point,
    input_scale,
    weight_scale,
    output_scale,
    stride,
    padding,
    dilation,
    groups,
):
    """
    quantized 2d convolution
    :param input: [torch.CharTensor] quantized input (torch.int8)
    :param weight: [torch.CharTensor] quantized weight (torch.int8)
    :param bias: [torch.IntTensor] shifted quantized bias or None (torch.int32)
    :param feature_bitwidth: [int] quantization bit width of input and output
    :param weight_bitwidth: [int] quantization bit width of weight
    :param input_zero_point: [int] input zero point
    :param output_zero_point: [int] output zero point
    :param input_scale: [float] input feature scale
    :param weight_scale: [torch.FloatTensor] weight per-channel scale
    :param output_scale: [float] output feature scale
    :return:
        [torch.(cuda.)CharTensor] quantized output feature
    """
    assert len(padding) == 4
    assert input.dtype == torch.int8
    assert weight.dtype == input.dtype
    assert bias is None or bias.dtype == torch.int32
    assert isinstance(input_zero_point, int)
    assert isinstance(output_zero_point, int)
    assert isinstance(input_scale, float)
    assert isinstance(output_scale, float)
    assert weight_scale.dtype == torch.float

    # Step 1: calculate integer-based 2d convolution (8-bit multiplication with 32-bit accumulation)
    input = torch.nn.functional.pad(input, padding, "constant", input_zero_point)
    if "cpu" in input.device.type:
        # use 32-b MAC for simplicity
        output = torch.nn.functional.conv2d(
            input.to(torch.int32),
            weight.to(torch.int32),
            None,
            stride,
            0,
            dilation,
            groups,
        )
    else:
        # current version pytorch does not yet support integer-based conv2d() on GPUs
        output = torch.nn.functional.conv2d(
            input.float(), weight.float(), None, stride, 0, dilation, groups
        )
        output = output.round().to(torch.int32)
    if bias is not None:
        output = output + bias.view(1, -1, 1, 1)

    ############### YOUR CODE STARTS HERE ###############
    # hint: this code block should be the very similar to quantized_linear()

    # Step 2: scale the output
    #         hint: 1. scales are floating numbers, we need to convert output to float as well
    #               2. the shape of weight scale is [oc, 1, 1, 1] while the shape of output is [batch_size, oc, height, width]
    output = output.float() * (input_scale * weight_scale / output_scale)

    # Step 3: shift output by output_zero_point
    #         hint: one line of code
    output = output + output_zero_point
    ############### YOUR CODE ENDS HERE #################

    # Make sure all value lies in the bitwidth-bit range
    output = output.round().clamp(*get_quantized_range(feature_bitwidth)).to(torch.int8)
    return output


"""
问题 9

最后，我们将所有内容整合起来，对模型进行训练后 int8 量化。我们将逐一将模型中的卷积层和线性层转换为量化版本

1. 首先，我们将 BatchNorm 层融合到卷积层中，这是量化前的标准做法。
    融合 BatchNorm 可以减少推理过程中的额外乘法运算

我们也将验证融合后的模型 model_fused 与原始模型具有相同的准确率(BN 融合是一种不改变网络功能的等价变换)
"""


def fuse_conv_bn(conv, bn):
    # modified from https://mmcv.readthedocs.io/en/latest/_modules/mmcv/cnn/utils/fuse_conv_bn.html
    assert conv.bias is None

    factor = bn.weight.data / torch.sqrt(bn.running_var.data + bn.eps)
    conv.weight.data = conv.weight.data * factor.reshape(-1, 1, 1, 1)
    conv.bias = nn.Parameter(-bn.running_mean.data * factor + bn.bias.data)

    return conv


print("Before conv-bn fusion: backbone length", len(model.backbone))
#  fuse the batchnorm into conv layers
recover_model()
model_fused = copy.deepcopy(model)
fused_backbone = []
ptr = 0
while ptr < len(model_fused.backbone):
    if isinstance(model_fused.backbone[ptr], nn.Conv2d) and isinstance(
        model_fused.backbone[ptr + 1], nn.BatchNorm2d
    ):
        fused_backbone.append(
            fuse_conv_bn(model_fused.backbone[ptr], model_fused.backbone[ptr + 1])
        )
        ptr += 2
    else:
        fused_backbone.append(model_fused.backbone[ptr])
        ptr += 1
model_fused.backbone = nn.Sequential(*fused_backbone)

print("After conv-bn fusion: backbone length", len(model_fused.backbone))
# sanity check, no BN anymore
for m in model_fused.modules():
    assert not isinstance(m, nn.BatchNorm2d)

#  the accuracy will remain the same after fusion
fused_acc = evaluate(model_fused, dataloader["test"])
print(f"Accuracy of the fused model={fused_acc:.2f}%")

"""
2. 我们将用一些样本数据运行模型，以获取每个特征图的范围，从而计算它们对应的缩放因子和零点
"""
# add hook to record the min max value of the activation
input_activation = {}
output_activation = {}


def add_range_recoder_hook(model):
    import functools

    def _record_range(self, x, y, module_name):
        x = x[0]
        input_activation[module_name] = x.detach()
        output_activation[module_name] = y.detach()

    all_hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ReLU)):
            all_hooks.append(
                m.register_forward_hook(
                    functools.partial(_record_range, module_name=name)
                )
            )
    return all_hooks


hooks = add_range_recoder_hook(model_fused)
sample_data = iter(dataloader["train"]).__next__()[0]
model_fused(sample_data.cuda())

# remove hooks
for h in hooks:
    h.remove()

# 3. 最后，让我们进行模型量化，我们将按以下映射关系转换模型
"""
nn.Conv2d -> QuantizedConv2d
nn.Linear -> QuantizedLinear
以下两个仅是包装器，因为当前的torch 模块不支持 int8 量化推理
我们将暂时将它们转换为 fp32 进行计算

nn.MaxPool2d -> QuantizedMaxPool2d
nn.AvgPool2d -> QuantizedAvgPool2d
"""


class QuantizedConv2d(nn.Module):
    def __init__(
        self,
        weight,
        bias,
        input_zero_point,
        output_zero_point,
        input_scale,
        weight_scale,
        output_scale,
        stride,
        padding,
        dilation,
        groups,
        feature_bitwidth=8,
        weight_bitwidth=8,
    ):
        super().__init__()
        # current version Pytorch does not support IntTensor as nn.Parameter
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

        self.input_zero_point = input_zero_point
        self.output_zero_point = output_zero_point

        self.input_scale = input_scale
        self.register_buffer("weight_scale", weight_scale)
        self.output_scale = output_scale

        self.stride = stride
        self.padding = (padding[1], padding[1], padding[0], padding[0])
        self.dilation = dilation
        self.groups = groups

        self.feature_bitwidth = feature_bitwidth
        self.weight_bitwidth = weight_bitwidth

    def forward(self, x):
        return quantized_conv2d(
            x,
            self.weight,
            self.bias,
            self.feature_bitwidth,
            self.weight_bitwidth,
            self.input_zero_point,
            self.output_zero_point,
            self.input_scale,
            self.weight_scale,
            self.output_scale,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class QuantizedLinear(nn.Module):
    def __init__(
        self,
        weight,
        bias,
        input_zero_point,
        output_zero_point,
        input_scale,
        weight_scale,
        output_scale,
        feature_bitwidth=8,
        weight_bitwidth=8,
    ):
        super().__init__()
        # current version Pytorch does not support IntTensor as nn.Parameter
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

        self.input_zero_point = input_zero_point
        self.output_zero_point = output_zero_point

        self.input_scale = input_scale
        self.register_buffer("weight_scale", weight_scale)
        self.output_scale = output_scale

        self.feature_bitwidth = feature_bitwidth
        self.weight_bitwidth = weight_bitwidth

    def forward(self, x):
        return quantized_linear(
            x,
            self.weight,
            self.bias,
            self.feature_bitwidth,
            self.weight_bitwidth,
            self.input_zero_point,
            self.output_zero_point,
            self.input_scale,
            self.weight_scale,
            self.output_scale,
        )


class QuantizedMaxPool2d(nn.MaxPool2d):
    def forward(self, x):
        # current version PyTorch does not support integer-based MaxPool
        return super().forward(x.float()).to(torch.int8)


class QuantizedAvgPool2d(nn.AvgPool2d):
    def forward(self, x):
        # current version PyTorch does not support integer-based AvgPool
        return super().forward(x.float()).to(torch.int8)


# we use int8 quantization, which is quite popular
feature_bitwidth = weight_bitwidth = 8
quantized_model = copy.deepcopy(model_fused)
quantized_backbone = []
ptr = 0
while ptr < len(quantized_model.backbone):
    if isinstance(quantized_model.backbone[ptr], nn.Conv2d) and isinstance(
        quantized_model.backbone[ptr + 1], nn.ReLU
    ):
        conv = quantized_model.backbone[ptr]
        conv_name = f"backbone.{ptr}"
        relu = quantized_model.backbone[ptr + 1]
        relu_name = f"backbone.{ptr + 1}"

        input_scale, input_zero_point = get_quantization_scale_and_zero_point(
            input_activation[conv_name], feature_bitwidth
        )

        output_scale, output_zero_point = get_quantization_scale_and_zero_point(
            output_activation[relu_name], feature_bitwidth
        )

        quantized_weight, weight_scale, weight_zero_point = (
            linear_quantize_weight_per_channel(conv.weight.data, weight_bitwidth)
        )
        quantized_bias, bias_scale, bias_zero_point = (
            linear_quantize_bias_per_output_channel(
                conv.bias.data, weight_scale, input_scale
            )
        )
        shifted_quantized_bias = shift_quantized_conv2d_bias(
            quantized_bias, quantized_weight, input_zero_point
        )

        quantized_conv = QuantizedConv2d(
            quantized_weight,
            shifted_quantized_bias,
            input_zero_point,
            output_zero_point,
            input_scale,
            weight_scale,
            output_scale,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            feature_bitwidth=feature_bitwidth,
            weight_bitwidth=weight_bitwidth,
        )

        quantized_backbone.append(quantized_conv)
        ptr += 2
    elif isinstance(quantized_model.backbone[ptr], nn.MaxPool2d):
        quantized_backbone.append(
            QuantizedMaxPool2d(
                kernel_size=quantized_model.backbone[ptr].kernel_size,
                stride=quantized_model.backbone[ptr].stride,
            )
        )
        ptr += 1
    elif isinstance(quantized_model.backbone[ptr], nn.AvgPool2d):
        quantized_backbone.append(
            QuantizedAvgPool2d(
                kernel_size=quantized_model.backbone[ptr].kernel_size,
                stride=quantized_model.backbone[ptr].stride,
            )
        )
        ptr += 1
    else:
        raise NotImplementedError(
            type(quantized_model.backbone[ptr])
        )  # should not happen
quantized_model.backbone = nn.Sequential(*quantized_backbone)

# finally, quantized the classifier
fc_name = "classifier"
fc = model.classifier
input_scale, input_zero_point = get_quantization_scale_and_zero_point(
    input_activation[fc_name], feature_bitwidth
)

output_scale, output_zero_point = get_quantization_scale_and_zero_point(
    output_activation[fc_name], feature_bitwidth
)

quantized_weight, weight_scale, weight_zero_point = linear_quantize_weight_per_channel(
    fc.weight.data, weight_bitwidth
)
quantized_bias, bias_scale, bias_zero_point = linear_quantize_bias_per_output_channel(
    fc.bias.data, weight_scale, input_scale
)
shifted_quantized_bias = shift_quantized_linear_bias(
    quantized_bias, quantized_weight, input_zero_point
)

quantized_model.classifier = QuantizedLinear(
    quantized_weight,
    shifted_quantized_bias,
    input_zero_point,
    output_zero_point,
    input_scale,
    weight_scale,
    output_scale,
    feature_bitwidth=feature_bitwidth,
    weight_bitwidth=weight_bitwidth,
)

# 量化过程完成！让我们打印并可视化模型架构，同时验证量化模型的准确率

"""
问题 9.1

要运行量化模型，我们需要额外的预处理步骤，将输入数据从 (0,1) 范围映射到 int8 的 (-128,127) 范围。请填写一下代码以完成额外的预处理

提示：你应该会发现量化模型的准确率与 fp32 版本大致相同
"""
print(quantized_model)


def extra_preprocess(x):
    # hint: you need to convert the original fp32 input of range (0, 1)
    #  into int8 format of range (-128, 127)
    ############### YOUR CODE STARTS HERE ###############
    # return torch.zeros_like(x).clamp(-128, 127).to(torch.int8)

    # return ( x* 255 - 128).round().clamp(-128, 127).to(torch.int8)
    return (x * 255).round().clamp(-128, 127).to(torch.int8) - 128

    ############### YOUR CODE ENDS HERE #################


int8_model_accuracy = evaluate(
    quantized_model, dataloader["test"], extra_preprocess=[extra_preprocess]
)
print(f"int8 model has accuracy={int8_model_accuracy:.2f}%")


"""
问题 9.2
请解释为什么线性量化模型中没有 ReLU 层。

答案:

  ReLU 在量化域中等价于一个整数比较操作, 不需要独立的神经网络层。

  推导:
    原始 fp32:       y = max(0, x)
    量化:            x = S × (q_x - Z)
    代入:            y = max(0, S × (q_x - Z))
    由于 S > 0:      y > 0  ⇔  q_x > Z
    所以:            q_y = max(Z, q_x)  = clamp(q_x, min=Z)

  也就是说, 量化域的 ReLU 就是把所有小于零点 Z 的值截断到 Z。
  这个操作在 quantized_conv2d / quantized_linear 的最后一步
  output.clamp_() 中已经包含了 — 只需要 clamp 到 [Z, q_max] 即可,
  不需要额外插入一个 ReLU 模块。

  为什么可以这样做:
  - 定点比较是纯整数运算, 零额外开销
  - 融合进前一层避免插入 quantize/dequantize 对(额外精度损失)

问题 10
请比较基于 k-means 的量化与线性量化的优缺点。你可以从准确率、延迟、硬件支持等方面进行讨论

答案:

  ┌──────────────┬────────────────────────────────┬──────────────────────────────────┐
  │              │ k-means 量化                    │ 线性量化                           │
  ├──────────────┼────────────────────────────────┼──────────────────────────────────┤
  │ 原理         │ 聚类 → 非均匀 centroids         │ 等距网格 → 统一 scale + zero_point │
  │ 量化值分布   │ 非均匀, centroids 可任意取值    │ 均匀, 相邻值间距固定为 S           │
  │ 精度(同bitwidth)│ 更高(centroids 可精确匹配分布) │ 对均匀分布好, 长尾分布差           │
  │ 反量化       │ 必须查表 centroids[label]       │ 纯数学公式 r = S×(q-Z)             │
  │              │ (多一次内存访问)                │ (一次乘法+加法, 1-2 cycle)         │
  │ 延迟         │ 较慢(查表开销)                  │ 较快(公式简单)                     │
  │ 硬件支持     │ 几乎没有                        │ 广泛: INT8 tensor core,            │
  │              │ (GPU 无 k-means 查表指令)       │ ARM NEON, x86 VNNI, NPU           │
  │ 码本存储     │ 需存储 2^n 个 fp32 centroids    │ 每通道 1 个 scale + 1 个 zero_point │
  │ 适用场景     │ 研究/极致压缩比/非均匀分布权重  │ 生产部署(移动端/边缘/服务器推理)   │
  │ QAT 训练     │ 每 step 重算 centroids(O(N))    │ 每 step 更新 scale/zp(O(1))        │
  └──────────────┴────────────────────────────────┴──────────────────────────────────┘

  总结:
  - 学术/探索场景 → k-means 量化(精度高但慢)
  - 工业/生产场景 → 线性量化(硬件原生支持, 推理快 2-4x)
  - 实际部署几乎 100% 用线性量化(INT8/INT4), k-means 主要用于量化研究
"""
