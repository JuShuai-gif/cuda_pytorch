import torch
import torch.nn as nn

from tqdm import tqdm
from sklearn.cluster import KMeans

__all__ = [
    "module_require_grad",
    "set_module_grad_status",
    "enable_bn_update",
    "enable_bias_update",
    "weight_quantization",
]


def module_require_grad(module):
    """检查模块是否需要计算梯度（通过第一个参数的 requires_grad 判断）"""
    return module.parameters().__next__().requires_grad


def set_module_grad_status(module, flag=False):
    """批量设置模块或模块列表的 requires_grad 状态。

    TinyTL 的核心操作之一：冻结主干权重（flag=False），仅保留少量可训练参数。
    """
    if isinstance(module, list):
        for m in module:
            set_module_grad_status(m, flag)
    else:
        for p in module.parameters():
            p.requires_grad = flag


def enable_bn_update(model):
    """仅启用 BatchNorm/GroupNorm 的权重更新（冻结卷积权重）。

    TinyTL 策略的一种变体：冻结卷积权重，但允许 BN/GroupNorm 的 weight/bias 继续训练，
    以适应新数据集的分布偏移。
    """
    for m in model.modules():
        if type(m) in [nn.BatchNorm2d, nn.GroupNorm] and m.weight is not None:
            set_module_grad_status(m, True)


def enable_bias_update(model):
    """仅启用偏置项（bias）的更新，冻结所有权重。

    TinyTL 的核心策略之一：卷积权重量大、更新成本高，而 bias 参数量极小，
    只更新 bias 即可在极低内存开销下完成迁移学习。
    """
    for m in model.modules():
        for name, param in m.named_parameters():
            if name == "bias":
                param.requires_grad = True


def k_means_cpu(weight, n_clusters, init="k-means++", max_iter=50):
    """使用 KMeans 聚类对权重进行量化聚簇。

    将连续的浮点权重值聚类为 n_clusters 个离散中心点，
    每个权重用最近的聚类中心替代，实现权重量化压缩。

    Returns:
        centroids: 聚类中心向量，shape 为 (1, n_clusters)
        labels: 每个权重所属的聚类索引，保持原始 shape
    """
    # 将权重展平为单特征向量，供 KMeans 处理
    org_shape = weight.shape
    weight = weight.reshape(-1, 1)  # single feature
    if n_clusters > weight.size:
        n_clusters = weight.size

    k_means = KMeans(
        n_clusters=n_clusters, init=init, n_init=1, max_iter=max_iter, n_jobs=20
    )
    k_means.fit(weight)

    centroids = k_means.cluster_centers_
    labels = k_means.labels_
    labels = labels.reshape(org_shape)
    return torch.from_numpy(centroids).view(1, -1), torch.from_numpy(labels).int()


def reconstruct_weight_from_k_means_result(centroids, labels):
    """从 KMeans 聚类结果重建量化后的权重张量。

    根据 labels 索引从 centroids 中取出对应的聚类中心值，
    还原为与原始权重相同 shape 的张量（但值已被量化为离散中心）。
    """
    weight = torch.zeros_like(labels).float()
    for i, c in enumerate(centroids.cpu().numpy().squeeze()):
        weight[labels == i] = c.item()
    return weight


def quantization(layer, bits=8, max_iter=50):
    """对单层的权重执行 KMeans 量化，并将量化后的权重写回层对象。

    将层的权重聚类为 2^bits 个离散值（例如 bits=8 → 256 个离散值）。
    这是一种压缩手段：用离散值替代连续浮点数，减少存储/传输开销。
    """
    w = layer.weight.data
    centroids, labels = k_means_cpu(w.cpu().numpy(), 2**bits, max_iter=max_iter)
    w_q = reconstruct_weight_from_k_means_result(centroids, labels)
    layer.weight.data = w_q.float()


def weight_quantization(model, bits=8, max_iter=50):
    """对模型中所有冻结层（Conv2d/Linear）的不可训练权重执行 KMeans 量化。

    TinyTL 场景下的典型用法：主干网络的卷积/全连接层权重已冻结，
    通过 KMeans 量化进一步压缩其存储（例如 32bit → 8bit），
    不影响训练过程（因为这些层不需要梯度）。
    """
    if bits is None:
        return

    # 收集所有需要量化的冻结层（Conv2d 和 Linear，且 requires_grad=False）
    to_quantize_modules = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if not m.weight.requires_grad:
                to_quantize_modules.append(m)

    with tqdm(
        total=len(to_quantize_modules), desc="%d-bits quantization start" % bits
    ) as t:
        for m in to_quantize_modules:
            quantization(m, bits, max_iter)
            t.update()
