# `tinytl/tinytl/` 子包总结

## 一句话定位

`tinytl/tinytl/` 是 TinyTL 的 **Python 核心子包**，提供网络构建（`model/`）、数据加载（`data_providers/`）、训练控制工具（`utils/`）三大模块，被上层训练脚本 `tinytl_fgvc_train.py` 调用。

## 目录结构

```
tinytl/
├── __init__.py                      # 空文件，标记为 Python 包
├── model/                           # 网络构建
│   ├── __init__.py                  # 导出 modules + network 的全部公开符号
│   ├── modules.py                   # LiteResidualModule + ReducedMBConvLayer + 层工厂函数
│   └── network.py                   # build_residual_block_from_config / build_network_from_config
├── data_providers/                  # 数据集加载与运行配置
│   ├── __init__.py                  # 导出 fgvc_data_providers + fgvc_run_config 的全部公开符号
│   ├── fgvc_data_providers.py       # 8 个 FGVC/通用数据集的 DataProvider 类定义
│   └── fgvc_run_config.py           # FGVCRunConfig（训练超参 + 数据集按名称分发 + 快速评估缓存）
└── utils/                           # 工具函数
    ├── __init__.py                  # 导出 common_utils + memory_cost_profiler 的全部公开符号
    ├── common_utils.py              # 梯度冻结/解冻控制 + KMeans 权重量化
    └── memory_cost_profiler.py      # 训练内存剖析（参数内存 + 激活峰值）
```

---

## 逐文件说明

### `__init__.py`（空）
仅标记该目录为 Python 包，无实际代码。

---

### `model/__init__.py`
```python
from .modules import *
from .network import *
```
将 `modules.py` 和 `network.py` 的所有 `__all__` 符号提升到 `tinytl.model` 命名空间，外部可通过 `from tinytl.model import LiteResidualModule` 直接引用。

---

### `model/modules.py`（338 行）
**整个 TinyTL 最核心的文件**，定义了端侧迁移学习的网络结构模块。

| 符号                     | 类型     | 作用                                                                                                                                                                                              |
| ------------------------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `my_set_layer_from_config` | 工厂函数 | 从配置字典构建层对象。支持 `LiteResidualModule` 和 `ReducedMBConvLayer`（TinyTL 特有），其他类型回退到 OFA 原生的 `set_layer_from_config`。是配置驱动的网络重建入口。                             |
| `LiteResidualModule`       | 模块类   | **TinyTL 核心创新**。包裹任意冻结的 MBConv 层，在同输入上运行极轻量旁路分支（池化 + depthwise Conv + 1×1 Conv + BN），与主分支输出相加。旁路初始化为零输出（final_bn.weight=0），训练中逐步学习。 |
| `LiteResidualModule.insert_lite_residual()` | 静态方法 | 遍历 `ProxylessNASNets` 的所有 blocks，在每一层 MBConv 外包裹 `LiteResidualModule`。自适应调整卷积核大小（不超过当前特征图分辨率）。**构造 TinyTL 模型的入口方法**。                                  |
| `LiteResidualModule.build_from_config()` | 静态方法 | 从序列化配置字典重建 `LiteResidualModule` 实例。                                                                                                                                                  |
| `LiteResidualModule.has_lite_residual_module()` | 静态方法 | 检测网络中是否已插入过 LiteResidual 模块（防止重复插入）。                                                                                                                                        |
| `ReducedMBConvLayer`       | 模块类   | 简化版 MBConv：expand depthwise Conv（+ 可选 SE）→ 1×1 reduce Conv。两步完成，比标准 MBConv（expand → depthwise → SE → project）更轻量但表达能力足够。                                             |

**LiteResidualModule 前向数据流**：
```
x ──┬── main_branch (冻结 MBConv) ────→ main_x ──┐
    │                                              ├──→ main_x + lite_residual_x
    └── lite_residual (可训练)                     │
         ├── AvgPool2d (downsample_ratio=2)        │
         ├── depthwise Conv (groups=n_groups)      │
         ├── BN + ReLU                             │
         ├── 1×1 Conv → BN (final_bn.weight=0)     │
         └── F.upsample (bilinear) ────→ lite_residual_x ──┘
```

---

### `model/network.py`（59 行）
从配置字典构建完整 TinyTL 网络。

| 符号                              | 类型   | 作用                                                                                                                                      |
| --------------------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `build_residual_block_from_config` | 函数   | 从配置构建单个残差块：通过 `my_set_layer_from_config` 解析 `conv` 和 `shortcut`，拼成 `ResidualBlock`。兼容 `conv` 和 `mobile_inverted_conv` 两种键名。 |
| `build_network_from_config`        | 函数   | 从配置字典构建完整网络：`first_conv → N×blocks → feature_mix_layer → classifier`，使用 `ProxylessNASNets` 容器组装，最后设置 BN 参数。     |

**输入**：一个 JSON 配置字典（包含 network、blocks、classifier 等键），通常从 OFA 预训练的 `net.config` 文件中加载。
**输出**：一个完整的 `ProxylessNASNets` 实例，其中 blocks 中的 conv 可能是 `LiteResidualModule`。

---

### `data_providers/__init__.py`
```python
from .fgvc_data_providers import *
from .fgvc_run_config import *
```

---

### `data_providers/fgvc_data_providers.py`（160 行）
定义 8 个数据集的 DataProvider 类，全部继承自 `FGVCDataProvider` → `ImagenetDataProvider`（OFA 的数据加载基础设施）。

| 符号                    | 类型      | 数据集        | 类别数 | 存储路径               |
| ----------------------- | --------- | ------------- | ------ | ---------------------- |
| `FGVCDataProvider`        | 基类       | —             | —      | 声明 name/n_classes/save_path 抽象接口 |
| `AircraftDataProvider`    | DataProvider | FGVC-Aircraft | 100    | `~/dataset/aircraft`       |
| `CarDataProvider`         | DataProvider | Stanford Cars | 196    | `~/dataset/stanford_car`   |
| `Flowers102DataProvider`  | DataProvider | Flowers-102   | 102    | `~/dataset/flowers102`     |
| `Food101DataProvider`     | DataProvider | Food-101      | 101    | `~/dataset/food101`        |
| `CUB200DataProvider`      | DataProvider | CUB-200-2011  | 200    | `~/dataset/cub200`         |
| `PetsDataProvider`        | DataProvider | Oxford Pets   | 37     | `~/dataset/pets`           |
| `CIFAR10DataProvider`     | DataProvider | CIFAR-10      | 10     | `~/dataset/cifar10`        |
| `CIFAR100DataProvider`    | DataProvider | CIFAR-100     | 100    | `~/dataset/cifar100`       |

每个子类只需覆写 3 个方法：`name()`（标识符）、`n_classes`（类别数）、`save_path`（本地路径）。FGVC 数据集需要手动下载到 save_path；CIFAR-10/100 额外覆写了 `train_dataset()` / `test_dataset()`，通过 `torchvision.datasets` 内置下载。

---

### `data_providers/fgvc_run_config.py`（77 行）
训练运行配置管理器，继承自 OFA 的 `ImagenetRunConfig`。

| 符号            | 类型 | 作用                                                                                                                                                                                                                                                      |
| --------------- | ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `FGVCRunConfig`   | 类   | 管理训练超参 + 数据集分发 + 数据加载器。核心功能：(1) `data_provider` 按 `--dataset` 名称延迟创建对应的 DataProvider 实例（支持 8 种数据集），结果缓存避免重复创建；(2) `valid_loader` / `test_loader` 在 `fast_evaluation=True` 模式下将所有数据一次性加载到内存列表，加速评估。 |

**关键属性**：
- `data_provider`：延迟创建并缓存，按名称匹配 `AircraftDataProvider.name()` 等
- `valid_loader`：`fast_evaluation=True` → 数据全量缓存到 `_in_memory_valid` 列表；否则从磁盘动态加载。`valid_size=None` 时回退到 test_loader。
- `test_loader`：同上逻辑，缓存到 `_in_memory_test` 列表。
- `fast_evaluation`：构造函数参数，默认 `True`。TinyTL 数据集规模不大，全量缓存到内存可行且大幅加速。

---

### `utils/__init__.py`
```python
from .common_utils import *
from .memory_cost_profiler import *
```

---

### `utils/common_utils.py`（117 行）
训练控制工具集，提供梯度管理和权重量化两大功能。

| 符号                                    | 类型 | 作用                                                                                                                                                  |
| --------------------------------------- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `module_require_grad(module)`             | 函数 | 判断模块是否需要梯度（通过第一个参数的 `requires_grad`）                                                                                              |
| `set_module_grad_status(module, flag)`    | 函数 | 批量设置模块（或模块列表）所有参数的 `requires_grad`。TinyTL 通过 `flag=False` 冻结主干网络的卷积权重。                                                |
| `enable_bn_update(model)`                 | 函数 | 仅解冻 BatchNorm/GroupNorm 的 weight/bias，冻结其余参数。一种轻量迁移学习基线：仅让 BN 适应新数据分布。                                                |
| `enable_bias_update(model)`               | 函数 | 仅解冻所有层的 bias 参数。TinyTL 核心策略之一：bias 参数量极小，更新开销几乎为零。                                                                    |
| `k_means_cpu(weight, n_clusters, ...)`    | 函数 | 使用 sklearn KMeans 将权重聚类为 `n_clusters` 个离散中心。返回聚类中心向量和 labels 张量。                                                              |
| `reconstruct_weight_from_k_means_result()` | 函数 | 根据 KMeans 结果重建量化后的权重张量。                                                                                                                |
| `quantization(layer, bits, ...)`          | 函数 | 对单层权重执行 KMeans 量化（聚类数 = `2^bits`），结果直接写回 `layer.weight.data`。                                                                    |
| `weight_quantization(model, bits, ...)`   | 函数 | 遍历模型中所有冻结的 Conv2d/Linear 层（`requires_grad=False`），执行 KMeans 量化压缩到 `2^bits` 个离散值。`bits=None` 时跳过。带 tqdm 进度条。          |

**调用关系**：
```
weight_quantization(model)
  └── quantization(layer)
        └── k_means_cpu(weight, 2^bits)
              └── KMeans.fit()
        └── reconstruct_weight_from_k_means_result()
```

---

### `utils/memory_cost_profiler.py`（166 行）
训练内存剖析工具，计算端侧训练所需的内存。

| 符号                   | 类型 | 作用                                                                                                                                                                                |
| ---------------------- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `count_model_size(net)`  | 函数 | 统计模型参数内存。区分可训练参数（默认 32bit=FP32）和冻结参数（默认 8bit=INT8）。冻结参数的位宽更小是 TinyTL 省内存的关键之一。                                                      |
| `count_activation_size(net)` | 函数 | 统计训练时的峰值激活内存。通过 hook 记录每层的输入/输出大小，再替换 forward 方法模拟训练时的内存累积/释放过程。区分不同层类型（Conv/Linear/BN/ReLU/Sigmoid）的梯度激活需求。 |
| `profile_memory_cost(net)`  | 函数 | 综合入口：`总内存 = 激活峰值 × batch_size + 参数量`。在训练脚本中用于剖析并记录到 `net_info.txt`。                                                                                |

**内存统计原理**：
1. **第一阶段（hook）**：注册 forward hook，记录每层 `tmp_activations`（临时激活）和 `grad_activations`（反向需要的梯度激活）
2. **第二阶段（forward 替换）**：替换每层 forward，在每个叶子节点调用前更新 `grad_activation_size`（累加）和 `peak_activation_size`（取 max）
3. **残差处理**：识别 `ResidualBlock`/`InvertedResidual` 等残差结构，在 shortcut 计算期间暂存输入大小到 `residual_size`

**调用关系**：
```
profile_memory_cost(net)
  ├── count_model_size(net)
  │     └── 遍历 net.parameters()，区分 requires_grad
  │
  └── count_activation_size(net)
        ├── Stage 1: model.apply(add_hooks) + model(x)
        │     └── 为每层注册 hook → 记录 tmp_activations / grad_activations
        │
        └── Stage 2: 替换 forward + model(x)
              └── 模拟训练内存分配/释放 → 计算峰值
```

---

## 模块间调用关系

```
tinytl_fgvc_train.py (上层入口)
    │
    ├── from tinytl.model import *
    │       ├── LiteResidualModule.insert_lite_residual(net)  ← 构建 TinyTL 网络
    │       └── build_network_from_config(config)             ← 从配置重建网络
    │
    ├── from tinytl.data_providers import *
    │       └── FGVCRunConfig(...)                            ← 数据加载 + 训练配置
    │             └── .data_provider → FGVCDataProvider 子类   ← 按名称分发数据集
    │
    └── from tinytl.utils import *
            ├── set_module_grad_status(...)                   ← 梯度冻结/解冻
            ├── enable_bias_update(...)                       ← 仅解冻 bias
            ├── enable_bn_update(...)                         ← 仅解冻 BN
            ├── weight_quantization(...)                      ← KMeans 量化冻结层
            └── profile_memory_cost(...)                      ← 训练前剖析内存
```

## 设计要点

1. **三类 `__all__` 导出**：每个子包的 `__init__.py` 通过 `from .xxx import *` 将所有公开符号平铺导出，外部只需 `from tinytl.model import LiteResidualModule` 即可使用。

2. **配置驱动**：`model/network.py` 中的网络构建函数 + `model/modules.py` 中的 `my_set_layer_from_config` + `LiteResidualModule.build_from_config` 支持从 JSON 配置字典完全重建网络结构（用于 OFA 的 specialized 分支）。

3. **与 OFA 深度耦合**：DataProvider 继承 OFA 的 `ImagenetDataProvider`、RunConfig 继承 `ImagenetRunConfig`、网络构建复用 `ProxylessNASNets`。TinyTL 子包是 OFA 框架的上层插件，不独立存在。
