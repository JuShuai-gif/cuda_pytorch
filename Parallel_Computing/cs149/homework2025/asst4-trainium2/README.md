# 作业 4：为机器学习加速器编程 #

**截止日期：11 月 13 日（周四）晚上 11:59**

**总分 100 分**

## 概述 ##

在本作业中，你将学习如何为 [AWS Trainium2](https://aws.amazon.com/ai/machine-learning/trainium/) 架构实现和优化内核。该架构配备了多个面向张量的加速处理引擎，以及软件管理的片上存储，为这些引擎提供高带宽的数据访问。

作业分为两部分。第 1 部分中，你将通过研究一些简单的向量加法内核并编写自己的矩阵转置内核，来熟悉 Trainium 架构和数据移动模式。第 2 部分中，你将在 Trainium2 上实现一个融合的卷积+最大池化层。

总体而言，本作业将：

1. 让你体验张量处理的底层细节以及加速器上片上 SRAM 的管理。

2. 展示关键局部性保持优化（如循环分块和循环融合）的价值。

## 环境配置 ##

你将在一台配备 Trainium 加速器的 AWS 虚拟机上编程和测试代码。请按照 [cloud_readme.md](cloud_readme.md) 中的说明设置运行作业的机器。

登录 AWS 机器后，你应从课程 GitHub 下载作业起始代码：

`git clone https://github.com/stanford-cs149/asst4-trainium2`

下载作业 4 仓库后，进入 `asst4-trainium2` 目录并**运行我们提供的安装脚本**：
```
cd asst4-trainium2
source install.sh
```
安装脚本将激活一个包含所有作业依赖项的 Python [虚拟环境](https://builtin.com/data-science/python-virtual-environment)。它还会修改你的 `~/.bashrc` 文件，以便将来登录机器时自动激活虚拟环境。最后，该脚本会设置你的 InfluxDB 凭证，以便使用 `neuron-profile`。

## 第 0 部分：熟悉 Trainium 和 Neuron Core 架构

### Trainium 架构概述

首先，让我们来认识 Trainium。

本作业使用的 `Trn2.3xlarge` 实例配备了一个 Trainium 设备，其中包含八个 NeuronCore。如下方图片所示，每个核都配有自己的专用 HBM（高带宽内存）。每个 NeuronCore 可视为一个独立的处理单元，包含自己的片上存储以及一组专用计算引擎，用于执行 128x128 矩阵运算（张量引擎）、128 宽度向量运算（向量引擎）等。虽然每个 Trainium 设备有八个 NeuronCore，但在本作业中，我们将编写在单个 NeuronCore 上执行的内核。

<p align="center">
  <img src="handout/trainium_chip.png" width=45% height=45%>
  <img src="handout/neuroncore_v3.png" width=30% height=30%>
</p>

有关 NeuronCore 中四个不同计算引擎的更多详细信息，请参见[此处](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/neuron-core-v3.html)。

### Trainium 内存层次结构

在作业 3 中，关键概念之一是学习 CUDA 所呈现的 GPU 内存层次结构：主内存、GPU 设备全局内存、每个线程块的共享内存以及每个 CUDA 线程的私有内存。在 Trainium 上，内存层次结构包括四个级别：**主内存（DRAM）**、**设备内存（HBM）**，以及两种快速的片上内存类型——**SBUF（状态缓冲区）**和 **PSUM（部分和缓冲区）**。在本作业中，我们只编写针对设备/片上内存的内核，因此可以忽略 DRAM（位于 Trainium 设备外部），专注于 HBM、SBUF 和 PSUM。

<p align="center">
  <img src="handout/memory_hierarchy.png" width=80% height=80%>
</p>

* **HBM** 是位于 Trainium 设备上的高带宽内存。HBM 作为设备的主内存，提供大容量存储（96 GiB）。在内核外部创建的大多数数据类型（如 NumPy 数组）默认分配在 HBM 中。
* **SBUF** 是 NeuronCore 上的片上存储。相比之下，SBUF 比 HBM 小得多（28 MiB），但提供更高的带宽（约为 HBM 的 20 倍）。程序员必须显式地将数据移入和移出 SBUF，才能使用 NeuronCore 执行计算。
* **PSUM** 是一个小型专用内存区（2 MiB），专用于存放张量引擎产生的矩阵乘法结果。

<p align="center">
  <img src="handout/neuron_core.png" width=40% height=40%>
</p>

回想一下，在具有传统数据缓存的系统中，关于哪些片外内存中的数据被复制并存储在片上存储中的决策由缓存做出（基于缓存组织和驱逐策略）。软件在给定的内存地址加载数据，硬件负责从内存中获取该数据并管理缓存中存储的内容，以便未来高效访问。换句话说，从软件正确性的角度来看，缓存并不存在——它只是硬件的实现细节。

相比之下，NeuronCore 可用的内存是**软件管理的**。这意味着软件必须使用数据移动命令显式地将数据移入和移出这些内存。要么程序员必须在程序中显式描述数据移动，要么 NKI 编译器必须分析应用程序并生成适当的数据移动操作。高效使用 NeuronCore 架构的一些最大挑战涉及高效地编排数据在机器中的移动。

## 第 1 部分：通过向量加法和矩阵转置学习 Neuron 内核接口（30 分）

在本节中，我们通过提供将一个向量加法应用程序的几种不同实现，来介绍 Trainium 编程模型的基础知识。然后我们将编写一个简单的二维矩阵转置内核。

相应代码组织在 `/part1` 目录中。具体来说，这里讨论的向量加法内核可以在 `kernels.py` 中找到。此外，我们提供了一个脚本 `run_benchmark.py`，它提供了方便的命令行接口来使用不同向量大小执行这些内核。该脚本还包含一个可选标志，用于收集性能分析指标。

```
用法: run_benchmark.py [-h] --kernel {naive,tiled,stream,transpose} -n N [-m M] [--profile_name PROFILE_NAME]

选项:
  -h, --help            显示此帮助信息并退出
  --kernel {naive,tiled,stream,transpose}
  -n N
  -m M
  --profile_name PROFILE_NAME
                        用于保存 .NEFF 和 .NTFF 文件的名称
```

### NKI 编程模型：

Neuron 内核接口（NKI）是一种用于开发在 Trainium 设备上运行的内核的语言和编译器。NKI 内核用 Python 编写，并使用三种类型的 NKI 操作：
1. **加载数据**：从 HBM 到片上 SBUF。
2. **计算**：在 NeuronCore 计算引擎上执行。
3. **存储输出**：从 SBUF 回到 HBM。

例如，以下内核定义了如何使用 NKI 执行向量加法。请注意，`@nki.jit` 是一个 Python 装饰器，表示该函数应编译为在 NeuronDevice 上运行，类似于 CUDA C++ 中的 `__global__` 函数装饰器指定一个函数编译为设备端函数并在 GPU 上运行。

类似于 CUDA 内核的参数是 CUDA 设备全局内存中的数组，被 `@nki.jit` 装饰的 Python 函数的参数是驻留在 NeuronCore 可访问的 HBM 中的张量。`@nki.compiler.skip_middle_end_transformations` 装饰器禁用了一些可能以意想不到的方式转换内核的编译器优化，这将使调试更容易。

在以下代码中，`a_vec` 和 `b_vec` 被假定为 HBM 中长度为 128 的向量。（该代码不适用于大于 128 的向量。我们稍后将解释原因。）
```
@nki.compiler.skip_middle_end_transformations
@nki.jit
def vector_add_naive(a_vec, b_vec):
    
    # 在 HBM 中为输出向量分配空间
    out = nl.ndarray(shape=a_vec.shape, dtype=a_vec.dtype, buffer=nl.hbm)

    # 在 SBUF 中为输入向量分配空间并从 HBM 复制它们
    a_sbuf = nl.ndarray(shape=(a_vec.shape[0], 1), dtype=a_vec.dtype, buffer=nl.sbuf)
    b_sbuf = nl.ndarray(shape=(b_vec.shape[0], 1), dtype=b_vec.dtype, buffer=nl.sbuf)
    
    nisa.dma_copy(src=a_vec, dst=a_sbuf)
    nisa.dma_copy(src=b_vec, dst=b_sbuf)

    # 将输入向量相加
    res = nisa.tensor_scalar(a_sbuf, nl.add, b_sbuf)

    # 将结果存储到 HBM
    nisa.dma_copy(src=res, dst=out)

    return out
```

在上述代码中……

- `a_vec` 和 `b_vec` 是在内核外部创建的 NumPy 数组，驻留在 HBM 中。
- `a_sbuf` 和 `b_sbuf` 是在 SBUF 中显式分配的数组，形状和 dtype 与 `a_vec` 和 `b_vec` 相同。
- `nisa.tensor_scalar(..., nl.add, ...)` 使用向量引擎执行向量加法。签名 `tensor_scalar` 表示第二个操作数预期为一个向量，即形状为 (N, 1)，或一个常量标量，这比一般的 `tensor_tensor` 操作稍快一些。
- `nisa.dma_copy` 在 HBM 和 SBUF 之间移动相关数据（概念上类似于 NVIDIA GPU 上的 `cudaMemcpyAsync`）。

<p align="center">
  <img src="handout/sbuf_layout.png" width=60% height=60%>
</p>

**查看上述代码时，请注意 NKI 操作是对张量（而非标量值）进行操作的。** 具体而言，片上内存 SBUF 和 PSUM 存储排列为二维内存数组的数据。二维数组的第一个维度称为"分区维度"`P`。第二个维度称为"自由维度"`F`。NeuronCore 能够并行加载和处理沿分区维度的数据，**但架构还施加了一个限制，即分区维度的大小为 128 或更小。**
换句话说，从 HBM 加载张量到 SBUF 时，张量的分区维度最多为 128。我们稍后会讨论自由维度的限制。

因此，在上述代码中，由于 `a_vec` 和 `b_vec` 是一维向量，它们唯一的维度就是分区维度，因此它们的大小限制为 128 个元素。换句话说，该代码仅适用于向量大小为 128 或更小的情况。

### 步骤 1：将向量分块以跨 128 个计算通道并行化（6 分）

要使代码适用于大小大于 128 的向量，我们需要以块（原始张量的子集）的形式加载向量。

```
@nki.compiler.skip_middle_end_transformations
@nki.jit
def vector_add_tiled(a_vec, b_vec):
    
    # 在 HBM 中为输出向量分配空间
    out = nl.ndarray(shape=a_vec.shape, dtype=a_vec.dtype, buffer=nl.hbm)

    # 获取向量的总行数
    M = a_vec.shape[0]
    
    # TODO: 步骤 1 中你应该修改此变量
    ROW_CHUNK = 1

    # 遍历总块数，我们可以使用 affine_range
    # 因为没有循环携带的依赖关系
    for m in nl.affine_range(M // ROW_CHUNK):

        # 为输入向量分配行块大小的分块
        a_tile = nl.ndarray((ROW_CHUNK, 1), dtype=a_vec.dtype, buffer=nl.sbuf)
        b_tile = nl.ndarray((ROW_CHUNK, 1), dtype=b_vec.dtype, buffer=nl.sbuf)
        
        # 加载一个行块
        nisa.dma_copy(src=a_vec[m * ROW_CHUNK : (m + 1) * ROW_CHUNK], dst=a_tile)
        nisa.dma_copy(src=b_vec[m * ROW_CHUNK : (m + 1) * ROW_CHUNK], dst=b_tile)

        # 将行块相加
        res = nisa.tensor_scalar(a_tile, nl.add, b_tile)

        # 将结果块存储到 HBM
        nisa.dma_copy(src=res, dst=out[m * ROW_CHUNK : (m + 1) * ROW_CHUNK])
    
    return out
```

上述示例将向量行分解为单元素块（块大小为向量的 1 个元素——是的，这效率很低，我们稍后会回来讨论这个问题）。这是通过使用标准 Python 切片语法 `Tensor[Index:Index:...]` 对向量进行索引来实现的。有关 NKI 中张量索引的更多详细信息，请参见[此处](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/programming_model.html#nki-tensor-indexing)。

在上述代码中，使用的 [affine_range](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.language.affine_range.html) 为循环迭代器生成一系列数字，类似于 Python 的 `range` 函数，但它要求迭代之间没有循环携带的依赖关系。对于存在循环携带依赖关系的情况，NKI 还提供了 [sequential_range](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.language.sequential_range.html)。

通常，`affine_range` 让 NKI 编译器可以更积极地优化循环迭代，以在计算引擎之间实现更多的流水线处理。然而，由于我们为了透明度/可重现性而禁用了编译器优化，这两个结构实际上是相同的。

**你需要做什么：**
1. 运行上述 `vector_add_tiled` 实现，其中 *row_chunk = 1*，向量大小为 25600（*这可能需要几分钟*）。你可以使用以下命令执行：
   ```
   python run_benchmark.py --kernel tiled -n 25600
   ```
   执行时间是多少微秒（μs）？

2. 记住，NeuronDevice 上一次可以加载的最大分区大小（行数）是 128。在 `kernels.py` 中，修改 `vector_add_tiled`，使其使用 *row_chunk = 128*。记录在使用 *row_chunk = 128* 对向量大小 25600 进行操作时 `vector_add_tiled` 的执行时间（以微秒 μs 为单位）。当 *row_chunk = 128* 时，`vector_add_tiled` 在向量大小 25600 上比 *row_chunk = 1* 时快多少？你认为为什么更快？（*提示：* 你应该将执行视为从 HBM 并行加载 `ROW_CHUNK` 个元素，然后对 SBUF 中的向量执行 `ROW_CHUNK` 宽度的向量加法。）

3. 尝试在 *row_chunk = 256* 时以向量大小 25600 运行 `vector_add_tiled`。你应该会看到一个错误。用一句话解释为什么在尝试运行 *row_chunk = 256* 时会出错。

### 步骤 2a：改进的数据流式传输（4 分）

到目前为止，我们已经能够利用向量引擎可以并行使用所有 128 个向量通道执行计算的事实，每个通道将一个元素流式传输到/从单个 SBUF/PSUM 分区。

然而，我们可以通过沿自由维度流式传输更多元素来进一步提高性能。为此，让我们更多地思考直接内存访问（DMA）传输。你应该将 DMA 传输（即对 `nisa.dma_copy` 的调用）视为一个异步操作，将数据块从 HBM 移动到 SBUF 或反之。

每个 NeuronCore 有 16 个 DMA 引擎，它们都可以并行处理不同的数据传输操作。需要注意的是，设置 DMA 传输并分配 DMA 引擎来处理它们会产生开销。为了减少这种设置开销，高效的实现应该力求在每次传输中移动大量数据，以分摊 DMA 传输开销。

虽然 SBUF 张量的第一维度（分区维度）不能大于 128，但单个 SBUF 向量指令的第二维度可以高达 64K 个元素。这意味着可以使用单条指令从 HBM 加载 128 × 64k = 8192k 个元素到 SBUF。此外，我们可以通过单条 `nisa.tensor_tensor` 指令对两个 8192k 元素的 SBUF 块执行向量加法。因此，不应该为向量的每个 128 元素块执行一次 `nisa.dma_copy`，而是应该使用每次 DMA 传输请求移动多个 128 行的块。这种精简的方法使我们能够分摊传输数据所需的设置时间。

为了减少 DMA 传输开销，我们需要将向量重塑为二维分块，而不是线性数组。在作业 3 中，我们使用 CUDA 线程块分区处理整个图像，为了将 CUDA 线程映射到图像像素，我们通过计算线程的全局线性索引来展平网格。你可以将 NeuronCore 的重塑过程视为相反的操作：目标是将一维向量转换为密集的二维矩阵。NumPy 内置了 [reshape 函数](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html)，允许你将数组重塑为你选择的形状。

<p align="center">
  <img src="handout/non_reshaped_DMA.png" width=48% height=48%>
  <img src="handout/reshaped_DMA.png" width=48% height=48%>
</p>


看看 `vector_add_stream`，它扩展了 `vector_add_tiled` 以减少 DMA 传输：
```
@nki.compiler.skip_middle_end_transformations
@nki.jit
def vector_add_stream(a_vec, b_vec):

    # 获取向量的总行数
    M = a_vec.shape[0]

    # TODO: 步骤 2a 中你应该修改此变量
    FREE_DIM = 2

    # 我们的分区维度的最大大小
    PARTITION_DIM = 128

    a_vec_re = a_vec.reshape((PARTITION_DIM, M // PARTITION_DIM))
    b_vec_re = b_vec.reshape((PARTITION_DIM, M // PARTITION_DIM))
    out = nl.ndarray(shape=a_vec_re.shape, dtype=a_vec_re.dtype, buffer=nl.hbm)

    # 遍历总分块数
    for m in nl.affine_range(M // (PARTITION_DIM * FREE_DIM)):

        # 为重塑的分块分配空间
        a_tile = nl.ndarray((PARTITION_DIM, FREE_DIM), dtype=a_vec.dtype, buffer=nl.sbuf)
        b_tile = nl.ndarray((PARTITION_DIM, FREE_DIM), dtype=b_vec.dtype, buffer=nl.sbuf)

        # 加载输入分块
        nisa.dma_copy(src=a_vec_re[:, m * FREE_DIM : (m + 1) * FREE_DIM], dst=a_tile)
        nisa.dma_copy(src=b_vec_re[:, m * FREE_DIM : (m + 1) * FREE_DIM], dst=b_tile)

        # 将分块相加。注意我们必须切换为 tensor_tensor 而不是 tensor_scalar
        res = nisa.tensor_tensor(a_tile, b_tile, op=nl.add)

        # 将结果分块存储到 HBM
        nisa.dma_copy(src=res, dst=out[:, m * FREE_DIM : (m + 1) * FREE_DIM])

    # 将输出向量重塑回其原始形状
    out = out.reshape(a_vec.shape)

    return out
```

**你需要做什么：**
1. 运行上述 `vector_add_stream` 实现，其中 *FREE_DIM = 2*。对向量大小 25600 运行需要多少微秒（μs）？与步骤 1 中使用 *row_chunk = 128* 的 `vector_add_tiled` 相比，这快了多少？
2. 当前 `vector_add_stream` 实现略微减少了 DMA 传输次数，但 DMA 传输次数还可以进一步减少。在 `kernels.py` 中，更改 `vector_add_stream` 的 *FREE_DIM* 值，以尽可能减少在大小为 25600 的向量上的 DMA 传输次数。

   你选择了什么 *FREE_DIM* 值？对于该 *FREE_DIM* 值，在向量大小 25600 上的执行时间是多少微秒（μs）？

   你选择的 *FREE_DIM* 值的 `vector_add_stream` 比 *FREE_DIM = 2* 的 `vector_add_stream` 快多少？你选择的 *FREE_DIM* 值的 `vector_add_stream` 比 *row_chunk = 128* 的 `vector_add_tiled` 快多少？

### 步骤 2b：学习使用 Neuron-Profile（5 分）

选择分块的自由维度大小存在权衡：
1. 分块太小会暴露显著的指令开销，导致引擎执行效率低下。
2. 分块太大通常会导致引擎之间的流水线效率低下，以及在数据重用情况下 SBUF 中的高内存压力（"内存压力"指 SBUF 可能会被填满）。

目前，我们已经探索了将分块大小增加到最大以分摊指令开销和 DMA 传输设置/拆除的好处。现在，我们将探讨为什么将自由维度设为尽可能大并不总是最佳解决方案。

对于此任务，你需要使用 NeuronDevice 的性能分析工具：`neuron-profile`，它可以提供在 NeuronCore 上运行的应用程序性能的详细分析。为了运行分析工具，你必须确保按照[环境配置](#环境配置)中的说明运行了安装脚本，并且在 ssh 登录到机器时转发了端口 3001 和 8086。重申后者，你应该运行的命令是：

 `ssh -i path/to/key_name.pem ubuntu@<public_dns_name> -L 3001:localhost:3001 -L 8086:localhost:8086`
 
 有关为什么需要这样做的更多详细信息，请参见 [cloud_readme.md](cloud_readme.md)。

**你需要做什么：**
1. 这次，我们将向量大小增加 10 倍，所以我们不是添加 25600 个元素，而是添加 256000 个元素。这将使我们能够看到由太大分块大小带来的权衡。

   首先，在 `vector_add_stream` 中设置 *FREE_DIM = 2000*。现在，就像之前的步骤一样，我们将执行内核，但这次我们将编译后的内核保存到 **.neff** 文件中，并将内核执行跟踪保存到 **.ntff** 跟踪文件中。让我们对向量大小为 256000 运行 `vector_add_stream`，并将编译后的内核和跟踪保存到前缀为 `stream_256k_fd2k` 的文件中，使用以下命令：

   ```
   python run_benchmark.py --kernel stream -n 256000 --profile_name stream_256k_fd2k
   ```

   你应该生成了两个文件：***stream_256k_fd2k.neff*** 和 ***stream_256k_fd2k.ntff***。（你可能会在 stdout 中看到一条错误消息，说"hw profiler overview not found"——这可以安全地忽略，不必担心。）

   现在，使用类似的工作流程，在 `vector_add_stream` 中设置 *FREE_DIM = 1000* 并对向量大小 256000 运行，将编译后的内核和跟踪保存到前缀为 `stream_256k_fd1k` 的文件中。

2. 这些生成的文件将允许我们使用 `neuron-profile` 工具收集内核执行指标。这些分析指标对于分析内核的性能非常有用。让我们通过运行以下命令来查看 *FREE_DIM = 2000* 的 `vector_add_stream` 的执行指标简要摘要：

   ```
   neuron-profile view --output-format summary-text -n stream_256k_fd2k.neff -s stream_256k_fd2k.ntff
   ```

   你将看到包含各种执行指标的摘要输出，按字母顺序排列。让我们查看两个特定的指标：

    * **dma_transfer_count**：DMA 传输次数
    * **total_time**：内核执行时间（以秒为单位）

   当 *FREE_DIM = 2000* 时，内核执行时间是多少秒？进行了多少次 DMA 传输？

   使用与之前相同的工作流程，查看当 *FREE_DIM = 1000* 时的执行指标摘要。

   当 *FREE_DIM = 1000* 时，内核执行时间是多少秒？进行了多少次 DMA 传输？

3. 虽然 *FREE_DIM = 1000* 的内核有更多的 DMA 传输，但它更快！让我们分析原因。

   我们可以使用 `neuron-profile` 的 GUI 功能更深入地了解内核执行指标。让我们通过运行以下命令为 *FREE_DIM = 2000* 的 `vector_add_stream` 启动 GUI：

   ```
   neuron-profile view -n stream_256k_fd2k.neff -s stream_256k_fd2k.ntff
   ```

   运行命令后，你将看到类似以下的输出：

   `View profile at http://localhost:3001/profile/...`

   将此 *http* 链接粘贴到你选择的浏览器中，以查看更深入的分析工具分析。（可以随意忽略页面顶部出现的任何警告。）

> [!NOTE]
> 仅当你在 ssh 到机器时正确转发了端口 3001 和 8086 时，你才能看到此内容。

   你应该会看到分析工具生成的图表，描绘了不同引擎随时间发出的指令。

   为了我们的目的让查看更容易，到底部的 `View Settings` 并执行以下操作：
   * 将 `Instructions color group` 更改为 `Instruction Type`
   * 在 `Timeline display options` 下关闭 `Show individual NeuronCore layout`
   * 在 `DMA display options` 下关闭 `Show expanded DMA`
   * 点击最底部的 `Save`。

   完成这些步骤后，分析工具图表应如下所示：

   ![Profiler GUI Example](handout/profiler_gui.png)

   你还可以将鼠标悬停在图表中的各种事件上以查看更多信息。尝试将鼠标悬停在以下类别的事件上：

   * **DMA-E79**：显示 DMA 引擎将输入和输出数据移动到/移出相应缓冲区（计算指令数量——这与对 `nisa.dma_copy` 的预期调用次数是否匹配？）
   * **VectorE**：显示向量引擎通过 `nisa.tensor_tensor` 将两个输入向量相加（这应该以绿色突出显示）
   * **Pending DMA Count**：显示随时间推移的待处理 DMA 传输数量
   * **DMA Throughput**：显示随时间推移的设备带宽利用率

   现在，在终端中按 `ctrl-c` 退出当前的 `neuron-profile view`。请注意，你仍然可以在浏览器中查看 *FREE_DIM = 2000* 的 `vector_add_stream` 的 GUI 分析，因为它们已临时存储在数据库中。按照相同的工作流程，为 *FREE_DIM = 1000* 的 `vector_add_stream` 启动 GUI 分析。

4. 在分析了 *FREE_DIM = 2000* 和 *FREE_DIM = 1000* 的 `vector_add_stream` 的 GUI 分析图表后，简要解释为什么 FREE_DIM = 1000 比 FREE_DIM = 2000 执行时间更快，尽管它需要更多的 DMA 传输（*提示：* 流水线）。

   你也可以随意探索 `neuron-profile` GUI 中的各种功能。你可能还想查看底部工具栏中的 `Summary` 标签。此标签显示我们在问题 2 中运行 `neuron-profile view --output-format summary-text ...` 时看到的相同执行指标简要摘要。随意从[用户指南](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-profile-user-guide.html)和 NKI 内核的[性能指南](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/nki_perf_guide.html)中了解更多关于 `neuron-profile` 功能和 NKI 内核有趣性能指标的信息。

### 步骤 3：矩阵转置（15 分 = 10 分编码 + 5 分书面报告（+ 1 分加分））
### NeuronCore 上的矩阵运算
在你开始之前，我们将演示如何在 NeuronCore 上执行矩阵运算。如前所述，NeuronCore 配备了各种计算引擎，每个引擎都针对特定类型的算术运算进行了优化。Trainium 上的张量引擎专门设计用于加速这些矩阵运算，如矩阵乘法和矩阵转置。

<p align="center">
  <img src="handout/tensor_engine.png" width=60% height=60%>
</p>

上图描绘了张量引擎的架构。张量引擎围绕一个 128x128 的[脉动处理阵列](https://gfxcourses.stanford.edu/cs149/fall25/lecture/proghardware/slide_10)构建，该阵列从 SBUF（片上存储）流式传输矩阵数据输入，并将输出写入 PSUM（也在片上存储中）。与 SBUF 一样，PSUM 是快速的片上内存，但它比 SBUF 小得多（2 MiB 对比 28 MiB），并且专门用于存储张量引擎计算的矩阵乘法结果。张量引擎能够对 PSUM 中的每个地址进行读取-加法-写入。因此，PSUM 在以分块方式执行大型矩阵乘法时非常有用，其中每个矩阵乘法的结果会累积到同一个输出分块中。

### 编写内核
在这里，你将尝试编写自己的小型内核，使用张量引擎转置矩阵，然后再进入第 2 部分涉及实际矩阵乘法的更复杂内核。查看 `kernels.py` 中的起始代码。你的内核应接受形状为 (M, N) 的单个二维张量作为输入，并返回形状为 (N, M) 的二维张量。对 M 和 N 的唯一限制是两者都能被最大分区维度 128 整除。

```
@nki.compiler.skip_middle_end_transformations
@nki.jit
def matrix_transpose(a_tensor):
    M, N = a_tensor.shape
    out = nl.ndarray((N, M), dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    tile_dim = 128

    assert M % tile_dim == N % tile_dim == 0, "矩阵维度不能被分块维度整除！"

    # TODO: 你的实现在这里。你应该使用的唯一计算指令是 `nisa.nc_transpose`。

    return out
```

要实际执行转置，你必须调用 [nisa.nc_transpose](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_transpose.html#nki.isa.nc_transpose)，这是一个使用张量引擎转置最大 128x128 大小的分块的内置指令，结果存储在 PSUM 中。你**不**允许使用其他计算指令，包括 `nisa.dma_transpose` 或 `nl.transpose`。（内存指令，包括 `nisa.dma_copy` 和 `nl.ndarray`，当然是被允许的。）

由于你将转置远大于 128x128 的矩阵，你的内核应管理数据分块进出 HBM/SBUF 的移动。重新阅读之前的向量加法内核，了解它们如何分配和移动数据，可能会很有用。

> [!TIP]
> `nisa.dma_copy` 仅适用于 SBUF/HBM 中的张量。由于 `nisa.nc_transpose` 的输出是一个 PSUM 分块，你需要先将其复制到 SBUF。你可能会发现 [`nisa.tensor_copy`](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.tensor_copy.html#nki.isa.tensor_copy) 对此很有用。

**你需要做什么：**
1. 用你的实现填充内核。然后通过运行以下命令在 1024x1024 矩阵上进行测试：
   ```
   python run_benchmark.py --kernel transpose -n 1024
   ```
   并记录执行时间（以微秒 μs 为单位）。
2. 在不使用分析工具的情况下，你认为你的内核是内存受限的还是计算受限的？解释你的答案。然后，通过分析与 `vector_add_stream` 相同的方式对你的代码进行分析来确认这一点。（你可以附上截图，但请提供书面描述说明它如何验证你的答案。）
3. **(加分，1 分)** 优化你的实现以最小化延迟。要获得学分，你应该能在 4096x4096 的转置上达到 <700 μs。确保在**不**传递 `--profile_name` 的情况下测量延迟（分析工具会改变执行时间）。

   对于这部分，可以随意尝试 `nisa.nc_transpose` 之外的其他 API。也请提交一份简要书面报告，说明你是如何识别性能瓶颈并解决它们的。

## 第 2 部分：实现融合卷积 - 最大池化层（70 分）

现在你已经学会了如何在 NeuronCore 上高效移动数据，是时候自己编写一个实际的 Trainium 内核了。在本节中，你的任务是实现一个同时执行卷积和称为"最大池化"的操作的内核。正如我们在课堂上讨论的，这两个操作是现代卷积神经网络（CNN）的基本组件，广泛用于计算机视觉任务。一个重要细节是，你对这两个操作的实现将是"融合"的，意味着你将在 Trainium 上实现计算，而不会将中间值转储到片外 HBM。

### NKI 矩阵乘法内核

回想一下，向量引擎能够对大小为 (128, 64k) 的 SBUF 分块进行操作。然而，张量引擎包含独特的 SBUF 分块大小约束，这与向量引擎不同。假设我们希望张量引擎执行矩阵乘法 C = A × B，其中 A 和 B 位于 SBUF 中，结果 C 存储在 PSUM 中。Trainium 施加了以下约束：
  - 矩阵 A——左手分块——不能大于 (128, 128)
  - 矩阵 B——右手分块——不能大于 (128, 512)。
  - PSUM 中的输出分块 C 限制为 (128, 512) 的大小。

鉴于张量引擎的这些约束，在 Trainium 上为任意矩阵维度实现矩阵乘法需要将计算分块，使其作为在固定大小分块上的一系列矩阵乘法执行。（这类似于第 1 部分中向量加法如何被分块以处理大型输入向量大小。）以下示例是我们从 [NKI 教程](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/tutorials)修改而来的，演示了如何使用分块方法实现矩阵乘法，其中分块的大小符合 Trainium 张量引擎的分块大小约束。注意：代码列表后附有代码说明。

```
@nki.compiler.skip_middle_end_transformations
@nki.jit
def nki_matmul_tiled_(lhsT, rhs, result):
  """NKI 内核，以分块方式计算矩阵乘法运算"""

  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT 和 rhs 必须具有相同的压缩维度"

  # 张量引擎上一般矩阵乘法的固定操作数的最大自由维度
  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128

  # 分块的最大分区维度
  TILE_K = nl.tile_size.pmax  # 128

  # 张量引擎上一般矩阵乘法的移动操作数的最大自由维度
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  # 使用 affine_range 遍历分块
  for m in nl.affine_range(M // TILE_M):
    for n in nl.affine_range(N // TILE_N):
      # 在 PSUM 中分配一个张量
      res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

      for k in nl.affine_range(K // TILE_K):
        # 在 SBUF 上声明分块
        lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
        rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)

        # 从 lhsT 和 rhs 加载分块
        nisa.dma_copy(dst=lhsT_tile, src=lhsT[k * TILE_K:(k + 1) * TILE_K, m * TILE_M:(m + 1) * TILE_M])
        nisa.dma_copy(dst=rhs_tile, src=rhs[k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N])

        # 将部分和累积到 PSUM
        res_psum += nisa.nc_matmul(lhsT_tile[...], rhs_tile[...])

      # 将结果从 PSUM 复制回 SBUF，并转换为期望的输出数据类型
      res_sb = nl.copy(res_psum, dtype=result.dtype)
      nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N], src=res_sb)
```

让我们分解计算矩阵乘法 `result = lhsT × rhs` 的内核的组成部分：

  - 输入张量：
      - `lhsT` 是左手边矩阵。但该矩阵以**转置格式**提供，形状为 `[K, M]`，其中 `K` 和 `M` 都是 128 的倍数。
      - `rhs` 是右手边矩阵，形状为 `[K, N]`，其中 `K` 是 128 的倍数，`N` 是 512 的倍数。
      - `result` 是形状为 `[M, N]` 的输出矩阵
      - 在矩阵乘法中，**压缩维度**指的是左手矩阵的列维度和右手矩阵的行维度。例如，假设我们有如下矩阵乘法：`A × B = C`。矩阵 `A` 的形状为 `[M, N]`，矩阵 `B` 的形状为 `[N, M]`。那么 `C` 的形状是 `[M, M]`。因此，被消除的维度是 `A` 的列维度和 `B` 的行维度。
      - 请注意，在上面的 `nki_matmul_tiled_` 示例中，矩阵是转置形式，其中 `lhsT=A^T`。`nisa.nc_matmul` 接受 `lhsT=A^T` 和 `rhs=B` 作为参数，并返回 `A × B`。
  - 分块维度：
      - 分块大小基于张量引擎矩阵乘法运算的约束设置，如[此处](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.isa.nc_matmul.html)所述。
        - `TILE_M`：128——`M` 维度的分块大小。
        - `TILE_K`：128——`K` 维度的分块大小。
        - `TILE_N`：512——`N` 维度的分块大小。
  - 遍历分块：
      - 内核使用 `affine_range` 循环沿 `result` 矩阵的 `M` 和 `N` 维度迭代分块。
      - 对于每个形状为 `(TILE_M, TILE_N)` 的输出分块，它在 PSUM 内存中分配一个临时的部分和张量 `res_psum`。
  - 加载分块：
      - 对于每个输出分块，将 `lhsT` 和 `rhs` 的分块加载到片上 SBUF 内存中以实现高效访问。
      - `lhsT_tile` 加载形状为 `[TILE_K, TILE_M]` 的切片，`rhs_tile` 加载形状为 `[TILE_K, TILE_N]` 的切片。
  - 矩阵乘法：
      - 使用加载的分块执行部分矩阵乘法，并将部分结果累积到 `res_psum` 中。
  - 存储结果：
      - 一旦给定结果块的分块完全计算完毕，`res_psum` 中的部分和被复制到 SBUF 并转换为所需的数据类型。
      - 最终结果存储回 `result` 张量的相应位置。

> 注意，我们将在线教程中的 `nl.matmul()` 和 `nl.load()/nl.store()` 替换为了 `nisa.nc_matmul()` 和 `nisa.dma_copy()`。这将 nki.lang API 降低到 nki.nisa。我们建议对任何计算指令使用 nki.isa API。这在其如何被降低方面具有更确定的行为，并且更少出现可能导致虚假编译错误的意外行为。

总之，这种分块实现通过将其分解为硬件兼容的分块大小来处理大型矩阵维度。它利用专用内存缓冲区（即 PSUM）来最小化内存延迟并优化矩阵乘法性能。你可以在[此处](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/tutorials/matrix_multiplication.html)阅读更多关于 NKI 矩阵乘法的信息。

### 卷积层概述

现在让我们将焦点转向卷积层。回想一下课堂上讨论的[卷积运算](https://gfxcourses.stanford.edu/cs149/fall25/lecture/dnninference/slide_26)。它涉及在**输入特征图**上滑动一个滤波器，在每个位置滤波器与重叠的输入区域交互。在每个重叠区域中，在滤波器权重和输入区域值之间执行逐元素乘法。这些逐元素乘法的结果随后被相加，在输出特征图中对应位置产生一个单一值。这个过程捕获了相邻特征之间的局部空间模式和关系。

<p align="center">
  <img src="handout/convolution.png" width=55% height=55%>
</p>

输入特征图通常包含多个通道。例如，图像通常包含三个 RGB 通道（红、绿、蓝）。在这种情况下，卷积不是仅在二维空间区域上计算加权和，而是计算二维空间区域和通道深度的加权和。下图描绘了在具有三个 RGB 通道的 32x32 输入图像上执行卷积层的示例。在图像中，一个 5x5x3 的滤波器应用于 32x32x3 的图像，生成一个 28x28x1 的输出特征图。

<p align="center">
  <img src="handout/cs231n_convolution.png" width=55% height=55%>
  <br>
  <em>来源：CS231N https://cs231n.stanford.edu/slides/2025/lecture_5.pdf</em>
</p>

**如图所示，每个滤波器产生一个单一输出通道。** 要生成多个输出通道，需要将多个滤波器应用于输入特征图。此外，每个卷积滤波器还包含一个标量偏置值，该值将添加到每个加权和中。

卷积算子的输入和输出可以总结如下（暂时忽略偏置）：

<p align="left">
  <img src="handout/conv2d_summary.png" width=50% height=50%>
</p>

此外，[卷积层](https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html)除输入特征图、滤波器权重和标量偏置外，还可以接受额外的超参数，如填充和步长。然而，我们**简化了卷积的约束**以使你的实现更容易。你只需**支持步长为 1**，并且**不必担心填充**，因为我们在将输入特征图传递到你的内核之前会为你填充。

### 将卷积映射到矩阵乘法

现在，我们的目标是将卷积算子映射到 Trainium 张量引擎支持的高性能矩阵运算上。为此，我们可以将卷积的数学公式与矩阵乘法进行比较。

**Conv2D：**

<p align="center">
  <img src="handout/conv2d_formula.png" width=65% height=65%>
</p>

**矩阵乘法：**

<p align="center">
  <img src="handout/matmul_formula.png" width=25% height=25%>
</p>

在课堂上，我们讨论了一种将多滤波器卷积转换为单个大型矩阵乘法的方法。我们这里将做同样的事情，但采用一种在 Trainium 上产生高效实现的不同的方法。在这种方法中，卷积运算被表述为一系列独立的矩阵乘法。下面展示了这种表述的视觉图示。

> [!NOTE]
> **这是一种不同于讲座中描述的那种为每个空间块创建单独行的 conv -> matmul 归约。**

<p align="center">
  <img src="handout/conv2d_matmul_diagram.png" width=100% height=100%>
</p>

在这种方法中，输入特征图的高度和宽度维度被展平为一个单一维度，将输入重塑为 `(高度 × 宽度) × 输入通道`。然后将此重塑后的输入乘以滤波器的每个位置，其中 `i` 和 `j` 分别从 `0` 到 `滤波器高度 - 1` 和从 `0` 到 `滤波器宽度 - 1` 范围。每个滤波器切片具有 `输入通道 × 输出通道` 的形状，生成的矩阵乘法沿 `输入通道` 维度压缩。为了将输入与每个滤波器切片对齐，必须将输入偏移对应于滤波器当前位置 `(i, j)` 的量。这些矩阵乘法的结果被累积以产生输出张量。

以下是所描述算法的伪代码：
```
- 输入图像形状为 (输入通道, 图像高度 * 图像宽度)
- 滤波器权重形状为 (滤波器高度, 滤波器宽度, 输入通道, 输出通道)
- 将输出初始化为适当的形状 (输出通道, 输出高度 * 输出宽度)

# 遍历滤波器高度
for i in range(过滤器_高度):
    # 遍历滤波器宽度
    for j in range(过滤器_宽度):

        # 将输入张量偏移 (i, j) 以与滤波器的当前位置对齐
        input_shifted = shift(input, (i, j))

        # 执行输入与滤波器切片之间的矩阵乘法
        # 注意这是一个完整的矩阵乘法，没有输入大小限制
        output += matmul(transpose(weight[i,j,:,:]), input_shifted)
```

> [!NOTE]
> **这只是一个算法描述，本作业的目的是让你弄清楚如何将此算法描述映射到该硬件上的高效实现！**

### 最大池化层概述
最大池化层通常用于 CNN 中连续的卷积层之间，以减小特征图的大小。这不仅防止了可能对计算资源造成问题的过大的特征图，而且减少了 CNN 中的参数数量，从而有效地减少了模型过拟合。

最大池化层的运作方式类似于卷积层，它在输入特征图上滑动一个空间滤波器。然而，不是为每个重叠区域计算加权和，最大池化层从每个区域中选择最大值并将其存储在输出特征图中。此操作独立地应用于特征图的每个通道，因此通道数量保持不变。例如，考虑一个具有三个 RGB 通道的 4x4 输入图像通过一个具有 2x2 滤波器的最大池化层。结果输出是一个具有三个 RGB 通道的 2x2 图像，表明空间维度减少了 2 倍，而通道数量保持不变。

<p align="center">
  <img src="handout/maxpool.png" width=37% height=37%>
</p>

如上所示，[最大池化层](https://pytorch.org/docs/stable/generated/torch.nn.functional.max_pool2d.html#torch.nn.functional.max_pool2d)通常具有单独的步长和滤波器大小超参数。与卷积层类似，我们简化了你需要实现的最大池化层的约束。你的内核不需要定义这两个参数，而是使用一个单一参数 `pool_size`，它同时对应滤波器大小和步长。`pool_size` 只能设置为 1 或 2。当 `pool_size` 为 2 时，最大池化操作如上图所示。当 `pool_size` 为 1 时，最大池化层充当即无操作（no-op），产生与输入相同的输出。虽然 `pool_size` 为 1 可能看起来毫无意义，但它实际上为你的融合层提供了额外的灵活性，你很快就会看到。

### 融合卷积与最大池化
你将实现一个 NKI 内核，将卷积层和最大池化层组合成一个单一的融合操作。下面，我们将概述你的融合层的详细规范和要求。

<p align="center">
  <img src="handout/fused_kernel.png" width=95% height=95%>
</p>

上图展示了你的融合内核对具有单个输入通道的 6x6 输入将执行的计算。融合内核执行一个标准卷积，使用一个滤波器和步长 1。然后融合内核对卷积结果执行最大池化，使用 2x2 池化滤波器。

你的融合内核接受以下参数：
  - `X`——一批输入图像。`X` 形状为 `(批量大小, 输入通道, 输入高度, 输入宽度)`。保证 `输入通道` 是 128 的倍数。
  - `W`——卷积滤波器权重。`W` 形状为 `(输出通道, 输入通道, 滤波器高度, 滤波器宽度)`。保证 `滤波器高度 == 滤波器宽度`。也保证 `输出通道` 是 128 的倍数。此外，你可以假设权重的大小始终可以完全放入 SBUF。
  - `bias`——卷积滤波器偏置。`bias` 形状为 `(输出通道)`
  - `pool_size`——最大池化滤波器的大小和池化步长。保证输入大小、滤波器大小和 `pool_size` 使得一切都能很好地整除。更具体地说，`(输入高度 - 滤波器高度 + 1) % 池化大小 == 0`。请注意，如果 `pool_size` 的值为 `1`，那么融合内核将作为普通卷积内核运行。这为我们提供了选择是否想要最大池化的灵活性。

可以随意使用关于卷积层实现的[课程幻灯片](https://gfxcourses.stanford.edu/cs149/fall25/lecture/dnninference/slide_57)作为起点。如果你参考课程幻灯片，在我们的命名方案中，`INPUT_DEPTH` 等同于 `输入通道`，`LAYER_NUM_FILTERS` 等同于 `输出通道`。请注意，你的融合内核的输入参数具有与卷积课程幻灯片中描绘的不同的形状。你可以自由地使用 [NumPy reshape 方法](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html)将输入重塑为你想要的任何形状，就像在第 1 部分的 `vector_add_stream` 内核中所做的那样。我们还在 `part2/conv2d_numpy.py` 中为你提供了卷积层和最大池化层的 NumPy 实现。NumPy 实现应该为你提供每层编程逻辑的大致轮廓。思考如何能将 NumPy 实现融合到单个层中可能是一个好的练习，这正是你在内核中将要做的。可以随意查看 [NKI 教程](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/tutorials.html)以了解有关其他优化或其他 API 函数的更多信息。你也可以查看 [NKI API 参考手册](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/index.html)以查看所有可用的 API 函数及其用法。你可能会发现其中一些很有用。*提示：* [nisa.tensor_reduce(nl.max, ...)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.tensor_reduce.html) 对最大池化应该有帮助。[nisa.tensor_tensor](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.tensor_tensor.html) 对添加偏置应该有帮助。

### 你需要做什么
对于本作业的这部分，专注于文件 `part2/conv2d.py`。我们提供了基本的起始代码；你的任务是完成函数 `fused_conv2d_maxpool` 中（融合）Conv2D 内核的实现。

#### 一般技巧
* **优先考虑正确性。** 我们建议从最简单的情况开始：小图像、无偏置、无最大池化。一旦你的内核对小图像有效，扩展其功能以处理太大而无法完全放入 SBUF 缓冲区的图像。之后，加入偏置加法，然后将最大池化操作融合到你的内核中。一旦你有了完全正确的解决方案，开始优化性能/加分。
  * 测试框架将按从易到难的顺序在测试用例上运行你的内核。此外，你可以选择在省略最大池化测试用例的情况下运行测试框架，如果你选择在有了高性能实现之后再处理融合最大池化和 conv2d 内核的话。
* **彻底理解你的算法。** 在考虑任何分块策略之前，确保你对上述算法所需的矩阵运算（乘法、偏移、加法）有扎实的理解。然后，画出矩阵及其维度，并思考如何将它们映射到硬件上，特别是关于内存层次结构。
  * 你可能还需要预处理输入数组（例如，重塑或转置它们）以实现更高效的访问。提示：如果你想知道为什么可能需要转置，请考虑 NKI 矩阵乘法接口的独特之处——第一个输入矩阵是转置的。
* **跟踪分块维度。** 由于你无法一次计算整个输出，你必须考虑将哪个输出维度分解为分块。回想一下 SBUF 分块的约束——分区维度最多为 128，并且必须是张量的第一个维度。一旦你决定了输出形状，这对你的输入意味着什么？换句话说，计算单个输出分块需要 `X` 和 `W` 的什么子集？
* **在保持数据局部性的同时安排循环顺序。** 你需要的 `for` 循环来自多个来源：算法定义的滤波器高度和宽度、分块矩阵乘法以及批处理。
  * 在识别这些循环后，建议的目标是将它们排列成使中间结果保留在 `PSUM` 中，直到每个分块的计算完全完成。这确保结果数组在 `SBUF` 中的每个部分只写入一次，提高输出数据局部性——尽管其他方法可能达到可比的性能。
  * 一旦这到位，排列剩余的循环以优化输入数据局部性。如果你不确定，尝试不同的数据访问模式以找到效果最好的，并思考为什么！
* **使用分析工具指导性能调优。** 一旦你有了一个可工作的内核，你很可能需要进一步调优性能以获得满分/加分。分析工具是你的朋友：寻找张量引擎空闲和利用率低的大间隙/阶段，并尝试重构你的代码以最小化在这些部分花费的时间。
  * 回顾第 1 部分也可能有帮助，在那里我们优化了一个简单的向量加法内核（以及转置内核，如果你尝试了加分题的话）。

#### 测试
使用提供的测试框架脚本验证你的实现。要运行测试，导航到 `part2/` 目录并执行：
```
python3 test_harness.py
```

要检查你带有融合最大池化的 Conv2D 内核实现的正确性和性能，使用 `--test_maxpool` 标志调用测试框架。

测试框架将首先运行正确性测试，然后运行性能检查。满分方案必须在保持正确性的同时达到参考内核 120% 以内的性能。它将使用 float32 和 float16 数据类型的输入张量调用你的内核，其中 float16 的性能要求更严格。确保你在编写内核时牢记这一点！

请注意，你的内核将在**没有** `--profile`（这会略微改变执行时间）的情况下测试性能，以与性能阈值的设置方式保持一致。

#### 书面报告与分析
学生需要提交一份书面报告，简要描述他们的实现。也描述你是如何进行优化你的实现的。确保对你的实现进行分析，并报告在 `float16` 和 `float32` 数据类型下达到的 MFU（模型 FLOPs 利用率）。你可以通过使用 `--profile <profile_name>` 标志运行测试框架来捕获跟踪，然后运行：
```
neuron-profile view -n [profile_name].neff -s [profile_name].ntff
```

> [!TIP]
> 当你打开分析工具时，你可能看到一些关于缺失基准测试参数的警告。你唯一需要在这里提交的参数是 MFU 值，它仍然可以通过将鼠标悬停在 GUI 中 Estimated MFU 部分的 "Cumulative Utilization" 线上获得，如下所示。（确保取最后面的 MFU。）

<p align="center">
  <img src="handout/mfu.png" alt="Profiler warning" width="90%">
</p>

### 使用 NKI 的技巧
* 在以下情况下优先使用 nki.isa API：
    * 所有计算操作
      * 使用 nisa.nc_matmul 代替 nl.matmul
      * 使用 nisa.tensor_scalar(op=nl.add, <>) 代替 nl.add
    * 优先使用 nisa.dma_copy() 代替 nl.load()/nl.store()。
    * 在调用 nisa 计算操作时，确保只传递 op=nl.* 代码作为参数。例如，不要传递 op=math.sin。
* 避免使用嵌套函数。在模块级别定义所有函数。
* 要调试你的实现，你可以使用 `--simulate` 标志运行测试框架。这会用 `nki.simulate_kernel()` 调用包装你的实现：你可以在[此处](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.simulate_kernel.html#nki.simulate_kernel)阅读更多。在模拟模式下运行时，你可以在内核中插入 `nl.device_print(str, tensor)` 来打印设备张量的中间值。然而，CPU 模拟和设备端执行之间**可能存在**一些差异。如果你对结果不确定，建议通过返回中间张量来调试。
* 在修改变量赋值时要小心。一些 nisa API 将目标张量作为参数，例如 nisa.dma_copy(src=<>, dst=<>)。其他 API 通过函数本身产生目标张量，可能需要用来修改现有张量。在未来的 NKI 版本中，所有 ISA API 都将接受 dst 作为参数。例如：
  * x_sbuf = nl.zeros(shape=hbm_tensor.shape, buffer=nl.sbuf)（创建数组）
  * nisa.dma_copy(src=hbm_tensor, dst=x_sbuf)（复制到数组中）
  * 具体来说，如果你选择使用 `nl.load(...)`，`x = nl.load(...)`（创建新数组）与 `x[...] = nl.load(...)`（修改现有数组）是不同的。
* 避免使用 block dimension，它是一个纯软件结构，不影响硬件。（如果你不知道它是什么，不用担心。）要么将其放入自由维度，要么使用张量列表。参见公共[文档](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.0/general/nki/nki_block_dimension_migration_guide.html#nki-block-dimension-migration-guide)。
* 对于张量索引，优先使用整数切片。当需要更高级的索引时，使用 [`nl.mgrid`](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.language.mgrid.html)。不要使用嵌套切片/mgrid（例如 t[0:128, 128:256][0:64, 0:64]）。不要使用 nl.arrange()。


## 额外加分
在更小的图像上再次运行 `neuron-profile`。较小和较大图像之间的 MFU 是否存在差异？如果是，你将如何为较小的图像优化你的融合卷积层？（了解 `nisa.nc_matmul` 可以接受 >2D 张量作为 `moving` 参数可能会有所帮助，只要 PSUM 的硬件约束得到遵守。）

最多五分额外加分将授予在小图像上达到性能目标（更严格的目标）的解决方案。你的书面报告必须清楚地解释你的方法以及你优化解决方案所采取的步骤。

## 评分指南

对于正确性测试，我们使用两种类型的图像。第一种是小图像，尺寸为 32x16。第二种是大图像，尺寸为 224x224，超出 SBUF 的容量，不能一次性全部放入。你的代码必须通过所有正确性测试才能获得性能分。

对于性能测试，我们评估你的内核在不同配置下相对于参考内核的性能：有无最大池化，使用 float16 和 float32 精度。

作为中间目标，我们包含了来自参考内核未优化版本的放宽延迟。如果你的 p99 延迟在放宽延迟的 120% 以内，你将获得 95% 的性能分。如果在优化参考延迟的 120% 以内，你将获得满性能分。

加分部分只有一个性能阈值，即参考延迟的 120%。

**书面报告：30 分**
  - 第 1 部分问题：20 分
  - 第 2 部分问题：10 分

**矩阵转置内核正确性：10 分（+1 分性能加分）**

**融合卷积 - 最大池化内核正确性：10 分**
  - 小图像：2.5 分
  - 大图像：2.5 分
  - 带偏置加法：2.5 分
  - 带最大池化：2.5 分

**融合卷积 - 最大池化内核性能：50 分（+5 分加分）**
  - 无最大池化 (float16)：17.5 分
  - 无最大池化 (float32)：17.5 分
  - 有最大池化 (float16)：7.5 分
  - 有最大池化 (float32)：7.5 分
  - 小图像无最大池化 (float16)：1.25 分（加分）
  - 小图像无最大池化 (float32)：1.25 分（加分）
  - 小图像有最大池化 (float16)：1.25 分（加分）
  - 小图像有最大池化 (float32)：1.25 分（加分）

## 提交说明

请通过 Gradescope 提交你的工作。如果你与伙伴合作，请记得在 Gradescope 上标记你的伙伴。

1. **请将你的书面报告提交为 `writeup.pdf` 文件。**
2. **请运行 `sh create_submission.sh` 生成 `asst4.tar.gz` 以提交到 Gradescope。** 如果脚本报错说 'Permission denied'，你应该运行 `chmod +x create_submission.sh`，然后重新运行脚本。还请仔细检查生成的 `tar.gz` 是否包括：
  * 包含你第 1 部分转置内核的 `kernels.py` 文件。
  * 包含你第 2 部分融合 Conv2D 内核的 `conv2d.py` 文件。
