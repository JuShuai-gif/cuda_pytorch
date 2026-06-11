# src 示例说明

这个目录用两个文件说明 PDF 中的 shared memory bank conflict 和 `LDS.64` / `LDS.128` 访存行为。

## 文件

- `shared_memory_bank_examples.cu`：CUDA 示例 kernel，包含连续 `uint2`、合并 `uint2`、连续 `uint4`、合并 `uint4`、以及带 bank conflict 的 `uint4` 模式。
- `bank_transaction_sim.py`：纯 Python 模拟器，不需要 CUDA，用来打印每个线程访问的 word、bank，以及粗略 transaction 分组。

## 运行 Python 模拟器

```bash
python bank_transaction_sim.py
```

输出会展示：

- `uint2_contiguous`：连续 `uint2` 访问，默认 2 个 half-warp transaction。
- `uint2_pair_broadcast`：相邻线程读同一个 `uint2`，满足 `i xor 1` 合并条件。
- `uint4_contiguous`：连续 `uint4` 访问，默认 4 个 quarter-warp transaction。
- `uint4_pair_merge`：`uint4` 访问满足合并条件，可降到每个 half warp 1 次。
- `uint4_conflict_like_pdf`：对应 PDF 中会出现 2-way bank conflict 的地址模式。

## 编译 CUDA 示例

需要本机安装 CUDA Toolkit。

```bash
nvcc -O3 -lineinfo -arch=sm_80 shared_memory_bank_examples.cu -o shared_memory_bank_examples
```

`sm_80` 适合 A100。其他 GPU 请替换为对应架构，例如 `sm_86`、`sm_89`、`sm_90`。

## 运行 CUDA 示例

```bash
./shared_memory_bank_examples
```

Windows PowerShell 下：

```powershell
.\shared_memory_bank_examples.exe
```

程序主要用于触发 kernel，方便配合 Nsight Compute 看 shared memory 指标。

## 用 Nsight Compute 观察

可以用下面两个指标对照 PDF：

```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum ./shared_memory_bank_examples
```

关注：

- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`：shared memory load 的 bank conflict 数量。
- `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum`：shared memory load transaction / wavefront 数量。

不同 GPU 架构、编译器版本、优化级别可能让指标细节略有差异。这个示例的重点是帮助对照访存模式，而不是构造完整性能 benchmark。

