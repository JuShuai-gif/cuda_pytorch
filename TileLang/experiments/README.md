# 实验目录（experiments）

> 本目录存放 TileLang 学习实验。每个实验 = 独立子目录（含脚本 + 运行记录）。实验需要 GPU 与编译环境（见 `12_编译与安装指南.md`）。

## 编号说明

- `01`-`12`：原计划的 12 个核心实验（与文档编号大体对应）。
- `09`-`22`：与文档编号对应、按学习过程陆续新增的实验（含多个实现示例、调试、性能、生成代码分析）。

## 实验清单

| 目录 | 内容 | 对应文档 | 状态 |
| --- | --- | --- | --- |
| `01_vector_add` | 向量加法（入门） | 07, 09 | 模板待填充 |
| `02_elementwise` | elementwise 算子族 | 09 | 模板待填充 |
| `03_gemm` | GEMM 基础实现 | 14, 22 | 模板待填充 |
| `04_softmax` | softmax（fragment） | 09 | 模板待填充 |
| `05_flash_attention` | FlashAttention | 14 | 模板待填充 |
| `06_autotune` | autotune 搜索 | 11 | 模板待填充 |
| `07_pipeline` | 软件流水线 | 17 | 模板待填充 |
| `08_ir_dump` | dump IR 观察 pass | 08, 16 | 已完成（模板+记录） |
| `09_dsl` | DSL 原语练习 | 09 | 模板待填充 |
| `10_jit` | JIT 缓存实验 | 10 | 模板待填充 |
| `11_autotune` | autotune 调优 | 11 | 模板待填充 |
| `12_cuda_comparison` | torch/tilelang/triton/cuda 对比 | 18 | 模板待填充 |
| `14_examples` | 示例研读 | 14 | 模板待填充 |
| `16_debug` | 调试实验 | 16 | 模板待填充 |
| `17_perf` | 性能调优 | 17 | 模板待填充 |
| `18_triton_compare` | Triton 对比 | 18 | 模板待填充 |
| `19_cuda_map` | CUDA 映射 | 19 | 模板待填充 |
| `20_ds` | 数据结构 | 20 | 模板待填充 |
| `21_trace` | 调用链追踪 | 21 | 模板待填充 |
| `22_gencode` | 生成代码分析 | 22 | 模板待填充 |

## 通用运行方式

```bash
# 每个子目录的脚本以 python 直接运行（需先编译安装 tilelang）
python <子目录>/xxx.py

# 记录结果：把运行输出保存为 <子目录>/RUN.log
python <子目录>/xxx.py > <子目录>/RUN.log 2>&1
```

## 完成规则

1. 每次实验必须能跑通（有输出、有校验通过）。
2. 记录：运行命令、关键输出、遇到的错误与解法 → 写进该目录 `README.md` 或 `RUN.log`。
3. 实验完成后回到 `项目分析状态.md` 更新对应状态。
