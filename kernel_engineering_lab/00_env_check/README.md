# 00_env_check - 环境验证

本模块确保在开始任何 kernel 工作之前，GPU 开发环境已正确配置。

## 目的

GPU kernel 开发需要以下特定技术栈：

- **NVIDIA 驱动**，支持 CUDA
- **CUDA Toolkit** (nvcc)，用于编译 CUDA C++ 扩展
- **PyTorch**，需以 CUDA 支持构建（非 CPU-only 版本）
- **Triton**，用于 Python 原生 GPU kernel 编程

这些组件之间的版本不匹配是"无法运行"问题的头号原因。

## 常见问题

### CUDA 驱动与 PyTorch CUDA 版本不匹配
PyTorch 自带其 CUDA 库，但内核驱动必须支持它们。
例如，如果 PyTorch 使用 CUDA 12.1 但驱动仅支持 CUDA 11.x，PyTorch 将回退到 CPU。

**解决方法**：确保驱动版本 >= PyTorch 使用的 CUDA toolkit 版本。

### PyTorch CPU-Only 构建
尽管 nvidia-smi 显示有 GPU，`torch.cuda.is_available()` 仍返回 `False`。

**解决方法**：安装 CUDA 版本：`pip install torch --index-url https://download.pytorch.org/whl/cu121`

### Triton 未安装或版本错误
Triton 需要兼容的 PyTorch 版本。Triton 2.1+ 适用于 PyTorch 2.1+。

**解决方法**：`pip install triton>=2.1.0`

### nvcc 不在 PATH 中
CUDA 编译需要 `nvcc`。即使 CUDA 已安装，PATH 可能缺失。

**解决方法**：将 `/usr/local/cuda/bin` 添加到 PATH，或使用完整路径。

## 用法

```bash
# 检查环境
python 00_env_check/check_env.py

# 运行验证测试
pytest 00_env_check/test_env.py -v
```
