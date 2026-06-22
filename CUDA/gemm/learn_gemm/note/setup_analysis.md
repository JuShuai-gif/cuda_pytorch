# setup.sh 脚本分析

## 概述

`setup.sh` 是一个 Bash 脚本，用于设置 **explore-gemm** 项目的开发环境。它主要完成以下任务：

- 检测并配置 Python 环境（conda / virtualenv）
- 验证 PyTorch 安装及其 C++ 组件（libtorch）
- 下载并配置 CUTLASS 库（用于 GPU 矩阵乘法优化）
- 下载 Catch2 测试框架头文件
- 安装额外的 Python 依赖包
- 创建符号链接，方便 CMake 找到 libtorch

## 使用方法

```bash
./setup.sh [OPTIONS]
```

### 选项

| 选项 | 说明 |
|------|------|
| `-t, --cutlass VERSION` | 指定 CUTLASS 版本（默认：4.3.0） |
| `-h, --help` | 显示帮助信息 |

### 示例

```bash
./setup.sh                    # 使用默认配置（CUTLASS 4.3.0）
./setup.sh -t 4.2.0          # 使用 CUTLASS 4.2.0
```

## 脚本执行流程

### 1. 环境检测与准备
- 检查 Python 是否可用
- 识别当前 Python 环境类型（conda / virtualenv / 系统 Python）
- 如果未检测到任何虚拟环境，提供创建虚拟环境的选项
  - 在 Debian/Ubuntu 上自动安装 `python3-venv` 和 `python3-dev`
  - 创建 `venv` 目录并激活环境
  - 安装 PyTorch（CUDA 12.8 版本）

### 2. PyTorch 验证
- 检查 PyTorch 是否已安装
- 获取 PyTorch 版本、安装路径和 CUDA 版本信息
- 验证 PyTorch C++ 组件（`include/`、`lib/`、`share/cmake/`）是否完整

### 3. 创建 libtorch 符号链接
- 在 `third-party/libtorch` 创建指向 PyTorch 安装路径的符号链接
- 如果已存在，提供覆盖选项

### 4. 下载 Catch2 头文件
- 从 GitHub 下载 Catch2 v2.13.10 的单头文件版本
- 保存到 `third-party/catch.hpp`

### 5. 下载 CUTLASS 库
- 根据指定版本从 GitHub Releases 下载 ZIP 包
- 解压到 `third-party/cutlass`
- 如果已存在，提供重新下载选项

### 6. 安装额外 Python 包
- 使用 pip 安装以下包：
  - `loguru`：日志记录
  - `pandas`：数据分析
  - `plotly`：可视化
  - `pytest`：测试框架
  - `click`：命令行工具
  - `ninja`：构建系统

### 7. 输出总结与下一步指引
- 显示 libtorch 符号链接位置
- 显示 PyTorch 版本和 CUDA 版本
- 显示 Catch2 和 CUTLASS 路径
- 提供 CMake 构建和运行测试的命令

## 关键功能详解

### 环境检测逻辑
```bash
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    # conda 环境
elif [ -n "$VIRTUAL_ENV" ]; then
    # virtualenv 环境
else
    # 无虚拟环境，提示创建
fi
```

### PyTorch C++ 组件检查
脚本检查以下目录和文件：
- `$PYTORCH_PATH/lib`：库文件目录
- `$PYTORCH_PATH/include`：头文件目录
- `$PYTORCH_PATH/lib/libtorch.so` 或 `libtorch.dylib`：动态库
- `$PYTORCH_PATH/share/cmake`：CMake 配置文件

### 依赖包自动安装
对于 Debian/Ubuntu 系统，如果缺少 `unzip` 或 `cmake`，脚本会使用 `sudo apt install` 自动安装。

## 注意事项

### 1. 权限要求
- 安装系统包时需要 sudo 权限
- 如果使用 conda base 环境，脚本会发出警告

### 2. 网络依赖
- 需要互联网连接以下载 CUTLASS 和 Catch2
- PyTorch 安装也从官方仓库下载

### 3. 环境激活
- 如果脚本创建了虚拟环境（venv），后续使用项目时需要手动激活：
  ```bash
  source venv/bin/activate
  ```

### 4. 兼容性
- 主要针对 Linux 系统（Debian/Ubuntu）开发
- 假设使用 CUDA 12.8 版本的 PyTorch
- 其他 Linux 发行版可能需要调整包管理命令

### 5. 错误处理
- 使用 `set -e` 在出错时立即退出
- 关键步骤都有错误检查和友好的提示信息

## 目录结构变化

运行脚本后，项目目录结构如下：

```
project-root/
├── third-party/
│   ├── libtorch -> /path/to/torch/installation (symbolic link)
│   ├── catch.hpp (Catch2 header)
│   └── cutlass/ (CUTLASS library)
├── venv/ (optional, if created)
└── ... (other project files)
```

## 常见问题

### Q: 脚本卡在下载阶段
A: 检查网络连接，或手动下载 CUTLASS ZIP 包放到 `third-party/` 目录。

### Q: PyTorch C++ 组件缺失
A: 使用 conda 重新安装 PyTorch（包含完整 C++ 库），或使用 pip 强制重装。

### Q: 权限被拒绝（sudo 失败）
A: 确保当前用户在 sudoers 列表中，或手动安装缺失的包。

### Q: 虚拟环境创建失败
A: 检查系统是否安装了 `python3-venv` 和 `python3-dev` 包。

## 扩展与自定义

### 修改默认版本
编辑脚本开头的变量：
```bash
CUTLASS_VERSION="4.3.0"
```

### 添加其他依赖
在“安装额外 Python 包”部分添加需要的包名。

### 支持其他系统
在包安装部分添加对其他包管理器（如 yum、pacman）的支持。

---

*最后更新：2025-03-28*  
*脚本路径：`/home/ghr/cuda_pytorch/gemm/learn_gemm/setup.sh`*