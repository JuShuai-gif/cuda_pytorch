# PyTorch 源码调试指南

本目录用于调试学习 PyTorch 源码。基于自行编译的 **DEBUG 版**(带调试符号),
配合 VSCode 实现 **Python 层** 与 **C++/CUDA 层** 的断点调试。

## 环境前提

- conda 环境:`torch_env`(已安装 debug 版 `torch 2.14.0a0`)
- PyTorch 源码:`/home/ghr/code/pytorch`(编译时 `DEBUG=1`,含调试符号)
- 调试配置:`.vscode/launch.json`(已就绪)

VSCode 需安装扩展:**Python**、**C/C++ (ms-vscode.cpptools)**。

启动 VSCode 前先激活环境,或在 VSCode 里选择解释器
`/home/ghr/miniconda3/envs/torch_env/bin/python`。

```bash
conda activate torch_env
code /home/ghr/code/learn_torch_source_code
```

---

## 示例文件

| 文件                 | 用途                                       |
| -------------------- | ------------------------------------------ |
| `01_python_debug.py` | Python 层调试:步入 `torch/nn`、autograd 等 |
| `02_cpp_debug.py`    | C++/CUDA 层调试:gdb 步入 ATen 算子内核     |

---

## 一、Python 层调试(`01_python_debug.py`)

目标:从自己的代码单步进入 PyTorch 的 **Python 源码**。
关键在于 `launch.json` 里的 `"justMyCode": false`,否则 F11 不会跳进库代码。

### 步骤

1. 打开 `01_python_debug.py`。
2. 在标注 `# <-- BREAKPOINT` 的 `y = model(x)` 行左侧点击设断点(红点)。
3. 按 `F5`,选择调试配置 **"Python: Debug Current File"**。
4. 程序在断点暂停后:
   - **F11 (Step Into)** 进入 `model(x)` → 会跳进
     `torch/nn/modules/module.py` 的 `_call_impl` → `forward`
     → `torch/nn/modules/linear.py` 的 `F.linear`。
   - 继续在 `loss.backward()` 上 F11 → 进入
     `torch/autograd/__init__.py`。
5. 常用操作:F10 单步跳过、F11 单步进入、Shift+F11 跳出、F5 继续。

### 观察点

- `模型 __call__` 如何经 hook 机制走到 `forward`(见 `_call_impl`)。
- `nn.Linear.forward` 最终调用 `torch.nn.functional.linear`,
  再往下就进入 C++(用第二个示例追)。

---

## 二、C++ / CUDA 层调试(`02_cpp_debug.py`)

目标:在 `torch.add` 执行时,用 **gdb** 断在 PyTorch 的 C++ 内核里。
原理:先用 Python 启动进程并 **暂停等待**,再用 gdb **attach** 到该进程,
下好 C++ 断点后继续执行,即可命中。

### 步骤

1. 打开 VSCode 集成终端,运行脚本(注意用 torch_env 的 python):

   ```bash
   conda activate torch_env
   cd /home/ghr/code/learn_torch_source_code/A_setup_debug
   python 02_cpp_debug.py
   ```

   脚本会打印 `PID = xxxxx` 并停在 `input()` 等待。

2. 在 VSCode 里按 `F5`,选择调试配置 **"(gdb) Attach to Python"**,
   在弹出的进程列表里选中上一步打印的 **PID**(可搜 PID 数字)。

3. attach 成功后,gdb 会暂停该进程。打开 **调试控制台 (Debug Console)**,
   输入 GDB 命令下断点(cpptools 里用 `-exec` 前缀执行原生 gdb 命令):

   ```
   -exec b at::TensorIteratorBase::build
   -exec continue
   ```

   `at::TensorIteratorBase::build` 是所有逐元素算子(含 add)都会走的函数,
   命中最稳。

4. 回到运行脚本的终端,按 **ENTER** 放行 `input()`。
   进程执行到 `a + b` 时会命中断点,VSCode 停在
   `aten/src/ATen/TensorIterator.cpp`。

5. 此时可单步(F10/F11)、查看调用栈(Call Stack)、打印变量:

   ```
   -exec bt              # 打印完整 C++ 调用栈
   -exec p iter          # 查看 TensorIterator
   ```

### 更深入的断点(可选)

| GDB 断点                                      | 位置说明                              |
| --------------------------------------------- | ------------------------------------- |
| `b at::native::add_stub`                      | add 的 dispatch stub(CPU/CUDA 分派)  |
| `b at::TensorIteratorBase::build`             | 建立迭代器(广播/遍历),必经         |
| 文件断点 `aten/src/ATen/native/ufunc/add.h:16` | 标量内核 `self + alpha * other`(真加法)|

在 gdb 里用 Tab 补全 `at::native::structured_ufunc_add_CPU` 可断到 CPU 的
add 结构化实现(位于匿名命名空间,建议用文件行号断点更直接)。

### CUDA 内核调试(进阶)

CPU 断点方式对 CUDA 路径的 **host 端**(dispatch、TensorIterator 构建)同样有效。
若要断在 **device 端 kernel**(GPU 上执行的代码),需改用 `cuda-gdb`:

```bash
cuda-gdb -p <PID>
(cuda-gdb) b at::native::add_stub
```

普通 gdb 无法在 `__global__` 设备函数内停下,这是学习 CUDA 内核时的关键区别。

---

## 调试原理小结

```
自己的 .py 代码
   │  (debugpy, justMyCode=false)  ← 示例一在这一层
   ▼
torch/*.py  (nn / autograd / functional)
   │  pybind11 绑定,跨入 C++
   ▼
torch/csrc + aten + c10  (C++/CUDA)   ← 示例二用 gdb attach 在这一层
```

- **Python 断点**靠 `debugpy`(launch 配置 `type: debugpy`)。
- **C++ 断点**靠 `gdb` attach 到同一个 python 进程(配置 `type: cppdbg`)。
- 两者可 **同时** 存在:先用示例一的方式启动并在 Python 断点停住,
  再用示例二的方式 attach gdb,即可实现 Python + C++ 混合调试。
- 能断进 C++ 的前提是 PyTorch 用 `DEBUG=1` 编译(保留调试符号)。

## 常见问题

- **F11 跳不进 torch 库代码**:检查 `launch.json` 的 `"justMyCode": false`。
- **gdb attach 失败/无符号**:确认调的是 `torch_env` 的 python,且 torch 为 debug 版
  (`python -c "import torch; print(torch.version.debug)"` 应为 `True`)。
- **找不到进程**:`02_cpp_debug.py` 打印的 PID 要和进程列表里选的一致。
- **ptrace 权限报错**:临时执行 `echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope`。
