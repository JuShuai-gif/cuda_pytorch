好的，我帮你把上面的内容翻译成中文，并尽量保持原文的技术细节和逻辑清晰度：

---

# 第16章

## 代码

为了学习目的，我们实现了本章中描述的几种技术。具体来说，我们实现了反向传播（backward pass）和池化（pooling）的顺序实现，更重要的是，我们实现了一个最小化版本的自动求导（autograd）系统。

### 手动实现 Autograd

本章最有趣的部分，无疑是手动实现的自动求导引擎。

按照“训练模型”和“卷积神经网络反向传播”小节的思路，我们实现了 `Linear`、`Conv2D` 和 `MaxPooling2D` 层的前向（forward）和反向（backward）传播。所有层都实现了以下接口：

```py
class Layer:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def parameters(self):
        return []

    def __call__(self, x):
        return self.forward(x)
```

对于每一层，我们实现了一个反向传播函数，给定输出梯度，能够：

* 计算所有参数的梯度
* 计算输入的梯度并返回，以供上一层使用

在前向和反向传播中，我们封装了 CUDA 代码，线性层使用 cuBLAS 的矩阵乘法，而 `Conv2D` 和 `MaxPooling2D` 层则使用自定义的 Conv2D 前向传播实现。

我们实现了两个简单的训练示例：

* [xor](./code/autograd_manual/examples/xor_example.py)：训练一个简单的两层神经网络解决 XOR 问题
* [mnist](./code/autograd_manual/examples/mnist_example.py)：训练一个简单 CNN 在 MNIST 数据集上，3 个 epoch 内达到约 98% 的准确率

可以通过运行 `main.py` 并传入 `--xor` 或 `--mnist` 标志直接运行示例。我们没有使用 PyTorch 的训练代码，以展示 Autograd 是如何在底层工作的。事实证明，实现反向传播和训练一个性能不错的图像分类模型所需代码非常少。

运行代码前，需要先编译前向和反向传播实现。执行 Makefile：

```bash
cd code/autograd_manual
make
```

#### XOR 示例

```bash
python main.py --xor
```

训练输出示例：

```
Epoch [1000/1000], Loss: 0.000033
Final predictions:
[0.0, 0.0] => 0.0047 (0.0)
[0.0, 1.0] => 0.9951 (1.0)
[1.0, 0.0] => 0.9951 (1.0)
[1.0, 1.0] => 0.0026 (0.0)
```

训练完成，最终 loss 极小，说明网络成功学习了 XOR 函数。

#### MNIST 示例

```bash
python main.py --mnist
```

训练输出示例：

```
Epoch [3/3] completed, Loss: 0.0382, Accuracy: 98.76%
Test Accuracy: 98.74%
```

训练 3 个 epoch 后模型在测试集上的准确率达到 98.74%，并展示了一些预测示例。

### 利用 Torch 实现 Autograd

我们也实现了一个最小版本，利用 `torch` 的 autograd。展示了一个类需要实现的最小接口，使其能够与 Torch autograd 配合使用：

```py
class LinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
            
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(input)
            
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
            
        return grad_input, grad_weight, grad_bias
```

通过上述实现，可以直观理解 Torch 底层的工作原理（尤其是 backward 方法非常关键）。

### 池化

对于练习 1，我们实现了顺序的池化层：

```bash
cd code/pooling
python setup.py build_ext --inplace
python main.py
```

测试结果：

```
Max Pooling - Maximum Difference: 0.0
Average Pooling - Maximum Difference: 0.0
✓ Tests passed!
```

性能基准：

```
Input size: [8, 64, 128, 128]
Max Pooling - Custom: 82.433ms, PyTorch: 9.454ms
Avg Pooling - Custom: 47.322ms, PyTorch: 3.836ms
```

### Conv2D 反向传播

对于练习 4，我们实现了顺序的 conv2d backward，并与 Torch 实现进行对比：

```
Maximum absolute difference: 2.86102294921875e-06
Results match: True
```

## 练习分析：Conv2D `unroll_Kernel` 内存访问模式

考虑如下 CUDA kernel：

```cpp
__global__ void unroll_Kernel(int C, int H, int W, int K, float* X, float* X_unroll) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;
    if (t < C * W_unroll) {
        int c = t / W_unroll;
        int w_unroll = t % W_unroll;
        int h_out = w_unroll / W_out;
        int w_out = w_unroll % W_out;
        int w_base = c * K * K;
        for(int p = 0; p < K; p++)
            for(int q = 0; q < K; q++) {
                int h_unroll = w_base + p*K + q;
                X_unroll[h_unroll, w_unroll] = X[c, h_out + p, w_out + q];
            }
    }
}
```

我们将数组线性化为行优先（row-major）存储：

```
X_unroll[h_unroll][w_unroll] = X_unroll[(h_unroll * W_unroll) + w_unroll]
X[c][h_out + p][w_out + q] = X[(c * H * W) + ((h_out + p) * W) + (w_out + q)]
```

`t` 用于计算：

* `c = t / W_unroll`
* `w_unroll = t % W_unroll`
* `h_out = w_unroll / W_out`
* `w_out = w_unroll % W_out`

当 `t` 增加 1 时：

#### 情况1 - `c` 不变（同通道）

* `w_unroll` 增加 1
* 如果 `w_out < W_out - 1`，`w_out` 增加 1，`h_out` 不变
  → 内存地址变化为 1，**完全协同访问（perfectly coalesced）**
* 如果 `w_out` 达到行尾（`w_out == W_out - 1`），`w_out` 回到 0，`h_out` 增加 1
  → 内存地址变化为 `K`，部分协同（partial coalescing），对于小卷积核（如 3×3）仍有一些协同效果

#### 情况2 - `c` 变化（跨通道）

* 内存访问变化约为 `H * W`，非常大，不存在协同访问

#### 总结三种情况

1. **同通道、同一行（完美协同）**：连续线程访问同一行相邻输出像素。

   * 典型图像如 224×224，卷积核 3×3，每行有 222 个连续线程完美协同
2. **同通道、行边界（部分协同）**：每行结束时发生，步长为 `K`，偶尔有协同
3. **跨通道（无协同）**：发生频率极低（约每 49,284 个线程一次）

因此，对于典型 CNN 图像处理任务，绝大多数内存访问（>99%）是完美协同的，性能表现非常好。

---

