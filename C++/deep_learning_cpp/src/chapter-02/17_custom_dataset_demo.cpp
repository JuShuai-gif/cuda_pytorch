/*
 * custom_dataset_demo.cpp
 * 第2章：C++ 中的数据准备与预处理
 *
 * PyTorch 的 C++ Dataset API 提供了加载和预处理数据的清晰抽象。
 * 它将数据存储与数据迭代分离，支持多线程加载、打乱和批处理等特性。
 *
 * 关键组件：
 *   - Dataset：定义如何加载单个样本（get / size）
 *   - DataLoader：遍历 Dataset，提供批处理、打乱以及通过工作线程进行并行加载。
 *
 * 何时使用自定义 Dataset：
 *   - 数据无法全部装入内存时（从磁盘延迟加载）
 *   - 每个样本需要自定义预处理时
 *   - 非标准数据格式（专有二进制格式、HDF5 等）
 *   - 多模态数据（图像 + 文本 + 元数据）
 *
 * 本示例创建一个简单的内存数据集，并演示如何使用 DataLoader 进行批处理，
 * 这对于训练循环至关重要。
 */

#include <torch/torch.h>
#include <iostream>

// ----------------------------------------------------------------
// 自定义 Dataset：将特征 (X) 和标签 (y) 存储在张量中。
// 每次调用 get() 返回一个单独的 {特征, 标签} 对。
// ----------------------------------------------------------------
struct CustomDataset : torch::data::datasets::Dataset<CustomDataset> {
    CustomDataset(torch::Tensor X, torch::Tensor y) : data_(X.contiguous().to(torch::kFloat32)),
                                                      labels_(y.contiguous().to(torch::kLong)) {
    }

    // 返回单个训练样本：{输入, 目标}
    torch::data::Example<> get(size_t index) override {
        return {data_[index], labels_[index]};
    }

    // 数据集中的样本数量
    torch::optional<size_t> size() const override {
        return data_.size(0);
    }

private:
    torch::Tensor data_, labels_;
};

int main() {
    std::cout << "=== PyTorch Custom Dataset & DataLoader Demo ===\n\n";

    // 创建合成数据：100 个样本，每个 5 个特征
    // 特征：服从正态分布的随机数
    // 标签：3 个类别 (0, 1, 2)
    int numSamples = 100;
    int numFeatures = 5;
    int numClasses = 3;

    torch::manual_seed(42);
    auto X = torch::randn({numSamples, numFeatures});
    auto y = torch::randint(0, numClasses, {numSamples});

    std::cout << "Synthetic data: " << numSamples << " samples, "
              << numFeatures << " features, " << numClasses << " classes\n\n";

    // 创建数据集
    auto dataset = CustomDataset(X, y)
                       .map(torch::data::transforms::Stack<>());

    // 带有批处理和打乱的 DataLoader
    // 打乱至关重要：防止模型学习批次顺序模式，提升泛化能力。
    int batchSize = 16;
    auto dataLoader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions()
            .batch_size(batchSize)
            .workers(0) // 0 = 单线程（便于调试）
    );

    // 遍历批次（模拟训练循环）
    std::cout << "Iterating batches (batch_size=" << batchSize << "):\n";
    int batchNum = 0;
    for (auto &batch : *dataLoader) {
        auto data = batch.data;
        auto targets = batch.target;

        std::cout << "  Batch " << batchNum++
                  << ": data shape " << data.sizes()
                  << ", labels shape " << targets.sizes()
                  << ", labels: [";

        // 打印每个批次的前 5 个标签
        auto targets_acc = targets.accessor<int64_t, 1>();
        for (int i = 0; i < std::min((int)targets.size(0), 5); ++i) {
            std::cout << targets_acc[i];
            if (i + 1 < std::min((int)targets.size(0), 5)) std::cout << ", ";
        }
        std::cout << "...]\n";

        if (batchNum >= 3) break; // 仅显示前 3 个批次
    }
    int totalBatches = (numSamples + batchSize - 1) / batchSize;
    std::cout << "  ... (" << totalBatches << " total batches)\n\n";

    // 关键 DataLoader 选项：
    std::cout << "Key DataLoader Options:\n";
    std::cout << "  batch_size: Number of samples per batch (GPU memory limit)\n";
    std::cout << "  workers: Parallel data loading threads (>0 for I/O-bound data)\n";
    std::cout << "  shuffle: Important for training (prevents batch-order bias)\n";
    std::cout << "  drop_last: Drop incomplete last batch (for batchnorm consistency)\n";

    return 0;
}
