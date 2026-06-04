/*
 * 第 8 章：生成网络、自编码器与大语言模型
 * 第 268-271 页：CNN 自编码器 (CNN Autoencoder)
 */

#include <torch/torch.h>
#include <iostream>

/*
 * 第 8 章：生成网络、自编码器与大语言模型
 */
struct CNNAutoEncoder : torch::nn::Module {
    // 编码器：压缩输入图像到紧凑的潜在表示
    torch::nn::Conv2d enc_conv1{nullptr};
    torch::nn::Conv2d enc_conv2{nullptr};
    torch::nn::Conv2d enc_conv3{nullptr};
    torch::nn::MaxPool2d pool{nullptr};

    // 瓶颈层：进一步压缩
    torch::nn::Conv2d bottleneck{nullptr};

    // 解码器：从潜在表示重建原始图像
    torch::nn::ConvTranspose2d dec_conv1{nullptr};
    torch::nn::ConvTranspose2d dec_conv2{nullptr};
    torch::nn::ConvTranspose2d dec_conv3{nullptr};
    torch::nn::Upsample upsample{nullptr};

    CNNAutoEncoder() {
        // 28x28 -> 14x14 -> 26x26
        enc_conv1 = register_module("enc_conv1",
                                    torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 3).padding(1)));
        pool = register_module("pool",
                               torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)));
        // 28x28 -> 14x14 -> 12x12
        enc_conv2 = register_module("enc_conv2",
                                    torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3)));
        // 12x12 -> 6x6 -> 4x4
        enc_conv3 = register_module("enc_conv3",
                                    torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3)));
        // 4x4 -> 2x2 -> 2x2
        bottleneck = register_module("bottleneck",
                                     torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 64, 3).padding(1)));

        // 2x2 -> 2x2 -> 4x4
        dec_conv1 = register_module("dec_conv1",
                                    torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(64, 128, 3).padding(1)));
        upsample = register_module("upsample",
                                   torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor({2, 2})));

        // 4x4 -> 8x8
        dec_conv2 = register_module("dec_conv2",
                                    torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(128, 64, 3).padding(1)));

        // 8x8 -> 16x16 -> 22x22 -> 28x28
        dec_conv3 = register_module("dec_conv3",
                                    torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(64, 1, 3)));
    }

    /*
     * 前向传播：编码 → 瓶颈 → 解码 → 重建
     */
    torch::Tensor forward(torch::Tensor input) {
        // 编码器路径
        auto x = torch::relu(enc_conv1->forward(input)); // [B, 32, 28, 28]
        x = pool->forward(x);                            // [B, 32, 14, 14]
        x = torch::relu(enc_conv2->forward(x));          // [B, 64, 12, 12]
        x = pool->forward(x);                            // [B, 64, 6, 6]
        x = torch::relu(enc_conv3->forward(x));          // [B, 128, 4, 4]
        x = pool->forward(x);                            // [B, 128, 2, 2]

        // 瓶颈层
        x = torch::relu(bottleneck->forward(x)); // [B, 64, 2, 2]

        // 解码器路径
        x = torch::relu(dec_conv1->forward(x)); // [B, 128, 4, 4]
        x = upsample->forward(x);               // [B, 128, 8, 8]
        x = torch::relu(dec_conv2->forward(x)); // [B, 64, 10, 10]
        x = upsample->forward(x);               // [B, 64, 20, 20]
        x = dec_conv3->forward(x);              // [B, 1, 22, 22]
        x = upsample->forward(x);               // [B, 1, 44, 44]

        // 最终 sigmoid 将输出值压缩到 [0, 1]
        x = torch::sigmoid(x);
        return x;
    }
};

/*
 * 第 8 章：生成网络、自编码器与大语言模型
 * 主程序：演示 CNN 自编码器的训练与重建
 */
int main() {
    // 选择设备：优先使用 CUDA，否则用 CPU
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "使用 CUDA 进行训练" << std::endl;
        device = torch::Device(torch::kCUDA);
    } else {
        std::cout << "使用 CPU 进行训练" << std::endl;
    }

    // 创建 CNN 自编码器模型
    CNNAutoEncoder model;
    model.to(device);

    // 打印模型架构摘要
    std::cout << "\n=== CNN 自编码器架构 ===" << std::endl;
    std::cout << model << std::endl;

    // 定义优化器
    torch::optim::Adam optimizer(
        model.parameters(), torch::optim::AdamOptions(0.001));

    int num_epochs = 5;
    int batch_size = 64;

    std::cout << "\n开始训练，" << num_epochs << " 个轮次..." << std::endl;

    // 演示训练循环
    for (int epoch = 1; epoch <= num_epochs; ++epoch) {
        model.train();
        float epoch_loss = 0.0;
        int num_batches = 10; // 每轮次 10 个批次

        for (int batch = 0; batch < num_batches; ++batch) {
            // 使用随机噪声作为演示数据
            auto data = torch::randn({batch_size, 1, 28, 28}).to(device);

            // 前向传播
            auto output = model.forward(data);

            // 计算 MSE 损失
            auto loss = torch::mse_loss(output, data);

            // 反向传播与参数更新
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            epoch_loss += loss.item<float>();
        }

        float avg_loss = epoch_loss / num_batches;
        std::cout << "轮次 [" << epoch << "/" << num_epochs
                  << "] 平均损失: " << avg_loss << std::endl;
    }

    // 训练后重建演示
    std::cout << "\n=== 重建演示 ===" << std::endl;
    model.eval();
    {
        // 取一个随机样本进行编码-解码
        auto sample = torch::randn({1, 1, 28, 28}).to(device);
        torch::NoGradGuard no_grad;
        auto reconstruction = model.forward(sample);

        // 打印相似度（MSE 越小越相似）
        auto similarity = torch::mse_loss(reconstruction, sample);
        std::cout << "重建输入与输出之间的 MSE: " << similarity.item<float>()
                  << std::endl;
        std::cout << "输入最小值/最大值: "
                  << sample.min().item<float>() << " / "
                  << sample.max().item<float>() << std::endl;
        std::cout << "重建最小值/最大值: "
                  << reconstruction.min().item<float>() << " / "
                  << reconstruction.max().item<float>() << std::endl;
    }

    std::cout << "\n自编码器演示完成！" << std::endl;
    return 0;
}
