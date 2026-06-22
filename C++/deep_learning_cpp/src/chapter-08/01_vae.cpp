/*
 * 第 8 章：生成网络、自编码器与大语言模型
 * 第 274-278 页：变分自编码器 (VAE) - 重参数化与 β-VAE 损失
 */

#include <torch/torch.h>
#include <iostream>

/*
 * 第 8 章：生成网络、自编码器与大语言模型
 */
struct VAE : torch::nn::Module {
    // 编码器卷积层
    torch::nn::Conv2d enc_conv1{nullptr};
    torch::nn::Conv2d enc_conv2{nullptr};
    torch::nn::Conv2d enc_conv3{nullptr};
    torch::nn::Conv2d enc_conv4{nullptr};

    // 潜在空间参数
    torch::nn::Linear fc_mu{nullptr};
    torch::nn::Linear fc_logvar{nullptr};

    // 解码器全连接 + 转置卷积层
    torch::nn::Linear fc_decode{nullptr};
    torch::nn::ConvTranspose2d dec_conv1{nullptr};
    torch::nn::ConvTranspose2d dec_conv2{nullptr};
    torch::nn::ConvTranspose2d dec_conv3{nullptr};
    torch::nn::ConvTranspose2d dec_conv4{nullptr};

    int latent_dim;

    VAE(int z_dim = 128) : latent_dim(z_dim) {
        // 编码器
        enc_conv1 = register_module("enc_conv1",
                                    torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 4)
                                                          .stride(2)
                                                          .padding(1)));
        enc_conv2 = register_module("enc_conv2",
                                    torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 4)
                                                          .stride(2)
                                                          .padding(1)));
        enc_conv3 = register_module("enc_conv3",
                                    torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 4)
                                                          .stride(2)
                                                          .padding(1)));
        enc_conv4 = register_module("enc_conv4",
                                    torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3)
                                                          .stride(1)
                                                          .padding(0)));

        // 潜在空间映射
        fc_mu = register_module("fc_mu",
                                torch::nn::Linear(256, latent_dim));
        fc_logvar = register_module("fc_logvar",
                                    torch::nn::Linear(256, latent_dim));

        // 解码器
        fc_decode = register_module("fc_decode",
                                    torch::nn::Linear(latent_dim, 256));
        dec_conv1 = register_module("dec_conv1",
                                    torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(256, 128, 3)));
        dec_conv2 = register_module("dec_conv2",
                                    torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(128, 64, 4)
                                                                   .stride(2)
                                                                   .padding(1)));
        dec_conv3 = register_module("dec_conv3",
                                    torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(64, 32, 4)
                                                                   .stride(2)
                                                                   .padding(1)));
        dec_conv4 = register_module("dec_conv4",
                                    torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(32, 1, 4)
                                                                   .stride(2)
                                                                   .padding(1)));
    }

    /*
     * 编码：输入图像 → 均值 μ 和对数方差 log σ²
     */
    std::pair<torch::Tensor, torch::Tensor> encode(torch::Tensor x) {
        // 28x28 -> 14x14 -> 7x7 -> 1x1(c=256)
        x = torch::relu(enc_conv1->forward(x)); // [B, 32, 14, 14]
        x = torch::relu(enc_conv2->forward(x)); // [B, 64, 7, 7]
        x = torch::relu(enc_conv3->forward(x)); // [B, 128, 4, 4]
        x = torch::relu(enc_conv4->forward(x)); // [B, 256, 2, 2]

        x = x.view({x.size(0), -1}); // [B, 256]

        auto mu = fc_mu->forward(x);
        auto logvar = fc_logvar->forward(x);
        return {mu, logvar};
    }

    /*
     * 解码：潜在向量 z → 重建图像
     */
    torch::Tensor decode(torch::Tensor z) {
        auto x = torch::relu(fc_decode->forward(z)); // [B, 256]
        x = x.view({x.size(0), 256, 1, 1});          // [B, 256, 1, 1]

        x = torch::relu(dec_conv1->forward(x)); // [B, 128, 3, 3]
        x = torch::relu(dec_conv2->forward(x)); // [B, 64, 8, 8]
        x = torch::relu(dec_conv3->forward(x)); // [B, 32, 18, 18]
        x = dec_conv4->forward(x);              // [B, 1, 38, 38]
        x = torch::sigmoid(x);
        return x;
    }

    /*
     * 重参数化技巧：使梯度能够通过随机采样反向传播
     * z = μ + σ * ε，  其中 ε ~ N(0, I)
     */
    torch::Tensor reparameterize(torch::Tensor mu, torch::Tensor logvar) {
        auto std = torch::exp(0.5 * logvar);
        auto eps = torch::randn_like(std);
        return mu + eps * std;
    }

    /*
     * 前向传播：编码 → 重参数化 → 解码
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
        torch::Tensor x) {
        auto [mu, logvar] = encode(x);
        auto z = reparameterize(mu, logvar);
        auto recon = decode(z);
        return {recon, mu, logvar};
    }

    /*
     * β-VAE 损失函数
     * 损失 = 重建损失 + β * KL 散度
     * 重建损失：二元交叉熵
     * KL 散度：-0.5 * Σ(1 + log(σ²) - μ² - σ²)
     */
    static torch::Tensor vae_loss(
        torch::Tensor recon_x,
        torch::Tensor x,
        torch::Tensor mu,
        torch::Tensor logvar,
        float beta = 1.0f) {
        // 重建损失：二元交叉熵
        auto bce = torch::nn::functional::binary_cross_entropy(
            recon_x, x, torch::nn::functional::BinaryCrossEntropyFuncOptions().reduction(torch::kSum));

        // KL 散度：-0.5 * Σ(1 + log(σ²) - μ² - exp(log(σ²)))
        auto kl_loss = -0.5 * torch::sum(1 + logvar - mu.pow(2) - torch::exp(logvar));

        return bce + beta * kl_loss;
    }
};

/*
 * 第 8 章：生成网络、自编码器与大语言模型
 * 主程序：演示 VAE 的重参数化技巧与 ELBO 概念
 */
int main() {
    std::cout << "=== 变分自编码器 (VAE) 演示 ===" << std::endl;

    // 选择设备
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "使用 CUDA" << std::endl;
        device = torch::Device(torch::kCUDA);
    } else {
        std::cout << "使用 CPU" << std::endl;
    }

    int latent_dim = 128;

    // 创建 VAE 模型
    auto model = std::make_shared<VAE>(latent_dim);
    model->to(device);

    // 打印模型架构
    std::cout << "\n=== VAE 架构 ===" << std::endl;
    std::cout << *model << std::endl;

    // 定义优化器
    torch::optim::Adam optimizer(
        model->parameters(), torch::optim::AdamOptions(0.001));

    /*
     * 重参数化技巧说明：
     * 直接从 N(μ, σ²) 采样会阻断梯度流，
     * 因为采样操作不可微。
     * 重参数化将随机性移到 ε ~ N(0, I) 上，
     * 使梯度可以通过 μ 和 σ 反向传播。
     */
    std::cout << "\n=== 重参数化技巧 ===" << std::endl;
    std::cout << "z = μ + σ * ε，其中 ε ~ N(0, I)" << std::endl;
    std::cout << "这使得梯度能够通过随机采样进行反向传播" << std::endl;

    /*
     * ELBO (Evidence Lower Bound) 概念：
     * L(θ,φ; x) = E_q[log p(x|z)] - β * D_KL(q(z|x) || p(z))
     *
     * 第一项：重建损失 - 鼓励解码器生成高质量重建
     * 第二项：KL 散度 - 鼓励潜空间分布接近标准正态分布
     */
    std::cout << "\n=== ELBO 概念 ===" << std::endl;
    std::cout << "损失 = 重建损失 + β * KL 散度" << std::endl;
    std::cout << "重建损失 (BCE)：测量输入与重建之间的差异" << std::endl;
    std::cout << "KL 散度：正则化潜在空间服从 N(0, I) 分布" << std::endl;
    std::cout << "β 权重控制重建质量与潜在空间正则化之间的权衡" << std::endl;

    // 从随机噪声生成样本演示
    std::cout << "\n=== 随机生成演示 ===" << std::endl;
    {
        torch::NoGradGuard no_grad;
        auto z = torch::randn({1, latent_dim}).to(device);
        auto generated = model->decode(z);

        std::cout << "从随机潜在向量 z ~ N(0, I) 生成的图像" << std::endl;
        std::cout << "潜在向量形状: [1, " << latent_dim << "]" << std::endl;
        std::cout << "生成图像形状: " << generated.sizes() << std::endl;
        std::cout << "生成图像值范围: ["
                  << generated.min().item<float>() << ", "
                  << generated.max().item<float>() << "]" << std::endl;
    }

    // 简化的训练演示
    std::cout << "\n=== 简单训练演示 (3 轮次) ===" << std::endl;
    model->train();
    for (int epoch = 1; epoch <= 3; ++epoch) {
        auto data = torch::rand({16, 1, 28, 28}).to(device);

        auto [recon, mu, logvar] = model->forward(data);

        auto loss = VAE::vae_loss(recon, data, mu, logvar, 1.0f);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        std::cout << "轮次 [" << epoch << "/3] VAE 损失: "
                  << loss.item<float>() << std::endl;
    }

    std::cout << "\nVAE 演示完成！" << std::endl;
    return 0;
}
