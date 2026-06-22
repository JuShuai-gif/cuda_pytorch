/*
 * 02_gan.cpp - 第 8 章：生成网络、自编码器与大语言模型
 * DCGAN（Deep Convolutional GAN）的 LibTorch 实现（对应原书第 280-287 页）
 *
 * 演示内容：
 *   1. 基于 ConvTranspose2d 的 Generator（噪声 → 64×64 RGB 图像）
 *   2. 基于 Conv2d + LeakyReLU 的 Discriminator（图像 → 真/假标量）
 *   3. 对抗训练循环：判别器与生成器的交替优化
 *   4. detach() 技巧防止生成器梯度泄漏到判别器
 *   5. GAN 训练稳定性与模式坍塌（Mode Collapse）概念说明
 *
 * 关键设计要点：
 *   - BatchNorm 用于稳定深层网络训练（DCGAN 策略）
 *   - LeakyReLU（α=0.2）替代 ReLU 防止判别器死神经元
 *   - Tanh 输出限制生成像素值到 [-1, 1]
 *   - Adam β1=0.5 降低动量提高 GAN 训练稳定性（原论文建议）
 */

#include <torch/torch.h>
#include <iostream>
#include <iomanip>

/* =================== Generator（生成器） ============================= */
/*
 * DCGAN 生成器：将 100 维随机噪声 z 映射为 64×64 RGB 图像。
 *
 * 结构（反卷积上采样路径）：
 *   z(100) → Linear(100→512*4*4) → reshape(512,4,4)
 *     → deconv1(512→256,s=2) + BN + ReLU → (256,8,8)
 *     → deconv2(256→128,s=2) + BN + ReLU → (128,16,16)
 *     → deconv3(128→64, s=2)  + BN + ReLU → (64, 32,32)
 *     → deconv4(64→3,  s=2)  + Tanh      → (3,   64,64)
 *
 * 输出范围：Tanh 激活 → [-1, 1]，与真实图像归一化一致。
 */
struct Generator : torch::nn::Module {
    torch::nn::Linear fc1{nullptr};              // 噪声 → 特征空间
    torch::nn::ConvTranspose2d deconv1{nullptr}, // 512 → 256, 4→8
        deconv2{nullptr},                        // 256 → 128, 8→16
        deconv3{nullptr},                        // 128 → 64,  16→32
        deconv4{nullptr};                        // 64  → 3,   32→64
    torch::nn::BatchNorm2d bn1{nullptr},         // 用于 deconv1 之后
        bn2{nullptr},                            // 用于 deconv2 之后
        bn3{nullptr};                            // 用于 deconv3 之后

    /*
     * 构造函数
     * latent_dim=100 ：隐变量维度（标准 DCGAN 配置）
     * img_channels=3 ：输出通道数（RGB = 3）
     */
    Generator(int latent_dim = 100, int img_channels = 3) {
        // 全连接：将 100 维噪声投影到 512 个 4×4 特征图
        fc1 = register_module("fc1",
                              torch::nn::Linear(latent_dim, 512 * 4 * 4));

        // 第一层反卷积：512 × 4×4 → 256 × 8×8
        deconv1 = register_module("deconv1",
                                  torch::nn::ConvTranspose2d(
                                      torch::nn::ConvTranspose2dOptions(512, 256, /*kernel_size=*/4)
                                          .stride(2)
                                          .padding(1)
                                          .bias(false))); // BatchNorm 后接时无需 bias
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(256));

        // 第二层反卷积：256 × 8×8 → 128 × 16×16
        deconv2 = register_module("deconv2",
                                  torch::nn::ConvTranspose2d(
                                      torch::nn::ConvTranspose2dOptions(256, 128, /*kernel_size=*/4)
                                          .stride(2)
                                          .padding(1)
                                          .bias(false)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(128));

        // 第三层反卷积：128 × 16×16 → 64 × 32×32
        deconv3 = register_module("deconv3",
                                  torch::nn::ConvTranspose2d(
                                      torch::nn::ConvTranspose2dOptions(128, 64, /*kernel_size=*/4)
                                          .stride(2)
                                          .padding(1)
                                          .bias(false)));
        bn3 = register_module("bn3", torch::nn::BatchNorm2d(64));

        // 第四层反卷积：64 × 32×32 → 3 × 64×64（最终 RGB 图像）
        deconv4 = register_module("deconv4",
                                  torch::nn::ConvTranspose2d(
                                      torch::nn::ConvTranspose2dOptions(64, img_channels, /*kernel_size=*/4)
                                          .stride(2)
                                          .padding(1)));
    }

    /*
     * 前向传播：z → 64×64 RGB 图像
     * 输入 z: (batch_size, latent_dim) 随机噪声向量
     * 返回:   (batch_size, 3, 64, 64) [-1, 1] 范围图像
     */
    torch::Tensor forward(torch::Tensor z) {
        // ① 全连接投影并 reshape 为 (batch, 512, 4, 4)
        auto x = fc1->forward(z);
        x = x.view({-1, 512, 4, 4});

        // ② 反卷积上采样序列（BatchNorm + ReLU）
        x = torch::relu(bn1->forward(deconv1->forward(x))); // 4×4  → 8×8
        x = torch::relu(bn2->forward(deconv2->forward(x))); // 8×8  → 16×16
        x = torch::relu(bn3->forward(deconv3->forward(x))); // 16×16 → 32×32

        // ③ 最后一层反卷积 + Tanh 激活（无 BatchNorm）
        x = torch::tanh(deconv4->forward(x)); // 32×32 → 64×64

        return x; // (batch, 3, 64, 64)
    }
};

/* =================== Discriminator（判别器） ============================= */
/*
 * DCGAN 判别器：将 64×64 RGB 图像映射为真/假概率标量。
 *
 * 结构（卷积下采样路径）：
 *   img(3,64,64) → conv1(3→64,   s=2) + LeakyReLU → (64,  32,32)
 *                → conv2(64→128,  s=2) + BN + LeakyReLU → (128, 16,16)
 *                → conv3(128→256, s=2) + BN + LeakyReLU → (256, 8,8)
 *                → conv4(256→512, s=2) + BN + LeakyReLU → (512, 4,4)
 *                → flatten → Dropout(0.3) → Linear(512*4*4 → 1) → Sigmoid
 *
 * 输出范围：Sigmoid → [0, 1]，1=真，0=假。
 */
struct Discriminator : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr},    // 3 → 64,  64→32
        conv2{nullptr},                  // 64 → 128, 32→16
        conv3{nullptr},                  // 128 → 256, 16→8
        conv4{nullptr};                  // 256 → 512, 8→4
    torch::nn::BatchNorm2d bn1{nullptr}, // 用于 conv2 之后
        bn2{nullptr},                    // 用于 conv3 之后
        bn3{nullptr};                    // 用于 conv4 之后
    torch::nn::Linear fc1{nullptr};      // 512*4*4 → 1
    torch::nn::Dropout dropout{nullptr}; // 正则化

    /*
     * 构造函数
     * img_channels=3 ：输入通道数（RGB = 3）
     */
    explicit Discriminator(int img_channels = 3) {
        // 第一层卷积：3×64×64 → 64×32×32（无 BatchNorm，原论文建议）
        conv1 = register_module("conv1",
                                torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(img_channels, 64, /*kernel_size=*/4)
                                        .stride(2)
                                        .padding(1)));

        // 第二层卷积：64×32×32 → 128×16×16
        conv2 = register_module("conv2",
                                torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(64, 128, /*kernel_size=*/4)
                                        .stride(2)
                                        .padding(1)
                                        .bias(false))); // BatchNorm 后接时无需 bias
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(128));

        // 第三层卷积：128×16×16 → 256×8×8
        conv3 = register_module("conv3",
                                torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(128, 256, /*kernel_size=*/4)
                                        .stride(2)
                                        .padding(1)
                                        .bias(false)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(256));

        // 第四层卷积：256×8×8 → 512×4×4
        conv4 = register_module("conv4",
                                torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(256, 512, /*kernel_size=*/4)
                                        .stride(2)
                                        .padding(1)
                                        .bias(false)));
        bn3 = register_module("bn3", torch::nn::BatchNorm2d(512));

        // 全连接分类头：512*4*4 → 1（真/假二分类）
        fc1 = register_module("fc1",
                              torch::nn::Linear(512 * 4 * 4, 1));

        // Dropout 正则化（训练时随机丢弃 30% 神经元）
        dropout = register_module("dropout", torch::nn::Dropout(0.3));
    }

    /*
     * 前向传播：图像 → 真/假概率
     * 输入 x: (batch_size, 3, 64, 64)
     * 返回:   (batch_size, 1) [0, 1] 概率
     */
    torch::Tensor forward(torch::Tensor x) {
        // ① 卷积下采样序列（LeakyReLU，负斜率 0.2）
        x = torch::leaky_relu(conv1->forward(x), 0.2);               // 64×64 → 32×32
        x = torch::leaky_relu(bn1->forward(conv2->forward(x)), 0.2); // 32×32 → 16×16
        x = torch::leaky_relu(bn2->forward(conv3->forward(x)), 0.2); // 16×16 → 8×8
        x = torch::leaky_relu(bn3->forward(conv4->forward(x)), 0.2); // 8×8  → 4×4

        // ② 展平为 (batch, 512*4*4)
        x = x.view({x.size(0), -1});

        // ③ Dropout 正则化 + 全连接
        x = dropout->forward(x);
        x = fc1->forward(x);

        // ④ Sigmoid 输出真/假概率
        x = torch::sigmoid(x);

        return x; // (batch, 1)
    }
};

/* ================================= main =================================== */

int main() {
    /* ------- 参数配置 ------- */
    const int64_t latent_dim = 100; // 隐变量维度（噪声向量长度）
    const int img_channels = 3;     // RGB 图像通道数
    const int64_t batch_size = 16;  // 批次大小
    const int num_epochs = 3;       // 演示迭代轮数
    const float lr = 0.0002f;       // 学习率（DCGAN 论文推荐值）

    std::cout << "=== GAN 对抗训练演示 ===\n"
              << std::endl;
    std::cout << "潜在维度: " << latent_dim << std::endl;
    std::cout << "图像尺寸: 64×64×3" << std::endl;
    std::cout << "批次大小: " << batch_size << std::endl;
    std::cout << "学习率:   " << lr << std::endl;
    std::cout << "训练轮数: " << num_epochs << "\n"
              << std::endl;

    /* ------- 创建生成器与判别器 ------- */
    Generator generator(latent_dim, img_channels);
    Discriminator discriminator(img_channels);

    // 统计模型参数
    int64_t gen_params = 0, disc_params = 0;
    for (const auto &p : generator.parameters()) gen_params += p.numel();
    for (const auto &p : discriminator.parameters()) disc_params += p.numel();
    std::cout << "生成器参数量:   " << gen_params << std::endl;
    std::cout << "判别器参数量:   " << disc_params << "\n"
              << std::endl;

    /* ------- 优化器（DCGAN 论文推荐超参数） ------- */
    /*
     * β1=0.5 而非默认 0.9：降低动量项衰减速度，
     * 使优化方向对梯度变化更敏感，有利于 GAN 的对抗博弈稳定。
     */
    torch::optim::Adam gen_optimizer(
        generator.parameters(),
        torch::optim::AdamOptions(lr)
            .betas(std::make_tuple(0.5, 0.999)));

    torch::optim::Adam disc_optimizer(
        discriminator.parameters(),
        torch::optim::AdamOptions(lr)
            .betas(std::make_tuple(0.5, 0.999)));

    /* ------- 对抗损失（二元交叉熵） ------- */
    /*
     * GAN 损失函数：
     *   判别器：max log(D(x)) + log(1 - D(G(z)))
     *   生成器：max log(D(G(z)))
     * 实现上统一使用二元交叉熵：
     *   L = -[y·log(p) + (1-y)·log(1-p)]
     * 其中 y 为标签（1=真, 0=假），p 为判别器输出概率。
     */
    auto adversarial_loss = &torch::binary_cross_entropy;

    /* ------- 固定标签张量（复用避免重复分配） ------- */
    auto real_labels = torch::ones({batch_size, 1});  // 真实图像标签 = 1
    auto fake_labels = torch::zeros({batch_size, 1}); // 生成图像标签 = 0

    /* =========================== 训练循环 ============================== */

    for (int epoch = 1; epoch <= num_epochs; ++epoch) {
        std::cout << "--- 第 " << epoch << " 轮训练 ----------" << std::endl;

        /* ---------- 步骤 1：准备真实图像数据 ---------- */
        // 实际应用中此处应加载真实数据集（如 CelebA、CIFAR-10）
        // 演示中使用标准正态分布随机张量作为模拟「真实图像」
        auto real_images = torch::randn({batch_size, img_channels, 64, 64});

        /* ---------- 步骤 2：生成伪造图像 ---------- */
        auto noise = torch::randn({batch_size, latent_dim}); // 随机噪声
        auto fake_images = generator.forward(noise);         // 生成器输出

        /* ---------- 步骤 3：标签已在循环外预设 ---------- */
        // real_labels = ones, fake_labels = zeros（见上文）

        /* ---------- 步骤 4：训练判别器 ---------- */
        /*
         * 目标：区分真实图像（输出→1）与生成图像（输出→0）。
         *
         * !!! 关键 !!! fake_images.detach()
         * 训练判别器时调用 .detach() 切断生成器的计算图，
         * 防止判别器的梯度回传干扰生成器参数更新。
         * 若不 detach，生成器会被间接优化两次（一次在步骤 5，
         * 一次通过判别器反向传播），导致训练不稳定。
         */
        disc_optimizer.zero_grad();

        // 真实图像通过判别器 → 期望输出接近 1
        auto real_pred = discriminator.forward(real_images);
        auto loss_real = adversarial_loss(real_pred, real_labels);

        // 生成图像通过判别器（detach 切断生成器梯度） → 期望输出接近 0
        auto fake_pred = discriminator.forward(fake_images.detach());
        auto loss_fake = adversarial_loss(fake_pred, fake_labels);

        // 判别器总损失 = (真实损失 + 伪造损失) / 2（取平均）
        auto disc_loss = (loss_real + loss_fake) / 2.0;
        disc_loss.backward();
        disc_optimizer.step();

        /* ---------- 步骤 5：训练生成器 ---------- */
        /*
         * 目标：欺骗判别器，使其将生成图像判定为真实图像（输出→1）。
         *
         * 策略：将生成图像再次送入判别器，但将标签设为 1（真实），
         * 迫使生成器学习生成更逼真的图像以降低损失。
         * 注意此处 noise 重新通过 generator 构建新的计算图，
         * 判别器参数在此步骤中不更新（冻结）。
         */
        gen_optimizer.zero_grad();

        // 生成图像 → 判别器 → 与真实标签（全 1）对比
        auto gen_pred = discriminator.forward(generator.forward(noise));
        auto gen_loss = adversarial_loss(gen_pred, real_labels);
        gen_loss.backward();
        gen_optimizer.step();

        /* ---------- 打印本轮损失 ---------- */
        std::cout << std::fixed << std::setprecision(6)
                  << "  判别器损失: " << disc_loss.item<float>()
                  << " | 生成器损失: " << gen_loss.item<float>() << std::endl;
    }

    /* =========================== 总结说明 ============================== */

    std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║           GAN 对抗训练演示完成                           ║\n";
    std::cout << "╠══════════════════════════════════════════════════════╣\n";
    std::cout << "║ 关于 GAN 训练稳定性与注意事项：                         ║\n";
    std::cout << "║                                                        ║\n";
    std::cout << "║ 1. 模式坍塌（Mode Collapse）：生成器可能只学会生成     ║\n";
    std::cout << "║    少数几种样本，丧失多样性。可通过以下方法缓解：     ║\n";
    std::cout << "║    · mini-batch discrimination（批次判别）            ║\n";
    std::cout << "║    · unrolled GAN（展开 GAN）                         ║\n";
    std::cout << "║    · Wasserstein GAN（WGAN）损失函数                   ║\n";
    std::cout << "║                                                        ║\n";
    std::cout << "║ 2. 训练不稳定：判别器可能过强或过弱，导致梯度消失/爆炸 ║\n";
    std::cout << "║    · 控制判别器学习率或使用梯度惩罚（Gradient Penalty）║\n";
    std::cout << "║    · 交替更新频率（如每 5 次判别器更新 1 次生成器）    ║\n";
    std::cout << "║    · 标签平滑（Label Smoothing）：真标签设为 0.9       ║\n";
    std::cout << "║                                                        ║\n";
    std::cout << "║ 3. 评估困难：没有客观的生成质量度量标准                 ║\n";
    std::cout << "║    · Inception Score（IS）/ FID（Frechet Inception Dist.）║\n";
    std::cout << "║    · 人工视觉检查仍然是重要手段                         ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n";

    return 0;
}
