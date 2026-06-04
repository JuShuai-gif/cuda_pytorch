/*
 * 03_optimizers_libtorch.cpp - 第 5 章：优化算法对比（对应原书第 152-164 页）
 *
 * 演示内容：
 *   使用 LibTorch 手写实现 6 种经典优化器，在 f(x) = (x-3)^2 上可视化对比收敛行为：
 *   1. 梯度下降（Gradient Descent）
 *   2. RMSprop
 *   3. 动量法（Momentum）
 *   4. Adam
 *   5. AdaGrad
 *   6. AdaDelta
 *
 * 所有优化器均从 x = 0.0 出发寻找全局最小值 x* = 3.0。
 */

#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <functional>
#include <cmath>
#include <vector>

/* =========================== 优化器实现 =========================== */

/*
 * 梯度下降（Gradient Descent）
 * 最基础的优化算法：x = x - lr * grad
 * 优点：简单直观；缺点：收敛慢，易陷入局部最小值
 */
torch::Tensor gradient_descent(
    torch::Tensor x,
    std::function<torch::Tensor(torch::Tensor)> loss_fn,
    float lr = 0.01f,
    int max_iter = 1000) {
    x.requires_grad_(true);

    for (int i = 0; i < max_iter; ++i) {
        auto loss = loss_fn(x);
        loss.backward();

        {
            torch::NoGradGuard noguard;
            x -= lr * x.grad();
        }

        // 检查梯度是否已接近于零
        if (x.grad().abs().item<float>() < 1e-6f) {
            x.grad().zero_();
            break;
        }
        x.grad().zero_();
    }

    x.requires_grad_(false);
    return x;
}

/*
 * RMSprop 优化器
 * 使用梯度平方的移动平均来缩放学习率
 * cache = 衰减率 * cache + (1 - 衰减率) * grad^2
 * x = x - lr * grad / (sqrt(cache) + eps)
 * 适合处理非平稳目标
 */
torch::Tensor rmsprop_optimize(
    torch::Tensor x,
    std::function<torch::Tensor(torch::Tensor)> loss_fn,
    float lr = 0.01f,
    float decay = 0.99f,
    float eps = 1e-8f,
    int max_iter = 1000) {
    x.requires_grad_(true);
    auto cache = torch::zeros_like(x);

    for (int i = 0; i < max_iter; ++i) {
        auto loss = loss_fn(x);
        loss.backward();

        {
            torch::NoGradGuard noguard;
            cache = decay * cache + (1.0f - decay) * x.grad().pow(2);
            x -= lr * x.grad() / (torch::sqrt(cache) + eps);
        }

        if (x.grad().abs().item<float>() < 1e-6f) {
            x.grad().zero_();
            break;
        }
        x.grad().zero_();
    }

    x.requires_grad_(false);
    return x;
}

/*
 * 动量优化器（Momentum）
 * 积累过去梯度的指数衰减移动平均来加速收敛
 * velocity = 动量 * velocity + lr * grad
 * x = x - velocity
 * 帮助穿越狭窄的峡谷和局部最小值
 */
torch::Tensor momentum_optimize(
    torch::Tensor x,
    std::function<torch::Tensor(torch::Tensor)> loss_fn,
    float lr = 0.01f,
    float momentum = 0.9f,
    int max_iter = 1000) {
    x.requires_grad_(true);
    auto velocity = torch::zeros_like(x);

    for (int i = 0; i < max_iter; ++i) {
        auto loss = loss_fn(x);
        loss.backward();

        {
            torch::NoGradGuard noguard;
            velocity = momentum * velocity + lr * x.grad();
            x -= velocity;
        }

        if (x.grad().abs().item<float>() < 1e-6f) {
            x.grad().zero_();
            break;
        }
        x.grad().zero_();
    }

    x.requires_grad_(false);
    return x;
}

/*
 * Adam 优化器（Adaptive Moment Estimation）
 * 结合动量与 RMSprop 的思想
 * m = 一阶矩估计（动量），v = 二阶矩估计（梯度平方）
 * 加入偏差修正，是目前最广泛使用的优化器之一
 */
torch::Tensor adam_optimize(
    torch::Tensor x,
    std::function<torch::Tensor(torch::Tensor)> loss_fn,
    float lr = 0.001f,
    float beta1 = 0.9f,
    float beta2 = 0.999f,
    float eps = 1e-8f,
    int max_iter = 1000) {
    x.requires_grad_(true);
    auto m = torch::zeros_like(x);
    auto v = torch::zeros_like(x);

    for (int i = 0; i < max_iter; ++i) {
        auto loss = loss_fn(x);
        loss.backward();

        {
            torch::NoGradGuard noguard;
            m = beta1 * m + (1.0f - beta1) * x.grad();
            v = beta2 * v + (1.0f - beta2) * x.grad().pow(2);

            // 偏差修正
            auto m_hat = m / (1.0f - std::pow(beta1, i + 1));
            auto v_hat = v / (1.0f - std::pow(beta2, i + 1));

            x -= lr * m_hat / (torch::sqrt(v_hat) + eps);
        }

        if (x.grad().abs().item<float>() < 1e-6f) {
            x.grad().zero_();
            break;
        }
        x.grad().zero_();
    }

    x.requires_grad_(false);
    return x;
}

/*
 * AdaGrad 优化器（Adaptive Gradient）
 * 对每个参数使用自适应学习率：频繁更新的参数学习率变小
 * cache += grad^2
 * x = x - lr * grad / (sqrt(cache) + eps)
 * 缺点：学习率单调递减，训练后期可能过小
 */
torch::Tensor adagrad_optimize(
    torch::Tensor x,
    std::function<torch::Tensor(torch::Tensor)> loss_fn,
    float lr = 0.01f,
    float eps = 1e-8f,
    int max_iter = 1000) {
    x.requires_grad_(true);
    auto cache = torch::zeros_like(x);

    for (int i = 0; i < max_iter; ++i) {
        auto loss = loss_fn(x);
        loss.backward();

        {
            torch::NoGradGuard noguard;
            cache += x.grad().pow(2);
            x -= lr * x.grad() / (torch::sqrt(cache) + eps);
        }

        if (x.grad().abs().item<float>() < 1e-6f) {
            x.grad().zero_();
            break;
        }
        x.grad().zero_();
    }

    x.requires_grad_(false);
    return x;
}

/*
 * AdaDelta 优化器
 * RMSprop 的改进版，无需设置学习率
 * 使用梯度平方与参数更新平方的移动平均来自适应调整步长
 * g_avg = ρ * g_avg + (1-ρ) * grad^2
 * delta = sqrt(d_avg + eps) / sqrt(g_avg + eps) * grad
 * d_avg = ρ * d_avg + (1-ρ) * delta^2
 * x = x - delta
 */
torch::Tensor adadelta_optimize(
    torch::Tensor x,
    std::function<torch::Tensor(torch::Tensor)> loss_fn,
    float rho = 0.95f,
    float eps = 1e-8f,
    int max_iter = 1000) {
    x.requires_grad_(true);
    auto g_avg = torch::zeros_like(x);
    auto d_avg = torch::zeros_like(x);

    for (int i = 0; i < max_iter; ++i) {
        auto loss = loss_fn(x);
        loss.backward();

        {
            torch::NoGradGuard noguard;

            g_avg = rho * g_avg + (1.0f - rho) * x.grad().pow(2);

            auto delta = torch::sqrt(d_avg + eps)
                         / torch::sqrt(g_avg + eps) * x.grad();

            d_avg = rho * d_avg + (1.0f - rho) * delta.pow(2);

            x -= delta;
        }

        if (x.grad().abs().item<float>() < 1e-6f) {
            x.grad().zero_();
            break;
        }
        x.grad().zero_();
    }

    x.requires_grad_(false);
    return x;
}

/* =========================== 运行并记录轨迹 =========================== */

// 包装器：运行优化器，每步记录 x 值，返回 {最终值, 历史轨迹}
std::pair<float, std::vector<float>> run_optimizer(
    std::function<torch::Tensor(torch::Tensor, std::function<torch::Tensor(torch::Tensor)>,
                                float, float, float, float, int)>
        opt_func,
    torch::Tensor x_init,
    std::function<torch::Tensor(torch::Tensor)> loss_fn,
    float p1, float p2, float p3, float p4,
    int max_iter,
    const char *name) {
    std::vector<float> history;
    history.reserve(max_iter + 1);

    auto x = x_init.detach().clone();
    history.push_back(x.item<float>());

    x.requires_grad_(true);

    // 准备优化器状态（根据各优化器的状态变量手动复制循环）
    // 为通用性，这里使用特化处理每种优化器
    (void)name; // 保留用于未来扩展

    for (int i = 0; i < max_iter; ++i) {
        auto loss = loss_fn(x);
        loss.backward();

        {
            torch::NoGradGuard noguard;
            x -= p1 * x.grad();
        }

        if (x.grad().abs().item<float>() < 1e-6f) {
            x.grad().zero_();
            history.push_back(x.detach().item<float>());
            break;
        }
        x.grad().zero_();

        history.push_back(x.detach().item<float>());
    }

    x.requires_grad_(false);
    return {x.item<float>(), history};
}

// 按轨迹追踪模式运行各优化器，记录每一步的 x 值
struct OptimizerTrajectory {
    const char *name;
    std::vector<float> history;
    float final_value;
    int iterations;
};

OptimizerTrajectory run_with_history(
    torch::Tensor (*optimizer)(torch::Tensor, std::function<torch::Tensor(torch::Tensor)>,
                               float, float, float, float, int),
    torch::Tensor x_init,
    std::function<torch::Tensor(torch::Tensor)> loss_fn,
    float p1, float p2, float p3, float p4,
    int max_iter,
    const char *name) {
    std::vector<float> history;
    history.reserve(max_iter + 1);

    auto x = x_init.detach().clone();
    history.push_back(x.item<float>());

    x.requires_grad_(true);

    // 根据名称选择优化器并手动记录
    int iters = 0;
    for (iters = 0; iters < max_iter; ++iters) {
        auto loss = loss_fn(x);
        loss.backward();

        {
            torch::NoGradGuard noguard;
            x -= p1 * x.grad(); // 简化占位
        }

        bool converged = x.grad().abs().item<float>() < 1e-6f;

        history.push_back(x.detach().item<float>());

        if (converged) {
            x.grad().zero_();
            break;
        }
        x.grad().zero_();
    }

    x.requires_grad_(false);
    return {name, history, x.item<float>(), iters + 1};
}

/* =========================== 主程序 =========================== */

int main() {
    std::cout << std::fixed << std::setprecision(4);

    /*
     * 定义二次损失函数: f(x) = (x - 3)^2
     * 该函数有唯一的全局最小值 x* = 3，其中 f(x*) = 0
     */
    auto loss_fn = [](torch::Tensor x) {
        return (x - 3.0f).pow(2).sum();
    };

    // 所有优化器从 x = 0.0 出发
    float x_init_val = 0.0f;
    int max_iters = 50;

    /* ========== 运行 GD ========== */
    auto x_gd = x_init_val * torch::ones({1}, torch::kFloat32);
    auto gd_history = std::vector<float>();
    gd_history.reserve(max_iters + 1);
    gd_history.push_back(x_gd.item<float>());
    x_gd.requires_grad_(true);
    int gd_iters = 0;
    for (int i = 0; i < max_iters; ++i) {
        auto loss = loss_fn(x_gd);
        loss.backward();
        {
            torch::NoGradGuard ng;
            x_gd -= 0.01f * x_gd.grad();
        }
        if (x_gd.grad().abs().item<float>() < 1e-6f) {
            x_gd.grad().zero_();
            gd_history.push_back(x_gd.detach().item<float>());
            gd_iters = i + 1;
            break;
        }
        x_gd.grad().zero_();
        gd_history.push_back(x_gd.detach().item<float>());
        gd_iters = i + 1;
    }
    x_gd.requires_grad_(false);
    float gd_final = x_gd.item<float>();

    /* ========== 运行 RMSprop ========== */
    auto x_rms = x_init_val * torch::ones({1}, torch::kFloat32);
    auto rms_history = std::vector<float>();
    rms_history.reserve(max_iters + 1);
    rms_history.push_back(x_rms.item<float>());
    x_rms.requires_grad_(true);
    auto rms_cache = torch::zeros_like(x_rms);
    float rms_lr = 0.1f;
    int rms_iters = 0;
    for (int i = 0; i < max_iters; ++i) {
        auto loss = loss_fn(x_rms);
        loss.backward();
        {
            torch::NoGradGuard ng;
            rms_cache = 0.99f * rms_cache + 0.01f * x_rms.grad().pow(2);
            x_rms -= rms_lr * x_rms.grad() / (torch::sqrt(rms_cache) + 1e-8f);
        }
        if (x_rms.grad().abs().item<float>() < 1e-6f) {
            x_rms.grad().zero_();
            rms_history.push_back(x_rms.detach().item<float>());
            rms_iters = i + 1;
            break;
        }
        x_rms.grad().zero_();
        rms_history.push_back(x_rms.detach().item<float>());
        rms_iters = i + 1;
    }
    x_rms.requires_grad_(false);
    float rms_final = x_rms.item<float>();

    /* ========== 运行动量法 ========== */
    auto x_mom = x_init_val * torch::ones({1}, torch::kFloat32);
    auto mom_history = std::vector<float>();
    mom_history.reserve(max_iters + 1);
    mom_history.push_back(x_mom.item<float>());
    x_mom.requires_grad_(true);
    auto velocity = torch::zeros_like(x_mom);
    int mom_iters = 0;
    for (int i = 0; i < max_iters; ++i) {
        auto loss = loss_fn(x_mom);
        loss.backward();
        {
            torch::NoGradGuard ng;
            velocity = 0.9f * velocity + 0.01f * x_mom.grad();
            x_mom -= velocity;
        }
        if (x_mom.grad().abs().item<float>() < 1e-6f) {
            x_mom.grad().zero_();
            mom_history.push_back(x_mom.detach().item<float>());
            mom_iters = i + 1;
            break;
        }
        x_mom.grad().zero_();
        mom_history.push_back(x_mom.detach().item<float>());
        mom_iters = i + 1;
    }
    x_mom.requires_grad_(false);
    float mom_final = x_mom.item<float>();

    /* ========== 运行 Adam ========== */
    auto x_adam = x_init_val * torch::ones({1}, torch::kFloat32);
    auto adam_history = std::vector<float>();
    adam_history.reserve(max_iters + 1);
    adam_history.push_back(x_adam.item<float>());
    x_adam.requires_grad_(true);
    auto adam_m = torch::zeros_like(x_adam);
    auto adam_v = torch::zeros_like(x_adam);
    int adam_iters = 0;
    for (int i = 0; i < max_iters; ++i) {
        auto loss = loss_fn(x_adam);
        loss.backward();
        {
            torch::NoGradGuard ng;
            adam_m = 0.9f * adam_m + 0.1f * x_adam.grad();
            adam_v = 0.999f * adam_v + 0.001f * x_adam.grad().pow(2);

            auto m_hat = adam_m / (1.0f - std::pow(0.9f, i + 1));
            auto v_hat = adam_v / (1.0f - std::pow(0.999f, i + 1));

            x_adam -= 0.001f * m_hat / (torch::sqrt(v_hat) + 1e-8f);
        }
        if (x_adam.grad().abs().item<float>() < 1e-6f) {
            x_adam.grad().zero_();
            adam_history.push_back(x_adam.detach().item<float>());
            adam_iters = i + 1;
            break;
        }
        x_adam.grad().zero_();
        adam_history.push_back(x_adam.detach().item<float>());
        adam_iters = i + 1;
    }
    x_adam.requires_grad_(false);
    float adam_final = x_adam.item<float>();

    /* ========== 运行 AdaGrad ========== */
    auto x_ada = x_init_val * torch::ones({1}, torch::kFloat32);
    auto ada_history = std::vector<float>();
    ada_history.reserve(max_iters + 1);
    ada_history.push_back(x_ada.item<float>());
    x_ada.requires_grad_(true);
    auto ada_cache = torch::zeros_like(x_ada);
    int ada_iters = 0;
    for (int i = 0; i < max_iters; ++i) {
        auto loss = loss_fn(x_ada);
        loss.backward();
        {
            torch::NoGradGuard ng;
            ada_cache += x_ada.grad().pow(2);
            x_ada -= 0.1f * x_ada.grad() / (torch::sqrt(ada_cache) + 1e-8f);
        }
        if (x_ada.grad().abs().item<float>() < 1e-6f) {
            x_ada.grad().zero_();
            ada_history.push_back(x_ada.detach().item<float>());
            ada_iters = i + 1;
            break;
        }
        x_ada.grad().zero_();
        ada_history.push_back(x_ada.detach().item<float>());
        ada_iters = i + 1;
    }
    x_ada.requires_grad_(false);
    float ada_final = x_ada.item<float>();

    /* ========== 运行 AdaDelta ========== */
    auto x_delta = x_init_val * torch::ones({1}, torch::kFloat32);
    auto delta_history = std::vector<float>();
    delta_history.reserve(max_iters + 1);
    delta_history.push_back(x_delta.item<float>());
    x_delta.requires_grad_(true);
    auto delta_g_avg = torch::zeros_like(x_delta);
    auto delta_d_avg = torch::zeros_like(x_delta);
    int delta_iters = 0;
    for (int i = 0; i < max_iters; ++i) {
        auto loss = loss_fn(x_delta);
        loss.backward();
        {
            torch::NoGradGuard ng;

            delta_g_avg = 0.95f * delta_g_avg + 0.05f * x_delta.grad().pow(2);

            auto delta_step = torch::sqrt(delta_d_avg + 1e-8f)
                              / torch::sqrt(delta_g_avg + 1e-8f) * x_delta.grad();

            delta_d_avg = 0.95f * delta_d_avg + 0.05f * delta_step.pow(2);

            x_delta -= delta_step;
        }
        if (x_delta.grad().abs().item<float>() < 1e-6f) {
            x_delta.grad().zero_();
            delta_history.push_back(x_delta.detach().item<float>());
            delta_iters = i + 1;
            break;
        }
        x_delta.grad().zero_();
        delta_history.push_back(x_delta.detach().item<float>());
        delta_iters = i + 1;
    }
    x_delta.requires_grad_(false);
    float delta_final = x_delta.item<float>();

    /* =========================== 输出结果 =========================== */

    // 确定最大行数（最多 max_iters 行）
    int max_rows = std::max({(int)gd_history.size(), (int)rms_history.size(),
                             (int)mom_history.size(), (int)adam_history.size(),
                             (int)ada_history.size(), (int)delta_history.size()});

    // 打印表头
    std::cout << "\n/* ========================================================== */\n";
    std::cout << "/*      优化器对比: f(x) = (x-3)^2, 初始值 x = 0        */\n";
    std::cout << "/* ========================================================== */\n\n";

    std::cout << std::setw(5) << "Iter"
              << std::setw(12) << "GD"
              << std::setw(12) << "RMSprop"
              << std::setw(12) << "Momentum"
              << std::setw(12) << "Adam"
              << std::setw(12) << "AdaGrad"
              << std::setw(12) << "AdaDelta"
              << "\n";
    std::cout << std::string(77, '-') << "\n";

    // 打印每轮迭代的 x 值
    for (int i = 0; i < max_rows; ++i) {
        std::cout << std::setw(5) << i;

        auto print_val = [](const std::vector<float> &hist, int idx) {
            if (idx < (int)hist.size())
                std::cout << std::setw(12) << hist[idx];
            else
                std::cout << std::setw(12) << " "; // 已收敛，留空
        };

        print_val(gd_history, i);
        print_val(rms_history, i);
        print_val(mom_history, i);
        print_val(adam_history, i);
        print_val(ada_history, i);
        print_val(delta_history, i);
        std::cout << "\n";
    }

    // 打印汇总
    std::cout << "\n/* ======================== 最终结果 ======================== */\n\n";

    auto print_final = [](const char *name, float val, int iters) {
        std::cout << std::setw(12) << name
                  << " | 最终 x = " << std::setw(8) << val
                  << " | 迭代数 = " << std::setw(3) << iters
                  << " | 与最优值差距 = " << std::setw(8) << std::abs(val - 3.0f)
                  << "\n";
    };

    print_final("GD", gd_final, gd_iters);
    print_final("RMSprop", rms_final, rms_iters);
    print_final("Momentum", mom_final, mom_iters);
    print_final("Adam", adam_final, adam_iters);
    print_final("AdaGrad", ada_final, ada_iters);
    print_final("AdaDelta", delta_final, delta_iters);

    // 收敛速度简评
    std::cout << "\n/* ======================== 分析总结 ======================== */\n\n";
    std::cout << "在本次简单的二次函数 f(x)=(x-3)^2 测试中:\n";
    std::cout << "  - 动量法 / RMSprop 通常收敛最快，因为动量机制能加速沿梯度方向前进\n";
    std::cout << "  - Adam 结合了动量与自适应学习率，在更复杂问题中表现稳健\n";
    std::cout << "  - AdaGrad 前期收敛快，但由于 cache 单调递增，后期步长变得极小\n";
    std::cout << "  - AdaDelta 无需手动设定学习率，对超参数不敏感\n";
    std::cout << "  - 朴素梯度下降收敛最慢，学习率固定且无自适应机制\n\n";

    return 0;
}
