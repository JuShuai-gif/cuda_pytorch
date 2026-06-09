#include "cache_bench.h"
#include "timer.h"

#include <iostream>
#include <iomanip>
#include <cstring>
#include <thread>
#include <cmath>

extern void print_header(const std::string &title);

// ============================================================================
// 卡尔曼跟踪预测步: x = F*x, P = F*P*F^T + Q
// 状态转移矩阵 F (恒定加速度模型):
// x(t+dt) = x + vx*dt + 0.5*ax*dt^2
// vx(t+dt) = vx + ax*dt
// ax(t+dt) = ax
// ============================================================================
void KalmanTrack::predict(float dt) {
    float dt2 = 0.5f * dt * dt;

    // 过程噪声 Q (简化: 小对角矩阵)
    constexpr float q_pos = 0.01f;
    constexpr float q_vel = 0.1f;
    constexpr float q_acc = 0.5f;

    // 预测状态: x = F * x
    float new_state[6];
    new_state[0] = state[0] + state[2] * dt + state[4] * dt2; // x
    new_state[1] = state[1] + state[3] * dt + state[5] * dt2; // y
    new_state[2] = state[2] + state[4] * dt;                  // vx
    new_state[3] = state[3] + state[5] * dt;                  // vy
    new_state[4] = state[4];                                  // ax
    new_state[5] = state[5];                                  // ay

    // 预测协方差: P = F * P * F^T + Q
    // F 矩阵 (6x6):
    // [1, 0, dt, 0, dt2, 0]
    // [0, 1, 0, dt, 0, dt2]
    // [0, 0, 1, 0, dt, 0]
    // [0, 0, 0, 1, 0, dt]
    // [0, 0, 0, 0, 1, 0]
    // [0, 0, 0, 0, 0, 1]

    float tmp[36]; // F * P 临时矩阵
    float new_cov[36];

    // tmp = F * P (行主序)
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
            float sum = 0.0f;
            // F 矩阵的第 r 行:
            if (r == 0) {
                sum = cov[0 * 6 + c] + dt * cov[2 * 6 + c] + dt2 * cov[4 * 6 + c];
            } else if (r == 1) {
                sum = cov[1 * 6 + c] + dt * cov[3 * 6 + c] + dt2 * cov[5 * 6 + c];
            } else if (r == 2) {
                sum = cov[2 * 6 + c] + dt * cov[4 * 6 + c];
            } else if (r == 3) {
                sum = cov[3 * 6 + c] + dt * cov[5 * 6 + c];
            } else if (r == 4) {
                sum = cov[4 * 6 + c];
            } else if (r == 5) {
                sum = cov[5 * 6 + c];
            }
            tmp[r * 6 + c] = sum;
        }
    }

    // new_cov = tmp * F^T = (F * P) * F^T
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
            float sum = 0.0f;
            if (c == 0) {
                sum = tmp[r * 6 + 0] + dt * tmp[r * 6 + 2] + dt2 * tmp[r * 6 + 4];
            } else if (c == 1) {
                sum = tmp[r * 6 + 1] + dt * tmp[r * 6 + 3] + dt2 * tmp[r * 6 + 5];
            } else if (c == 2) {
                sum = tmp[r * 6 + 2] + dt * tmp[r * 6 + 4];
            } else if (c == 3) {
                sum = tmp[r * 6 + 3] + dt * tmp[r * 6 + 5];
            } else if (c == 4) {
                sum = tmp[r * 6 + 4];
            } else if (c == 5) {
                sum = tmp[r * 6 + 5];
            }
            new_cov[r * 6 + c] = sum;
        }
    }

    // 添加过程噪声 Q (对角矩阵)
    new_cov[0 * 6 + 0] += q_pos;
    new_cov[1 * 6 + 1] += q_pos;
    new_cov[2 * 6 + 2] += q_vel;
    new_cov[3 * 6 + 3] += q_vel;
    new_cov[4 * 6 + 4] += q_acc;
    new_cov[5 * 6 + 5] += q_acc;

    for (int i = 0; i < 6; ++i) state[i] = new_state[i];
    for (int i = 0; i < 36; ++i) cov[i] = new_cov[i];
}

// ============================================================================
// 跟踪数组: 8 个跟踪位于同一条缓存行 (伪共享)
// ============================================================================
struct UnpaddedTracks {
    KalmanTrack tracks[8];
};

// ============================================================================
// 跟踪数组: 每个跟踪位于独立的缓存行 (无伪共享)
// ============================================================================
struct alignas(64) PaddedTrack {
    KalmanTrack track;
    char pad[64 - sizeof(KalmanTrack) % 64];
};

struct PaddedTracks {
    alignas(64) PaddedTrack tracks[8];
};

void demo_false_sharing() {
    print_header("演示 1: 伪共享 (卡尔曼滤波预测)");

    const int64_t iterations = 500000;
    const int num_samples = 5;

    // 初始化跟踪数据
    auto init_track = [](KalmanTrack &t, int id) {
        t.track_id = id;
        t.state[0] = static_cast<float>(id * 10.0f); // x
        t.state[1] = static_cast<float>(id * 5.0f);  // y
        t.state[2] = 1.5f;                           // vx
        t.state[3] = 0.2f;                           // vy
        t.state[4] = 0.0f;                           // ax
        t.state[5] = 0.0f;                           // ay
        for (int i = 0; i < 36; ++i) t.cov[i] = (i % 7 == 0) ? 1.0f : 0.0f;
    };

    // 测试未填充 (伪共享): 2 个线程各处理 4 个跟踪
    double unpadded_ms = 0.0;
    for (int s = 0; s < num_samples; ++s) {
        UnpaddedTracks ut;
        for (int i = 0; i < 8; ++i) init_track(ut.tracks[i], i);

        Timer timer;
        timer.start();
        std::thread t1([&]() {
            for (int64_t iter = 0; iter < iterations; ++iter) {
                float dt = 0.1f;
                for (int i = 0; i < 4; ++i) ut.tracks[i].predict(dt);
            }
        });
        std::thread t2([&]() {
            for (int64_t iter = 0; iter < iterations; ++iter) {
                float dt = 0.1f;
                for (int i = 4; i < 8; ++i) ut.tracks[i].predict(dt);
            }
        });
        t1.join();
        t2.join();
        unpadded_ms += timer.elapsed_ms();
    }
    unpadded_ms /= num_samples;

    // 测试填充 (无伪共享)
    double padded_ms = 0.0;
    for (int s = 0; s < num_samples; ++s) {
        PaddedTracks pt;
        for (int i = 0; i < 8; ++i) init_track(pt.tracks[i].track, i);

        Timer timer;
        timer.start();
        std::thread t1([&]() {
            for (int64_t iter = 0; iter < iterations; ++iter) {
                float dt = 0.1f;
                for (int i = 0; i < 4; ++i) pt.tracks[i].track.predict(dt);
            }
        });
        std::thread t2([&]() {
            for (int64_t iter = 0; iter < iterations; ++iter) {
                float dt = 0.1f;
                for (int i = 4; i < 8; ++i) pt.tracks[i].track.predict(dt);
            }
        });
        t1.join();
        t2.join();
        padded_ms += timer.elapsed_ms();
    }
    padded_ms /= num_samples;

    double ops_per_iter = 8.0 * 400.0; // 每个卡尔曼预测约 400 次浮点运算, 8 个跟踪
    double total_ops = ops_per_iter * iterations;

    std::cout << "\n两个线程各自运行 4 个跟踪的卡尔曼预测，"
              << iterations << " 次迭代:\n\n";
    std::cout << std::left
              << std::setw(35) << "布局"
              << std::setw(16) << "平均耗时(ms)"
              << std::setw(20) << "吞吐量(Mop/s)\n";
    std::cout << std::string(71, '-') << "\n";
    std::cout << std::left
              << std::setw(35) << "未填充 (伪共享)"
              << std::setw(16) << std::fixed << std::setprecision(2) << unpadded_ms
              << std::setw(20) << std::fixed << std::setprecision(1)
              << (total_ops / unpadded_ms / 1e3) << "\n";
    std::cout << std::left
              << std::setw(35) << "已填充 (独立缓存行)"
              << std::setw(16) << std::fixed << std::setprecision(2) << padded_ms
              << std::setw(20) << std::fixed << std::setprecision(1)
              << (total_ops / padded_ms / 1e3) << "\n";
    std::cout << std::left
              << std::setw(35) << "加速比 (已填充 vs 未填充)"
              << std::setw(16) << std::fixed << std::setprecision(2)
              << (unpadded_ms / padded_ms) << "x\n";

    std::cout << "\n解释: 当 8 个 KalmanTrack 结构体共享缓存行时，\n"
              << "访问不同跟踪的线程会导致缓存行乒乓效应。\n"
              << "将每个跟踪填充到 64 字节可以消除伪共享。\n";
}

// ============================================================================
// 演示 3: 缓存抖动 - 图像块处理
// 对 640x480x3 图像应用 3x3 盒式模糊。
// 行主序: 先遍历行再遍历列 (连续访问)
// 列主序: 先遍历列再遍历行 (跨步访问)
// ============================================================================
void demo_cache_thrashing() {
    print_header("演示 3: 缓存抖动 (图像 3x3 模糊)");

    const int W = 640;
    const int H = 480;
    const int C = 3;
    const int num_samples = 5;

    // 分配一维图像: [C][H][W] 通道优先
    float *src = new float[C * H * W];
    float *dst = new float[C * H * W];

    // 用合成图案初始化
    for (int c = 0; c < C; ++c) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                src[c * H * W + y * W + x] =
                    static_cast<float>((x * 7 + y * 13 + c * 31) % 256) / 255.0f;
            }
        }
    }

    // 行主序遍历 (缓存友好)
    double row_major_ms = 0.0;
    for (int s = 0; s < num_samples; ++s) {
        Timer timer;
        timer.start();
        for (int c = 0; c < C; ++c) {
            for (int y = 1; y < H - 1; ++y) {
                for (int x = 1; x < W - 1; ++x) {
                    float sum = 0.0f;
                    for (int ky = -1; ky <= 1; ++ky) {
                        for (int kx = -1; kx <= 1; ++kx) {
                            sum += src[c * H * W + (y + ky) * W + (x + kx)];
                        }
                    }
                    dst[c * H * W + y * W + x] = sum / 9.0f;
                }
            }
        }
        row_major_ms += timer.elapsed_ms();
    }
    row_major_ms /= num_samples;

    // 列主序遍历 (缓存不友好)
    double col_major_ms = 0.0;
    for (int s = 0; s < num_samples; ++s) {
        Timer timer;
        timer.start();
        for (int c = 0; c < C; ++c) {
            for (int x = 1; x < W - 1; ++x) {
                for (int y = 1; y < H - 1; ++y) {
                    float sum = 0.0f;
                    for (int ky = -1; ky <= 1; ++ky) {
                        for (int kx = -1; kx <= 1; ++kx) {
                            sum += src[c * H * W + (y + ky) * W + (x + kx)];
                        }
                    }
                    dst[c * H * W + y * W + x] = sum / 9.0f;
                }
            }
        }
        col_major_ms += timer.elapsed_ms();
    }
    col_major_ms /= num_samples;

    double pixels_per_blur = static_cast<double>(C) * (W - 2) * (H - 2);

    std::cout << "\n在 " << W << "x" << H << "x" << C
              << " 图像 (" << (C * W * H) / 1000 << "K 像素) 上进行 3x3 盒式模糊:\n\n";
    std::cout << std::left
              << std::setw(35) << "遍历顺序"
              << std::setw(16) << "平均耗时(ms)"
              << std::setw(22) << "吞吐量(Mpix/s)\n";
    std::cout << std::string(73, '-') << "\n";
    std::cout << std::left
              << std::setw(35) << "行主序 (缓存友好)"
              << std::setw(16) << std::fixed << std::setprecision(2) << row_major_ms
              << std::setw(22) << std::fixed << std::setprecision(1)
              << (pixels_per_blur / row_major_ms / 1e3) << "\n";
    std::cout << std::left
              << std::setw(35) << "列主序 (缓存不友好)"
              << std::setw(16) << std::fixed << std::setprecision(2) << col_major_ms
              << std::setw(22) << std::fixed << std::setprecision(1)
              << (pixels_per_blur / col_major_ms / 1e3) << "\n";
    std::cout << std::left
              << std::setw(35) << "加速比 (行主序 vs 列主序)"
              << std::setw(16) << std::fixed << std::setprecision(2)
              << (col_major_ms / row_major_ms) << "x\n";

    std::cout << "\n解释: 行主序遍历访问连续内存\n"
              << "(步长-1)，利用空间局部性和硬件预取。\n"
              << "列主序遍历每次迭代跨越 W*sizeof(float) 字节，\n"
              << "导致几乎每次访问都会缓存未命中。\n";

    delete[] src;
    delete[] dst;
}

// ============================================================================
// 工具: 估算 CPU 缓存行大小
// ============================================================================
int estimate_cache_line_size() {
#ifdef _SC_LEVEL1_DCACHE_LINESIZE
    long sz = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    return static_cast<int>(sz > 0 ? sz : 64);
#else
    return 64;
#endif
}
