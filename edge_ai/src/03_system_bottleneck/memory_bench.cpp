#include "memory_bench.h"
#include "timer.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>
#include <cstdint>
#include <algorithm>

extern void print_header(const std::string &title);

// ============================================================================
// 相机帧环形缓冲区: 固定大小的缓冲区，带写入指针。
// 用于比较拷贝帧与传递 const 指针。
// ============================================================================
class CameraRingBuffer {
public:
    explicit CameraRingBuffer(int num_slots, size_t frame_bytes) : slots_(num_slots), frame_bytes_(frame_bytes), write_idx_(0) {
        for (auto &slot : slots_) {
            slot.data = new uint8_t[frame_bytes];
        }
    }
    ~CameraRingBuffer() {
        for (auto &slot : slots_) {
            delete[] slot.data;
        }
    }

    // 通过拷贝数据写入一帧 (成本高)
    void write_copy(const uint8_t *src, int64_t timestamp_us) {
        auto &slot = slots_[write_idx_];
        std::memcpy(slot.data, src, frame_bytes_);
        slot.timestamp_us = timestamp_us;
        write_idx_ = (write_idx_ + 1) % slots_.size();
    }

    // 通过交换底层缓冲区指针写入一帧 (零拷贝)
    // 获取调用者缓冲区的所有权; 调用者获得旧缓冲区。
    void write_swap(uint8_t *&src, int64_t timestamp_us) {
        auto &slot = slots_[write_idx_];
        std::swap(slot.data, src);
        slot.timestamp_us = timestamp_us;
        write_idx_ = (write_idx_ + 1) % slots_.size();
    }

    int num_slots() const {
        return static_cast<int>(slots_.size());
    }

private:
    struct Slot {
        uint8_t *data = nullptr;
        int64_t timestamp_us = 0;
    };
    std::vector<Slot> slots_;
    size_t frame_bytes_;
    int write_idx_;
};

// ============================================================================
// 演示 4: 使用相机帧尺寸进行内存拷贝开销测试
// ============================================================================
void demo_memory_copy() {
    print_header("演示 4: 内存拷贝开销 (相机帧)");

    // 相机帧: 1920x1080x3 uint8 = 6,220,800 字节
    constexpr int IMG_W = 1920;
    constexpr int IMG_H = 1080;
    constexpr int IMG_C = 3;
    constexpr size_t FRAME_BYTES = IMG_W * IMG_H * IMG_C;
    const int num_samples = 10;
    const int ring_slots = 8;

    std::cout << "\n相机帧: " << IMG_W << "x" << IMG_H << "x" << IMG_C
              << " uint8 = " << (FRAME_BYTES / (1024.0 * 1024.0)) << " MB\n\n";

    // 分配源和目标
    uint8_t *src = new uint8_t[FRAME_BYTES];
    uint8_t *dst = new uint8_t[FRAME_BYTES];

    // 填充合成图像数据
    for (size_t i = 0; i < FRAME_BYTES; ++i) {
        src[i] = static_cast<uint8_t>(i * 7 % 256);
    }

    std::cout << std::left
              << std::setw(28) << "方法"
              << std::setw(14) << "耗时(us)"
              << std::setw(20) << "带宽(GB/s)\n";
    std::cout << std::string(62, '-') << "\n";

    // 1. std::memcpy
    {
        double total_us = 0.0;
        for (int s = 0; s < num_samples; ++s) {
            Timer timer;
            timer.start();
            std::memcpy(dst, src, FRAME_BYTES);
            total_us += timer.elapsed_us();
        }
        double avg_us = total_us / num_samples;
        double bw = (FRAME_BYTES / 1e9) / (avg_us / 1e6);
        std::cout << std::left
                  << std::setw(28) << "std::memcpy (完全拷贝)"
                  << std::setw(14) << std::fixed << std::setprecision(1) << avg_us
                  << std::setw(20) << std::fixed << std::setprecision(2) << bw
                  << "\n";
    }

    // 2. 指针交换 (零拷贝)
    {
        double total_us = 0.0;
        uint8_t *a = src;
        uint8_t *b = dst;
        for (int s = 0; s < num_samples; ++s) {
            Timer timer;
            timer.start();
            std::swap(a, b); // 仅交换指针 (各 8 字节)
            total_us += timer.elapsed_us();
        }
        double avg_us = total_us / num_samples;
        std::cout << std::left
                  << std::setw(28) << "指针交换 (零拷贝)"
                  << std::setw(14) << std::fixed << std::setprecision(3) << avg_us
                  << std::setw(20) << "O(1)"
                  << "\n";
    }

    // 3. 带非时态提示模拟的 memcpy (2x 展开拷贝)
    {
        double total_us = 0.0;
        for (int s = 0; s < num_samples; ++s) {
            Timer timer;
            timer.start();
            size_t i = 0;
            // 每次拷贝 32 字节 (相当于 2x __m128i)
            for (; i + 32 <= FRAME_BYTES; i += 32) {
                uint64_t w0, w1, w2, w3;
                std::memcpy(&w0, src + i, 8);
                std::memcpy(&w1, src + i + 8, 8);
                std::memcpy(&w2, src + i + 16, 8);
                std::memcpy(&w3, src + i + 24, 8);
                std::memcpy(dst + i, &w0, 8);
                std::memcpy(dst + i + 8, &w1, 8);
                std::memcpy(dst + i + 16, &w2, 8);
                std::memcpy(dst + i + 24, &w3, 8);
            }
            for (; i < FRAME_BYTES; ++i) dst[i] = src[i];
            total_us += timer.elapsed_us();
        }
        double avg_us = total_us / num_samples;
        double bw = (FRAME_BYTES / 1e9) / (avg_us / 1e6);
        std::cout << std::left
                  << std::setw(28) << "手动 32 字节展开"
                  << std::setw(14) << std::fixed << std::setprecision(1) << avg_us
                  << std::setw(20) << std::fixed << std::setprecision(2) << bw
                  << "\n";
    }

    std::cout << "\n";

    // 4. 相机帧环形缓冲区基准测试
    std::cout << "环形缓冲区 (8 个槽位, " << FRAME_BYTES / (1024.0 * 1024.0)
              << " MB 帧, 1000 次迭代):\n\n";
    std::cout << std::left
              << std::setw(28) << "方法"
              << std::setw(16) << "总耗时(ms)"
              << std::setw(16) << "每帧(us)\n";
    std::cout << std::string(60, '-') << "\n";

    const int ring_iterations = 1000;

    // 使用 memcpy 的环形缓冲区
    {
        CameraRingBuffer rb(ring_slots, FRAME_BYTES);
        Timer timer;
        timer.start();
        for (int i = 0; i < ring_iterations; ++i) {
            // 修改源数据以模拟新帧
            src[0] = static_cast<uint8_t>(i);
            rb.write_copy(src, i * 33000);
        }
        double total_ms = timer.elapsed_ms();
        std::cout << std::left
                  << std::setw(28) << "通过 memcpy 写入"
                  << std::setw(16) << std::fixed << std::setprecision(1) << total_ms
                  << std::setw(16) << std::fixed << std::setprecision(1)
                  << (total_ms * 1000.0 / ring_iterations)
                  << "\n";
    }

    // 使用指针交换的环形缓冲区
    {
        CameraRingBuffer rb(ring_slots, FRAME_BYTES);
        Timer timer;
        timer.start();
        uint8_t *swap_buf = new uint8_t[FRAME_BYTES];
        std::memcpy(swap_buf, src, FRAME_BYTES);
        for (int i = 0; i < ring_iterations; ++i) {
            // 指针交换交换所有权: 旧缓冲区复用于下一帧
            uint8_t *write_ptr = swap_buf;
            write_ptr[0] = static_cast<uint8_t>(i);
            rb.write_swap(write_ptr, i * 33000);
            swap_buf = write_ptr; // 取回缓冲区以供复用
        }
        double total_ms = timer.elapsed_ms();
        std::cout << std::left
                  << std::setw(28) << "通过指针交换写入"
                  << std::setw(16) << std::fixed << std::setprecision(1) << total_ms
                  << std::setw(16) << std::fixed << std::setprecision(4)
                  << (total_ms * 1000.0 / ring_iterations)
                  << "\n";
        delete[] swap_buf;
    }

    delete[] src;
    delete[] dst;

    std::cout << "\n解释:\n"
              << "  - memcpy 每帧拷贝约 6MB; 在高 FPS 下成为瓶颈。\n"
              << "  - 指针交换无论帧大小都是 O(1); 非常适合流水线。\n"
              << "  - 带交换的环形缓冲区: 消费者获取缓冲区的 const 引用;\n"
              << "    生产者复用旧缓冲区，消除分配开销。\n"
              << "  - 在 30 FPS 下，6MB/帧 = 180 MB/s 持续内存带宽。\n"
              << "    在 60 FPS 带 3 个相机 = >1 GB/s，超过许多\n"
              << "    嵌入式平台的内存带宽。零拷贝是必需的。\n";
}
