#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>

// 单生产者单消费者无锁环形缓冲区
// 用于 ROS2 实时线程与非实时线程之间的数据传输
// 适用于摄像头图像、检测结果等传感器数据
template <typename T, size_t N>
class SPSCRingBuffer {
    static_assert(N >= 2, "缓冲区容量至少为 2");
    static_assert((N & (N - 1)) == 0, "缓冲区容量必须是 2 的幂 (以便使用位掩码替代取模)");

public:
    SPSCRingBuffer() : write_idx_(0), read_idx_(0) {
        // 预分配内存：在 RT 路径外完成，避免 RT 线程调用 malloc
        for (size_t i = 0; i < N; ++i) {
            new (&slots_[i]) T();
        }
    }

    ~SPSCRingBuffer() {
        for (size_t i = 0; i < N; ++i) {
            slots_[i].~T();
        }
    }

    SPSCRingBuffer(const SPSCRingBuffer &) = delete;
    SPSCRingBuffer &operator=(const SPSCRingBuffer &) = delete;

    // 非阻塞 push，缓冲区满则返回 false
    // 生产者 (Non-RT 线程) 调用
    bool try_push(const T &item) {
        const size_t current_write = write_idx_.load(std::memory_order_relaxed);
        const size_t next_write = (current_write + 1) & mask_;

        // 如果 next_write 追上 read_idx，说明缓冲区已满
        if (next_write == read_idx_cached_) {
            read_idx_cached_ = read_idx_.load(std::memory_order_acquire);
            if (next_write == read_idx_cached_) {
                return false; // 缓冲区满
            }
        }

        // 先写数据，再更新写索引 (release 语义确保数据对消费者可见)
        slots_[current_write] = item;
        write_idx_.store(next_write, std::memory_order_release);
        return true;
    }

    // 覆盖式 push：缓冲区满时覆盖最旧的数据
    // 适用于传感器数据的 latest-is-best 语义
    // 生产者 (Non-RT 线程) 调用
    void push_overwrite(const T &item) {
        const size_t current_write = write_idx_.load(std::memory_order_relaxed);
        const size_t next_write = (current_write + 1) & mask_;

        slots_[current_write] = item;
        write_idx_.store(next_write, std::memory_order_release);

        // 如果覆盖了未消费的数据，推进读索引
        if (next_write == read_idx_.load(std::memory_order_acquire)) {
            read_idx_.store((next_write + 1) & mask_, std::memory_order_release);
        }
    }

    // 非阻塞 pop，缓冲区空则返回 false
    // 消费者 (RT 线程) 调用
    bool try_pop(T &item) {
        const size_t current_read = read_idx_.load(std::memory_order_relaxed);

        // 如果读索引追上写索引，说明缓冲区为空
        if (current_read == write_idx_cached_) {
            write_idx_cached_ = write_idx_.load(std::memory_order_acquire);
            if (current_read == write_idx_cached_) {
                return false; // 缓冲区空
            }
        }

        // 先读数据，再更新读索引 (acquire 语义确保读到生产者写入的完整数据)
        item = slots_[current_read];
        read_idx_.store((current_read + 1) & mask_, std::memory_order_release);
        return true;
    }

    // 获取最新数据，但不弹出 (peek)
    // 消费者 (RT 线程) 调用
    bool peek_latest(T &item) {
        const size_t current_write = write_idx_.load(std::memory_order_acquire);
        const size_t current_read = read_idx_.load(std::memory_order_relaxed);

        if (current_write == current_read) {
            return false; // 缓冲区空
        }

        // 返回最新写入的条目 (即 write_idx 的前一个位置)
        item = slots_[(current_write - 1) & mask_];
        return true;
    }

    // 清空缓冲区，将所有未消费的数据丢弃
    // 用于快速同步，例如只在 RT 线程需要最新数据时
    void drain() {
        read_idx_.store(write_idx_.load(std::memory_order_acquire),
                        std::memory_order_release);
    }

    bool empty() const {
        return read_idx_.load(std::memory_order_acquire)
               == write_idx_.load(std::memory_order_acquire);
    }

    bool full() const {
        const size_t next_write = (write_idx_.load(std::memory_order_relaxed) + 1) & mask_;
        return next_write == read_idx_.load(std::memory_order_acquire);
    }

    size_t capacity() const {
        return N;
    }

    // 获取当前缓冲区中的条目数
    size_t size() const {
        const size_t w = write_idx_.load(std::memory_order_acquire);
        const size_t r = read_idx_.load(std::memory_order_acquire);
        if (w >= r) return w - r;
        return N - r + w;
    }

private:
    static constexpr size_t CACHE_LINE_SIZE = 64;

    // 位掩码替代取模运算 (N 必须是 2 的幂)
    static constexpr size_t mask_ = N - 1;

    // 数据槽位数组
    T slots_[N];

    // === 缓存行填充，防止伪共享 ===
    // write_idx 和 read_idx 放在不同的缓存行上
    // 这样生产者更新 write_idx 不会导致消费者核心上的 read_idx 缓存行失效

    // 写入索引 (生产者拥有)
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> write_idx_;

    // 读取索引 (消费者拥有)
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> read_idx_;

    // === 第二个缓存行填充: 内部缓存的索引 ===
    // 生产者缓存 read_idx 以减少跨核心原子读取
    alignas(CACHE_LINE_SIZE) size_t read_idx_cached_ = 0;

    // 消费者缓存 write_idx 以减少跨核心原子读取
    alignas(CACHE_LINE_SIZE) size_t write_idx_cached_ = 0;
};

// 大缓冲区特化：用于摄像头图像 (1920×1080×3 字节)
template <size_t N>
class ImageRingBuffer {
    static_assert(N >= 2, "缓冲区容量至少为 2");
    static_assert((N & (N - 1)) == 0, "缓冲区容量必须是 2 的幂");

public:
    static constexpr size_t IMAGE_WIDTH = 1920;
    static constexpr size_t IMAGE_HEIGHT = 1080;
    static constexpr size_t IMAGE_CHANNELS = 3;
    static constexpr size_t IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS;

    ImageRingBuffer() : write_idx_(0), read_idx_(0) {
    }

    ImageRingBuffer(const ImageRingBuffer &) = delete;
    ImageRingBuffer &operator=(const ImageRingBuffer &) = delete;

    // 获取可写入的缓冲区指针 (生产者调用)
    unsigned char *get_write_buffer(size_t idx) {
        return buffers_[idx & mask_];
    }

    // 通知写入完成
    void commit_write() {
        const size_t current = write_idx_.load(std::memory_order_relaxed);
        write_idx_.store(current + 1, std::memory_order_release);
    }

    // 获取可读取的缓冲区指针 (消费者调用)
    // 返回读取索引，失败返回 -1
    int get_read_index() {
        const size_t current = read_idx_.load(std::memory_order_relaxed);
        const size_t write = write_idx_.load(std::memory_order_acquire);
        if (current == write) return -1;
        return static_cast<int>(current & mask_);
    }

    // 通知读取完成
    void commit_read() {
        const size_t current = read_idx_.load(std::memory_order_relaxed);
        read_idx_.store(current + 1, std::memory_order_release);
    }

    bool empty() const {
        return read_idx_.load(std::memory_order_acquire)
               == write_idx_.load(std::memory_order_acquire);
    }

private:
    static constexpr size_t CACHE_LINE_SIZE = 64;
    static constexpr size_t mask_ = N - 1;

    unsigned char buffers_[N][IMAGE_SIZE];

    alignas(CACHE_LINE_SIZE) std::atomic<size_t> write_idx_;
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> read_idx_;
};
