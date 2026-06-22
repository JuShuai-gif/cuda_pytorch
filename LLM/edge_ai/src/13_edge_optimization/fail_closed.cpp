#include "fail_closed.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

extern volatile long g_sink;

namespace {

class Timer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

void print_header(const std::string &title) {
    std::cout << "\n"
              << std::string(70, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(70, '=') << "\n";
}

// ============================================================================
// 模拟视觉管线的一帧处理
//
// 实际 RK3588 管线:
//   alloc_dma_buffer() → rga_resize() → dma_sync_start() → memcpy() → dma_sync_end()
//
// 每个步骤都可能失败:
//   - alloc: OOM
//   - rga_resize: RGA 硬件错误
//   - dma_sync_start: ioctl 失败（kernel 错误）
//   - memcpy: 实际上不会失败，但此处演示完整性检查
//
// 关键原则: 任何一步失败 → 整帧丢弃（fail-closed），
// 绝不使用 stale data 或跳过错误继续处理。
// ============================================================================
struct FrameResult {
    int frame_id = 0;          // 帧编号
    bool ready = false;        // 帧数据是否有效
    std::string error_step;    // 失败步骤（用于诊断）
    int error_code = 0;        // 错误码
    std::vector<uint8_t> data; // 处理后的帧数据

    // 管线各阶段耗时（用于性能分析）
    double alloc_ms = 0;
    double resize_ms = 0;
    double sync_ms = 0;
    double copy_ms = 0;
};

// 模拟的管线步骤（返回 true=成功, false=失败）
struct PipelineState {
    void *dma_buf = nullptr;
    void *scaled_buf = nullptr;
    int dma_fd = -1;
    int frame_id = 0;
};

bool step_alloc_dma_buffer(PipelineState &state, FrameResult &result) {
    // 模拟 DMA buffer 分配（4MB 帧）
    static std::mt19937 gen(42);
    static std::uniform_real_distribution<double> err_dist(0.0, 1.0);

    // 分配失败概率: 1%（模拟 OOM 场景）
    if (err_dist(gen) < 0.01) {
        result.error_step = "alloc_dma_buffer";
        result.error_code = -1;
        std::cout << "      [FAIL] DMA buffer 分配失败（模拟 OOM）\n";
        return false;
    }

    state.dma_buf = std::aligned_alloc(64, 4 * 1024 * 1024);
    if (!state.dma_buf) {
        result.error_step = "alloc_dma_buffer";
        result.error_code = -12; // ENOMEM
        std::cout << "      [FAIL] aligned_alloc 返回 nullptr\n";
        return false;
    }

    state.dma_fd = 42; // 模拟 fd
    std::cout << "      [PASS] DMA buffer 已分配 (4MB)\n";
    return true;
}

bool step_rga_resize(PipelineState &state, FrameResult &result) {
    static std::mt19937 gen(42);
    static std::uniform_real_distribution<double> err_dist(0.0, 1.0);

    // RGA 错误概率: 3%（模拟硬件偶发错误）
    if (err_dist(gen) < 0.03) {
        result.error_step = "rga_resize";
        result.error_code = -5; // EIO
        std::cout << "      [FAIL] RGA 硬件错误（模拟）\n";
        return false;
    }

    // 模拟 RGA 处理: 填充"处理后的图像数据"
    uint8_t *buf = static_cast<uint8_t *>(state.dma_buf);
    for (size_t i = 0; i < 1024 * 1024; ++i) {
        buf[i] = static_cast<uint8_t>((i + state.frame_id) & 0xFF);
    }

    std::cout << "      [PASS] RGA resize 完成\n";
    return true;
}

bool step_dma_sync_start(PipelineState & /* state */, FrameResult &result) {
    static std::mt19937 gen(42);
    static std::uniform_real_distribution<double> err_dist(0.0, 1.0);

    // DMA_BUF_IOCTL_SYNC 失败概率: 0.5%（模拟 ioctl 失败）
    if (err_dist(gen) < 0.005) {
        result.error_step = "dma_sync_start(START_READ)";
        result.error_code = -1;
        std::cout << "      [FAIL] DMA_BUF_IOCTL_SYNC START_READ 失败！\n";
        std::cout << "             → 设置 origReady=false，不使用可能过期的缓存数据\n";
        return false;
    }

    std::cout << "      [PASS] DMA_BUF_IOCTL_SYNC START_READ（cache invalidate）\n";
    return true;
}

bool step_memcpy_cached(PipelineState &state, FrameResult &result) {
    if (!state.dma_buf) {
        result.error_step = "memcpy_cached";
        result.error_code = -14; // EFAULT
        return false;
    }

    // 模拟 memcpy: 从 DMA buffer 拷贝到 scaled buffer
    // 实际上这里就是 cached 速度的 memcpy（约 0.3ms for 6MB）
    result.data.resize(1024 * 1024);
    std::memcpy(result.data.data(), state.dma_buf, result.data.size());

    std::cout << "      [PASS] CPU memcpy @ cached 速度（~0.3ms for 6MB）\n";
    return true;
}

bool step_dma_sync_end(PipelineState & /* state */, FrameResult &result) {
    static std::mt19937 gen(42);
    static std::uniform_real_distribution<double> err_dist(0.0, 1.0);

    // END_READ 失败概率: 0.1%（极少见但不为零）
    if (err_dist(gen) < 0.001) {
        result.error_step = "dma_sync_end(END_READ)";
        result.error_code = -1;
        std::cout << "      [WARN] DMA_BUF_IOCTL_SYNC END_READ 失败！"
                  << "数据已拷贝，释放 DMA buffer 控制权。\n";
        // 注意: END 失败时数据已经 memcpy 完成，result.ready 仍然可以 true
        // 因为数据已经安全地拷贝到 result.data 中
        return false; // 标记失败但不影响已拷贝的数据
    }

    std::cout << "      [PASS] DMA_BUF_IOCTL_SYNC END_READ（释放 DMA 控制）\n";
    return true;
}

// ============================================================================
// Fail-Closed 管线: 任何步骤失败 → 整帧丢弃
// ============================================================================
FrameResult process_frame_fail_closed(int frame_id) {
    FrameResult result;
    result.frame_id = frame_id;
    PipelineState state;
    state.frame_id = frame_id;

    std::cout << "    ── 帧 #" << frame_id << " Fail-Closed 处理 ──\n";

    // 步骤 1: 分配 DMA buffer
    Timer t;
    t.start();
    if (!step_alloc_dma_buffer(state, result)) {
        result.ready = false;
        result.alloc_ms = t.elapsed_ms();
        std::cout << "      => 决策: origReady=false，丢弃帧 #" << frame_id
                  << "（绝不用不确定数据）\n";
        return result;
    }
    result.alloc_ms = t.elapsed_ms();

    // 步骤 2: RGA 缩放
    t.start();
    if (!step_rga_resize(state, result)) {
        result.ready = false;
        result.resize_ms = t.elapsed_ms();
        std::cout << "      => 决策: origReady=false，丢弃帧 #" << frame_id
                  << "（RGA 数据可能损坏）\n";
        std::free(state.dma_buf);
        return result;
    }
    result.resize_ms = t.elapsed_ms();

    // 步骤 3: DMA_BUF_IOCTL_SYNC START_READ
    t.start();
    if (!step_dma_sync_start(state, result)) {
        result.ready = false;
        result.sync_ms = t.elapsed_ms();
        std::cout << "      => 决策: origReady=false，丢弃帧 #" << frame_id
                  << "（可能读到 cache 过期数据）\n";
        std::free(state.dma_buf);
        return result;
    }
    result.sync_ms = t.elapsed_ms();

    // 步骤 4: CPU memcpy（cached 速度）
    t.start();
    if (!step_memcpy_cached(state, result)) {
        result.ready = false;
        result.copy_ms = t.elapsed_ms();
        std::cout << "      => 决策: origReady=false，丢弃帧 #" << frame_id
                  << "（memcpy 失败）\n";
        std::free(state.dma_buf);
        return result;
    }
    result.copy_ms = t.elapsed_ms();

    // 步骤 5: DMA_BUF_IOCTL_SYNC END_READ
    // END 失败不影响已拷贝的数据（数据已经安全在 result.data 中）
    step_dma_sync_end(state, result);

    // 全部通过 → 帧可用
    result.ready = true;
    result.data[0] = static_cast<uint8_t>(frame_id & 0xFF); // 标记帧ID
    std::cout << "      => 决策: origReady=true，帧 #" << frame_id << " 可用\n";

    std::free(state.dma_buf);
    return result;
}

// ============================================================================
// Fail-Open 管线（危险！仅用于对比演示）
// ============================================================================
FrameResult process_frame_fail_open(int frame_id) {
    FrameResult result;
    result.frame_id = frame_id;
    PipelineState state;
    state.frame_id = frame_id;

    // 预先填充 result.data 为"过期数据"（模拟上一帧残留）
    result.data.resize(1024 * 1024);
    for (auto &b : result.data) {
        b = 0xDE; // 0xDEAD = "stale data"
    }

    std::cout << "    ── 帧 #" << frame_id << " Fail-Open 处理（危险演示）──\n";

    // 尝试分配 → 失败也继续
    state.dma_buf = std::aligned_alloc(64, 4 * 1024 * 1024);
    if (!state.dma_buf) {
        std::cout << "      [FAIL] alloc 失败 → 但继续（危险！用过期数据）\n";
        result.ready = true; // 错误！应该 false
        return result;
    }
    std::cout << "      [PASS] alloc 完成\n";

    // 尝试 RGA → 失败也继续
    static std::mt19937 gen(43);
    static std::uniform_real_distribution<double> err_dist(0.0, 1.0);
    if (err_dist(gen) < 0.03) {
        std::cout << "      [FAIL] RGA 错误 → 但继续（危险！用上一次的旧数据）\n";
        result.ready = true; // 致命错误！
        std::free(state.dma_buf);
        return result;
    }
    std::cout << "      [PASS] RGA 完成\n";

    // 跳过 sync（假设忘了调用或调用失败被忽略）
    std::cout << "      [SKIP] DMA_BUF_IOCTL_SYNC 被跳过 → 可能读到 cache 过期数据\n";

    // 直接 memcpy（数据可能过期的！）
    std::memcpy(result.data.data(), state.dma_buf, result.data.size());
    std::cout << "      [PASS] memcpy 完成（但数据可能过期/损坏）\n";

    result.ready = true;
    std::free(state.dma_buf);
    std::cout << "      => 决策: origReady=true（危险！数据可能不可靠）\n";
    return result;
}

} // namespace

// ============================================================================
// 主演示
// ============================================================================
void demo_fail_closed_pattern() {
    print_header("Fail-Closed 错误处理模式演示");

    constexpr int NUM_FRAMES = 8;

    std::cout << "\n  ╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "  ║  管线步骤:                                                ║\n";
    std::cout << "  ║    1. alloc_dma_buffer()  - 分配 DMA buffer               ║\n";
    std::cout << "  ║    2. rga_resize()        - RGA 硬件缩放                 ║\n";
    std::cout << "  ║    3. dma_sync_start()    - Cache invalidate（START_READ）║\n";
    std::cout << "  ║    4. memcpy()            - CPU cached 拷贝              ║\n";
    std::cout << "  ║    5. dma_sync_end()      - 释放 DMA 控制（END_READ）    ║\n";
    std::cout << "  ╠══════════════════════════════════════════════════════════╣\n";
    std::cout << "  ║  Fail-Closed:  任一步失败 → ready=false → 丢帧          ║\n";
    std::cout << "  ║  Fail-Open:    忽略错误 → ready=true  → 用过期数据！⚠   ║\n";
    std::cout << "  ╚══════════════════════════════════════════════════════════╝\n\n";

    // ---------- Fail-Closed 演示 ----------
    std::cout << "  ╔═══ Fail-Closed 模式 ═══╗\n\n";
    int closed_success = 0;
    int closed_failure = 0;
    std::vector<double> closed_times;

    for (int id = 0; id < NUM_FRAMES; ++id) {
        Timer frame_timer;
        frame_timer.start();

        FrameResult r = process_frame_fail_closed(id);

        if (r.ready) {
            closed_success++;
        } else {
            closed_failure++;
        }
        closed_times.push_back(frame_timer.elapsed_ms());
        std::cout << "\n";
    }

    // ---------- Fail-Open 演示 ----------
    std::cout << "\n  ╔═══ Fail-Open 模式（危险演示） ═══╗\n\n";
    int open_success = 0;
    int open_failure = 0;
    int open_dangerous = 0; // ready=true 但数据可能过期的帧数

    for (int id = 0; id < NUM_FRAMES; ++id) {
        FrameResult r = process_frame_fail_open(id);

        if (r.ready) {
            open_success++;
            // 检查数据是否为 stale（0xDE 标记）
            if (r.data[0] == 0xDE) {
                open_dangerous++;
            }
        } else {
            open_failure++;
        }
        std::cout << "\n";
    }

    // ---------- 总结对比 ----------
    std::cout << "  ╔═══ 对比总结 ═══╗\n\n";
    std::cout << "  " << std::left << std::setw(25) << "指标"
              << std::right << std::setw(15) << "Fail-Closed"
              << std::setw(15) << "Fail-Open" << "\n";
    std::cout << "  " << std::string(55, '-') << "\n";
    std::cout << "  " << std::left << std::setw(25) << "总帧数"
              << std::right << std::setw(15) << NUM_FRAMES
              << std::setw(15) << NUM_FRAMES << "\n";
    std::cout << "  " << std::left << std::setw(25) << "成功帧(ready=true)"
              << std::right << std::setw(15) << closed_success
              << std::setw(15) << open_success << "\n";
    std::cout << "  " << std::left << std::setw(25) << "丢弃帧(ready=false)"
              << std::right << std::setw(15) << closed_failure
              << std::setw(15) << open_failure << "\n";
    std::cout << "  " << std::left << std::setw(25) << "危险帧(过期数据)"
              << std::right << std::setw(15) << "0 ✓"
              << std::setw(15) << (std::to_string(open_dangerous) + " ⚠") << "\n";
    std::cout << "  " << std::left << std::setw(25) << "数据可靠性"
              << std::right << std::setw(15) << "100% ✓"
              << std::setw(15) << "不可靠 ⚠" << "\n";

    if (closed_times.size() > 0) {
        double avg_ms = std::accumulate(closed_times.begin(),
                                        closed_times.end(), 0.0)
                        / closed_times.size();
        std::cout << "  " << std::left << std::setw(25) << "平均处理时间"
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(15) << avg_ms << " ms"
                  << std::setw(15) << "—\n";
    }

    std::cout << "\n  => 核心原则:\n";
    std::cout << "     1. DMA_BUF_IOCTL_SYNC START 失败 → 绝不读取 buffer\n";
    std::cout << "     2. 任何硬件操作返回错误 → 设置 ready=false，不使用数据\n";
    std::cout << "     3. 绝不用 virtual-address RGA 作为 fallback（RK3588 上会 freeze）\n";
    std::cout << "     4. 宁可丢帧，不可用不确定数据——机器人误动作代价远大于丢一帧\n";
    std::cout << "     5. 每一帧的 ready flag 是下游模块做决策的唯一依据\n";

    // 输出决策树示意图
    std::cout << "\n  ── 决策树示意 ──\n\n";
    std::cout << "  alloc_dma_buffer()\n";
    std::cout << "    ├─ 成功 → rga_resize()\n";
    std::cout << "    │          ├─ 成功 → dma_sync_start(START_READ)\n";
    std::cout << "    │          │          ├─ 成功 → memcpy() @ cached 速度\n";
    std::cout << "    │          │          │          └─ ready=true ✓\n";
    std::cout << "    │          │          └─ 失败 → ready=false（拒绝读，数据可能过期）\n";
    std::cout << "    │          └─ 失败 → ready=false（RGA 输出损坏，拒绝使用）\n";
    std::cout << "    └─ 失败 → ready=false（无 buffer 可用，拒绝分配）\n\n";
    std::cout << "  注意: Fail-Open 在任何失败后仍设置 ready=true——"
              << "这是导致幽灵 bug 的根源。\n";
}
