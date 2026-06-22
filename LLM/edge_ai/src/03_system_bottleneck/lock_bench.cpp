#include "lock_bench.h"
#include "timer.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <random>
#include <cstdint>

extern void print_header(const std::string &title);

// ============================================================================
// 边界框 IoU 计算
// ============================================================================
static float nms_iou(const DetectionBoxNMS &a, const DetectionBoxNMS &b) {
    float ax1 = a.x, ay1 = a.y, ax2 = a.x + a.w, ay2 = a.y + a.h;
    float bx1 = b.x, by1 = b.y, bx2 = b.x + b.w, by2 = b.y + b.h;
    float ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1);
    float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
    float iw = std::max(0.0f, ix2 - ix1), ih = std::max(0.0f, iy2 - iy1);
    float inter = iw * ih;
    float area_a = a.w * a.h, area_b = b.w * b.h;
    float uni = area_a + area_b - inter;
    return (uni > 0.0f) ? inter / uni : 0.0f;
}

// ============================================================================
// 简单自旋锁
// ============================================================================
class SpinLock {
public:
    void lock() {
        while (flag_.test_and_set(std::memory_order_acquire)) { /* 自旋 */
        }
    }
    void unlock() {
        flag_.clear(std::memory_order_release);
    }

private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

// ============================================================================
// NMS 处理: 抑制与给定框 IoU > 阈值的框。
// 用作临界区工作负载。
// ============================================================================
static void suppress_boxes(DetectionBoxNMS *boxes, size_t count,
                           size_t keep_idx, float iou_threshold) {
    const DetectionBoxNMS &kept = boxes[keep_idx];
    for (size_t j = keep_idx + 1; j < count; ++j) {
        if (boxes[j].suppressed) continue;
        if (nms_iou(kept, boxes[j]) > iou_threshold) {
            boxes[j].suppressed = true;
        }
    }
}

// ============================================================================
// 演示 2: 锁竞争 - 在竞争锁 vs 无锁情况下运行 NMS
// ============================================================================
void demo_lock_contention() {
    print_header("演示 2: 锁竞争 (检测框上的 NMS)");

    const int num_threads = 4;
    const int boxes_per_thread = 200;
    const int total_boxes = num_threads * boxes_per_thread;
    const int num_rounds = 500;
    const float iou_threshold = 0.5f;

    std::cout << "\n在 " << total_boxes << " 个检测框上运行 NMS，"
              << num_rounds << " 轮, " << num_threads << " 个线程:\n\n";
    std::cout << std::left
              << std::setw(18) << "方法"
              << std::setw(14) << "耗时(ms)"
              << std::setw(18) << "保留框数\n";
    std::cout << std::string(50, '-') << "\n";

    // 生成基础随机框集合
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> pos_dist(0.0f, 640.0f);
    std::uniform_real_distribution<float> size_dist(10.0f, 200.0f);
    std::uniform_real_distribution<float> conf_dist(0.1f, 1.0f);

    std::vector<DetectionBoxNMS> base_boxes(total_boxes);
    for (int i = 0; i < total_boxes; ++i) {
        base_boxes[i].x = pos_dist(rng);
        base_boxes[i].y = pos_dist(rng);
        base_boxes[i].w = size_dist(rng);
        base_boxes[i].h = size_dist(rng);
        base_boxes[i].confidence = conf_dist(rng);
        base_boxes[i].class_id = 0;
        base_boxes[i].suppressed = false;
    }

    // 自旋锁测试
    {
        SpinLock sl;
        Timer timer;
        timer.start();
        for (int r = 0; r < num_rounds; ++r) {
            std::vector<DetectionBoxNMS> boxes = base_boxes;
            size_t kept_count = 0;

            // 按置信度降序排序
            std::sort(boxes.begin(), boxes.end(),
                      [](const DetectionBoxNMS &a, const DetectionBoxNMS &b) {
                          return a.confidence > b.confidence;
                      });

            // 使用自旋锁处理
            std::vector<std::thread> threads;
            for (int t = 0; t < num_threads; ++t) {
                threads.emplace_back([&, t]() {
                    int start = t * boxes_per_thread;
                    int end = start + boxes_per_thread;
                    for (int i = start; i < end; ++i) {
                        sl.lock();
                        if (!boxes[i].suppressed) {
                            suppress_boxes(boxes.data(), total_boxes, i, iou_threshold);
                        }
                        sl.unlock();
                    }
                });
            }
            for (auto &th : threads) th.join();

            for (const auto &b : boxes) {
                if (!b.suppressed) ++kept_count;
            }
        }
        double elapsed = timer.elapsed_ms();
        std::cout << std::left
                  << std::setw(18) << "自旋锁"
                  << std::setw(14) << std::fixed << std::setprecision(2) << elapsed
                  << std::setw(18) << "~每轮\n";
    }

    // 互斥锁测试
    {
        std::mutex mtx;
        Timer timer;
        timer.start();
        for (int r = 0; r < num_rounds; ++r) {
            std::vector<DetectionBoxNMS> boxes = base_boxes;
            std::sort(boxes.begin(), boxes.end(),
                      [](const DetectionBoxNMS &a, const DetectionBoxNMS &b) {
                          return a.confidence > b.confidence;
                      });

            std::vector<std::thread> threads;
            for (int t = 0; t < num_threads; ++t) {
                threads.emplace_back([&, t]() {
                    int start = t * boxes_per_thread;
                    int end = start + boxes_per_thread;
                    for (int i = start; i < end; ++i) {
                        {
                            std::lock_guard<std::mutex> lock(mtx);
                            if (!boxes[i].suppressed) {
                                suppress_boxes(boxes.data(), total_boxes, i, iou_threshold);
                            }
                        }
                    }
                });
            }
            for (auto &th : threads) th.join();
        }
        double elapsed = timer.elapsed_ms();
        std::cout << std::left
                  << std::setw(18) << "std::mutex"
                  << std::setw(14) << std::fixed << std::setprecision(2) << elapsed
                  << std::setw(18) << "~每轮\n";
    }

    // 无锁: 每个线程在独立分区上工作，然后合并
    // 这是 NMS 的实用无锁方案
    {
        Timer timer;
        timer.start();
        for (int r = 0; r < num_rounds; ++r) {
            std::vector<DetectionBoxNMS> boxes = base_boxes;
            std::sort(boxes.begin(), boxes.end(),
                      [](const DetectionBoxNMS &a, const DetectionBoxNMS &b) {
                          return a.confidence > b.confidence;
                      });

            // 基于分区的无锁: 每个线程获取一个不相交的子集
            // 独立处理高置信度框
            std::vector<std::thread> threads;
            std::vector<std::vector<size_t>> kept_per_thread(num_threads);

            for (int t = 0; t < num_threads; ++t) {
                threads.emplace_back([&, t]() {
                    // 每个线程取每隔 N 个框作为自己的锚框集
                    for (int i = t; i < total_boxes; i += num_threads) {
                        if (!boxes[i].suppressed) {
                            // 仅抑制置信度更低的框 (j > i)
                            for (size_t j = i + 1; j < static_cast<size_t>(total_boxes); ++j) {
                                if (boxes[j].suppressed) continue;
                                if (nms_iou(boxes[i], boxes[j]) > iou_threshold) {
                                    boxes[j].suppressed = true;
                                }
                            }
                        }
                    }
                });
            }
            for (auto &th : threads) th.join();
        }
        double elapsed = timer.elapsed_ms();
        std::cout << std::left
                  << std::setw(18) << "无锁 (分区)"
                  << std::setw(14) << std::fixed << std::setprecision(2) << elapsed
                  << std::setw(18) << "~每轮\n";
    }

    std::cout << "\n分析:\n"
              << "  - 自旋锁: 等待时消耗 CPU 周期; 适用于极短的临界区。\n"
              << "  - std::mutex: 竞争时由操作系统辅助休眠; 适用于较长的临界区。\n"
              << "  - 无锁 (分区): 线程处理不相交的锚框集，\n"
              << "    以部分冗余的 IoU 检查为代价，完全消除竞争。\n"
              << "    随线程数增加几乎线性扩展。\n";
}
