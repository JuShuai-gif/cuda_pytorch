#include "task_executor.h"
#include "double_buffer.h"
#include "pipeline.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <atomic>
#include <vector>
#include <chrono>

// ============================================================================
// 机器人流水线：带真实数据处理的 7 节点 DAG 任务图
// ============================================================================
static void run_robot_pipeline(int num_threads) {
    std::fprintf(stderr, "\n============================================================\n");
    std::fprintf(stderr, "  机器人处理任务图 (DAG)\n");
    std::fprintf(stderr, "  7 个节点, %d 个线程\n", num_threads);
    std::fprintf(stderr, "============================================================\n");

    TaskGraphExecutor executor(num_threads);
    PipelineContext &ctx = executor.context();

    // 节点 0：sensor_capture（无依赖）
    executor.add_node({"sensor_capture", {}, [&ctx]() { node_sensor_capture(ctx); }});

    // 节点 1：image_preprocess（依赖 sensor_capture）
    executor.add_node({"image_preprocess", {0}, [&ctx]() { node_image_preprocess(ctx); }});

    // 节点 2：lidar_preprocess（依赖 sensor_capture，与 image_preprocess 并行运行）
    executor.add_node({"lidar_preprocess", {0}, [&ctx]() { node_lidar_preprocess(ctx); }});

    // 节点 3：detection（同时依赖 image_preprocess 和 lidar_preprocess）
    executor.add_node({"detection", {1, 2}, [&ctx]() { node_detection(ctx); }});

    // 节点 4：tracking（依赖 detection）
    executor.add_node({"tracking", {3}, [&ctx]() { node_tracking(ctx); }});

    // 节点 5：planning（依赖 tracking）
    executor.add_node({"planning", {4}, [&ctx]() { node_planning(ctx); }});

    // 节点 6：control（依赖 planning）
    executor.add_node({"control", {5}, [&ctx]() { node_control(ctx); }});

    executor.build_graph();

    Timer total_timer;
    total_timer.start();
    executor.execute();
    double wall_time_us = total_timer.elapsed_us();

    // 打印结果
    std::fprintf(stderr, "\n%-20s %-30s %12s\n", "节点", "依赖", "耗时 (us)");
    std::fprintf(stderr, "%.70s\n",
                 "----------------------------------------------------------------------");

    double total_seq = 0.0;
    for (const auto &node : executor.nodes()) {
        std::string dep_str;
        for (size_t i = 0; i < node.dependencies.size(); ++i) {
            if (i > 0) dep_str += ", ";
            dep_str += executor.nodes()[node.dependencies[i]].name;
        }
        if (dep_str.empty()) dep_str = "(无)";
        std::fprintf(stderr, "%-20s %-30s %12.2f\n",
                     node.name.c_str(), dep_str.c_str(), node.elapsed_us);
        total_seq += node.elapsed_us;
    }
    std::fprintf(stderr, "%.70s\n",
                 "----------------------------------------------------------------------");
    std::fprintf(stderr, "%-20s %-30s %12.2f\n", "合计（顺序执行）", "", total_seq);
    std::fprintf(stderr, "\n并行总耗时:  %.2f us (%.3f ms)\n",
                 wall_time_us, wall_time_us / 1000.0);
    std::fprintf(stderr, "顺序耗时: %.2f us (%.3f ms)\n",
                 total_seq, total_seq / 1000.0);
    if (wall_time_us > 0.0) {
        std::fprintf(stderr, "加速比:             %.2fx\n",
                     total_seq / wall_time_us);
    }

    // 控制输出摘要
    std::fprintf(stderr, "\n控制输出:\n");
    std::fprintf(stderr, "  油门: %.3f  刹车: %.3f  转向: %.3f\n",
                 ctx.control_cmd.throttle,
                 ctx.control_cmd.brake,
                 ctx.control_cmd.steering);
    std::fprintf(stderr, "  规划路径长度: %zu 个航点\n",
                 ctx.trajectory.path.size());
    std::fprintf(stderr, "  检测: %zu 个框, %zu 个跟踪\n",
                 ctx.detections.boxes.size(),
                 ctx.tracking_result.tracks.size());

    executor.write_profile_json("task_graph_profile.json", num_threads, wall_time_us);
    std::fprintf(stderr, "\n输出已写入 task_graph_profile.json\n");
}

// ============================================================================
// 双缓冲演示：真实传感器数据
// ============================================================================
static void run_double_buffer_demo() {
    std::fprintf(stderr, "\n============================================================\n");
    std::fprintf(stderr, "  双缓冲：传感器采集 + 处理\n");
    std::fprintf(stderr, "============================================================\n");

    const int num_frames = 5;
    DoubleBuffer<PipelineContext> double_buffer;

    std::atomic<bool> done{false};

    // 生产者：传感器采集线程
    auto producer = [&]() {
        for (int i = 0; i < num_frames; ++i) {
            int buf_idx = double_buffer.producer_acquire();
            PipelineContext &ctx = double_buffer.buffer(buf_idx);

            // 生成传感器数据
            node_sensor_capture(ctx);

            auto ts = std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count();
            std::fprintf(stderr, "  [生产者] 第 %d 帧: 相机 %dx%d, LiDAR %zu 个点 (ts=%ld)\n",
                         i,
                         ctx.camera_image.width, ctx.camera_image.height,
                         ctx.point_cloud.points.size(), ts);

            double_buffer.producer_release(buf_idx);
            double_buffer.producer_swap();
        }
        double_buffer.producer_done();
        done = true;
    };

    // 消费者：图像预处理线程
    std::thread prod_thread(producer);

    for (int i = 0; i < num_frames; ++i) {
        int buf_idx = double_buffer.consumer_acquire();
        if (buf_idx < 0) break;

        PipelineContext &ctx = double_buffer.buffer(buf_idx);

        // 处理：图像预处理
        node_image_preprocess(ctx);

        std::fprintf(stderr, "  [消费者] 第 %d 帧: 预处理至 %dx%d, 范围 [%.3f, %.3f]\n",
                     i, ctx.preprocessed_image.width, ctx.preprocessed_image.height,
                     *std::min_element(ctx.preprocessed_image.data.begin(),
                                       ctx.preprocessed_image.data.end()),
                     *std::max_element(ctx.preprocessed_image.data.begin(),
                                       ctx.preprocessed_image.data.end()));

        double_buffer.consumer_release(buf_idx);
    }

    prod_thread.join();
}

// ============================================================================
// 流水线并行演示：有界队列
// ============================================================================
static void run_pipeline_demo() {
    std::fprintf(stderr, "\n============================================================\n");
    std::fprintf(stderr, "  流水线并行：3 阶段机器人处理\n");
    std::fprintf(stderr, "============================================================\n");

    const int num_frames = 8;

    BoundedQueue q0(2), q1(2), q2(2), q3(2);

    auto source = [&]() {
        for (int i = 0; i < num_frames; ++i) q0.push(i);
        q0.set_done();
    };

    // 阶段 0：传感器采集
    auto stage0 = [&](PipelineContext &shared_ctx) {
        int fid;
        while (q0.pop(fid)) {
            node_sensor_capture(shared_ctx);
            q1.push(fid);
        }
        q1.set_done();
    };

    // 阶段 1：图像预处理
    auto stage1 = [&](PipelineContext &shared_ctx) {
        int fid;
        while (q1.pop(fid)) {
            node_image_preprocess(shared_ctx);
            q2.push(fid);
        }
        q2.set_done();
    };

    // 阶段 2：检测 + 跟踪
    auto stage2 = [&](PipelineContext &shared_ctx) {
        int fid;
        while (q2.pop(fid)) {
            node_lidar_preprocess(shared_ctx);
            node_detection(shared_ctx);
            node_tracking(shared_ctx);
            q3.push(fid);
        }
        q3.set_done();
    };

    auto sink = [&]() {
        int fid;
        while (q3.pop(fid)) {
            std::fprintf(stderr, "  第 %d 帧: %zu 个检测, %zu 个跟踪\n",
                         fid,
                         0UL, // ctx 无法从 sink 访问，仅统计帧数
                         0UL);
        }
    };

    PipelineContext ctx;
    Timer pipeline_timer;
    pipeline_timer.start();

    std::thread src_th(source);
    std::thread s0_th(stage0, std::ref(ctx));
    std::thread s1_th(stage1, std::ref(ctx));
    std::thread s2_th(stage2, std::ref(ctx));
    std::thread sink_th(sink);

    src_th.join();
    s0_th.join();
    s1_th.join();
    s2_th.join();
    sink_th.join();

    double elapsed = pipeline_timer.elapsed_ms();
    std::fprintf(stderr, "\n  3 阶段流水线: %d 帧, %.1f ms (%.1f FPS)\n",
                 num_frames, elapsed, num_frames / elapsed * 1000.0);
}

// ============================================================================
// 主函数
// ============================================================================
int main(int argc, char *argv[]) {
    int num_threads = 4;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::fprintf(stderr, "用法: %s [--threads N]\n", argv[0]);
            return 0;
        }
    }

    std::fprintf(stderr, "============================================================\n");
    std::fprintf(stderr, "  机器人任务图运行时 & 流水线并行\n");
    std::fprintf(stderr, "============================================================\n");

    // 主演示：机器人 DAG 任务图（7 个节点，真实处理）
    run_robot_pipeline(num_threads);

    // 附加演示：双缓冲和流水线并行
    run_double_buffer_demo();
    run_pipeline_demo();

    std::fprintf(stderr, "\n所有演示已完成。\n");
    return 0;
}
