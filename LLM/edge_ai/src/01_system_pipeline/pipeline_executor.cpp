#include "pipeline_executor.h"
#include "pipeline_stage.h"

#include <cstdio>
#include <vector>

// ============================================================================
// 顺序执行：一次只处理一帧，当前帧的三个阶段全部完成后才处理下一帧
// 延迟 = sum(感知,规划,控制)，吞吐 = 1/延迟
// ============================================================================

void run_sequential(const PipelineConfig &cfg, LatencyStats &stats) {
    std::vector<int64_t> e2e_latencies;
    e2e_latencies.reserve(cfg.num_frames);

    for (int frame_id = 0; frame_id < cfg.num_frames; frame_id++) {
        int64_t frame_start_ns = now_ns();

        // 生成传感器数据
        PipelineSensorData sensor;
        sensor.frame_id = frame_id;
        sensor.timestamp_ns = frame_start_ns;
        sensor.camera_image = generate_camera_image(1920, 1080, 3);
        sensor.point_cloud = generate_lidar_point_cloud(100000, 64);

        // 阶段 1: 感知
        int64_t preprocess_ns = 0, lidar_ns = 0, detection_ns = 0;
        Detections detections = run_perception(sensor,
                                               &preprocess_ns, &lidar_ns, &detection_ns);
        int64_t percep_total = preprocess_ns + lidar_ns + detection_ns;
        stats.record("perception", percep_total);
        stats.record("perception.image_preprocess", preprocess_ns);
        stats.record("perception.lidar_preprocess", lidar_ns);
        stats.record("perception.detection", detection_ns);

        PipelinePerceptionOut percep_out;
        percep_out.frame_id = frame_id;
        percep_out.timestamp_ns = now_ns();
        percep_out.e2e_start_ns = frame_start_ns;
        percep_out.detections = detections;
        percep_out.perception_time_ns = percep_total;

        // 阶段 2: 规划
        PipelinePlanningOut plan_out = run_planning(percep_out);
        stats.record("planning", plan_out.planning_time_ns);

        // 阶段 3: 控制
        PipelineControlOut ctrl_out = run_control(plan_out);
        stats.record("control", ctrl_out.control_time_ns);

        // 端到端
        int64_t frame_end_ns = now_ns();
        int64_t e2e_ns = frame_end_ns - frame_start_ns;
        stats.record("end_to_end", e2e_ns);
        e2e_latencies.push_back(e2e_ns);

        if (cfg.verbose && (frame_id % cfg.stats_interval_frames == 0)) {
            std::fprintf(stderr,
                         "Frame %4d | P:%7ld ns | L:%7ld ns | C:%7ld ns | E2E:%8ld ns\n",
                         frame_id, percep_total, plan_out.planning_time_ns,
                         ctrl_out.control_time_ns, e2e_ns);
        }
    }
}

// ============================================================================
// PipelinedExecutor 实现：三个工作线程并发运行，同时处理不同帧
// 数据流：main线程生产传感器帧 → perception_worker → planning_worker → control_worker
// 延迟 = sum(感知,规划,控制)，吞吐 = 1/max(感知,规划,控制)
// ============================================================================

PipelinedExecutor::PipelinedExecutor(const PipelineConfig &cfg,
                                     LatencyStats &stats) : cfg_(cfg), stats_(stats) {
}

void PipelinedExecutor::run() {
    // 启动三个工作线程：分别负责感知、规划、控制
    std::thread perception_th(&PipelinedExecutor::perception_worker, this);
    std::thread planning_th(&PipelinedExecutor::planning_worker, this);
    std::thread control_th(&PipelinedExecutor::control_worker, this);

    // main 线程充当"传感器数据生产者和控制指令消费者"：逐帧生成传感器数据，推入输入队列
    // 向流水线注入帧数据
    for (int frame_id = 0; frame_id < cfg_.num_frames; frame_id++) {
        PipelineSensorData sensor;
        sensor.frame_id = frame_id;
        sensor.timestamp_ns = now_ns();
        sensor.camera_image = generate_camera_image(1920, 1080, 3);
        sensor.point_cloud = generate_lidar_point_cloud(100000, 64);
        {
            std::lock_guard<std::mutex> lock(in_mutex_);
            input_queue_.push(sensor);
        }
        in_cv_.notify_one(); // 唤醒 perception_worker：有新帧可处理
    }

    // 所有帧已推入完毕，通知所有工作线程可以收尾退出了
    stop_ = true;
    in_cv_.notify_all(); // 唤醒所有可能在等待的消费者，让它们检查 stop_ 标志

    // 等待三个工作线程处理完所有剩余帧后退出
    perception_th.join();
    planning_th.join();
    control_th.join();
}

void PipelinedExecutor::perception_worker() {
    while (true) {
        PipelineSensorData sensor;
        bool has_frame = false;
        {
            std::unique_lock<std::mutex> lock(in_mutex_);
            // 等待条件：input_queue_ 非空（有帧可处理）或 stop_=true（收到退出信号）
            // cv.wait 在等待时会释放 lock，被唤醒后重新获取 lock，避免了持锁休眠的活锁
            in_cv_.wait(lock, [this]() {
                return !input_queue_.empty() || stop_;
            });
            // 唤醒后发现队列已空且 stop_=true → 所有帧已处理完，退出循环
            if (input_queue_.empty() && stop_) break;
            if (!input_queue_.empty()) {
                sensor = input_queue_.front();
                input_queue_.pop();
                has_frame = true;
            }
        }
        if (!has_frame) continue;

        // ===== 实际执行感知阶段 =====
        int64_t e2e_start = now_ns();
        int64_t preprocess_ns = 0, lidar_ns = 0, detection_ns = 0;
        Detections detections = run_perception(sensor,
                                               &preprocess_ns, &lidar_ns,
                                               &detection_ns);

        int64_t percep_total = preprocess_ns + lidar_ns + detection_ns;
        stats_.record("perception", percep_total);
        stats_.record("perception.image_preprocess", preprocess_ns);
        stats_.record("perception.lidar_preprocess", lidar_ns);
        stats_.record("perception.detection", detection_ns);

        PipelinePerceptionOut out;
        out.frame_id = sensor.frame_id;
        out.timestamp_ns = now_ns();
        out.e2e_start_ns = e2e_start;
        out.detections = detections;
        out.perception_time_ns = percep_total;

        // 感知结果推入中间队列，唤醒 planning_worker
        {
            std::lock_guard<std::mutex> lock(pq_mutex_);
            perception_queue_.push(out);
        }
        pq_cv_.notify_one();
    }
}

void PipelinedExecutor::planning_worker() {
    while (true) {
        PipelinePerceptionOut percep;
        bool has_data = false;
        {
            std::unique_lock<std::mutex> lock(pq_mutex_);
            // 等待条件：perception_queue_ 非空（有感知结果可消费），或上游全部清空+stop
            // stop_ && input_queue_.empty() && perception_queue_.empty()
            //   → main 已停止生产，perception_worker 已消费完所有帧，当前队列也已空
            //   → 说明不会再有新的感知结果产生，可以安全退出
            pq_cv_.wait(lock, [this]() {
                return !perception_queue_.empty() || (stop_ && input_queue_.empty() && perception_queue_.empty());
            });
            if (perception_queue_.empty()) {
                // 被唤醒但队列仍为空（是 stop 信号而非数据信号），二次确认后退出
                std::lock_guard<std::mutex> in_lock(in_mutex_);
                if (stop_ && input_queue_.empty() && perception_queue_.empty())
                    break;
                continue;
            }
            percep = perception_queue_.front();
            perception_queue_.pop();
            has_data = true;
        }
        if (!has_data) continue;

        // ===== 实际执行规划阶段 =====
        PipelinePlanningOut plan = run_planning(percep);
        stats_.record("planning", plan.planning_time_ns);

        // 规划结果推入输出队列，唤醒 control_worker
        {
            std::lock_guard<std::mutex> lock(lq_mutex_);
            planning_queue_.push(plan);
        }
        lq_cv_.notify_one();
    }
}

void PipelinedExecutor::control_worker() {
    while (true) {
        PipelinePlanningOut plan;
        bool has_data = false;
        {
            std::unique_lock<std::mutex> lock(lq_mutex_);
            // 等待条件：planning_queue_ 非空，或流水线中所有上游队列全部清空+stop
            // 需要检查所有三个队列(input、perception、planning)均已空且stop，
            // 确保上游不会再产生任何新数据，控制阶段是最后一级，退出条件最严格
            lq_cv_.wait(lock, [this]() {
                return !planning_queue_.empty() || (stop_ && input_queue_.empty() && perception_queue_.empty() && planning_queue_.empty());
            });
            if (planning_queue_.empty()) {
                // 再次加锁确认所有上游队列已空，防止 TOCTOU 竞态
                std::lock_guard<std::mutex> in_lock(in_mutex_);
                std::lock_guard<std::mutex> pq_lock(pq_mutex_);
                if (stop_ && input_queue_.empty() && perception_queue_.empty() && planning_queue_.empty())
                    break;
                continue;
            }
            plan = planning_queue_.front();
            planning_queue_.pop();
            has_data = true;
        }
        if (!has_data) continue;

        // ===== 实际执行控制阶段 =====
        PipelineControlOut ctrl = run_control(plan);
        stats_.record("control", ctrl.control_time_ns);

        // 计算端到端延迟：从感知开始到控制结束的总时间
        int64_t e2e_ns = ctrl.timestamp_ns - ctrl.e2e_start_ns;
        stats_.record("end_to_end", e2e_ns);

        if (cfg_.verbose && (plan.frame_id % cfg_.stats_interval_frames == 0)) {
            std::fprintf(stderr,
                         "Frame %4d | C:%7ld ns | E2E:%8ld ns | steer:%.2f\n",
                         plan.frame_id, ctrl.control_time_ns, e2e_ns,
                         ctrl.command.steering);
        }
    }
}
