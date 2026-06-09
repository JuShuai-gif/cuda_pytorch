#include "pipeline_executor.h"
#include "pipeline_stage.h"

#include <cstdio>
#include <vector>

// ============================================================================
// 顺序执行
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
// PipelinedExecutor 实现
// ============================================================================

PipelinedExecutor::PipelinedExecutor(const PipelineConfig &cfg,
                                     LatencyStats &stats) : cfg_(cfg), stats_(stats) {
}

void PipelinedExecutor::run() {
    std::thread perception_th(&PipelinedExecutor::perception_worker, this);
    std::thread planning_th(&PipelinedExecutor::planning_worker, this);
    std::thread control_th(&PipelinedExecutor::control_worker, this);

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
        in_cv_.notify_one();
    }

    stop_ = true;
    in_cv_.notify_all();

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
            in_cv_.wait(lock, [this]() {
                return !input_queue_.empty() || stop_;
            });
            if (input_queue_.empty() && stop_) break;
            if (!input_queue_.empty()) {
                sensor = input_queue_.front();
                input_queue_.pop();
                has_frame = true;
            }
        }
        if (!has_frame) continue;

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
            pq_cv_.wait(lock, [this]() {
                return !perception_queue_.empty() || (stop_ && input_queue_.empty() && perception_queue_.empty());
            });
            if (perception_queue_.empty()) {
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

        PipelinePlanningOut plan = run_planning(percep);
        stats_.record("planning", plan.planning_time_ns);

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
            lq_cv_.wait(lock, [this]() {
                return !planning_queue_.empty() || (stop_ && input_queue_.empty() && perception_queue_.empty() && planning_queue_.empty());
            });
            if (planning_queue_.empty()) {
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

        PipelineControlOut ctrl = run_control(plan);
        stats_.record("control", ctrl.control_time_ns);

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
