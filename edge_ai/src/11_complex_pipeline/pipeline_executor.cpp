#include "pipeline_executor.h"

#include <iomanip>
#include <iostream>

// ============================================================================
// 顺序执行器
// ============================================================================

void SequentialExecutor::run() {
    std::cout << "\n[模式：顺序执行] 流水线深度 = 1，阶段数 = 7\n\n";

    for (int frame_id = 0; frame_id < cfg_.num_frames; frame_id++) {
        int64_t e2e_start = now_ns();

        Timer stage_timer;

        // 阶段 1：传感器
        stage_timer.start();
        auto sensor = run_sensor(frame_id, cfg_, rng_);
        stats_.record_stage("sensor", stage_timer.elapsed_us());

        // 阶段 2：预处理
        stage_timer.start();
        auto prep = run_preprocess(sensor, cfg_, rng_);
        stats_.record_stage("preprocess", stage_timer.elapsed_us());

        // 阶段 3：检测
        stage_timer.start();
        auto det = run_detection(prep, cfg_, rng_);
        stats_.record_stage("detection", stage_timer.elapsed_us());

        // 阶段 4：跟踪
        stage_timer.start();
        auto track = run_tracking(det, cfg_, rng_);
        stats_.record_stage("tracking", stage_timer.elapsed_us());

        // 阶段 5：预测
        stage_timer.start();
        auto pred = run_prediction(track, cfg_, rng_);
        stats_.record_stage("prediction", stage_timer.elapsed_us());

        // 阶段 6：规划
        stage_timer.start();
        auto plan = run_planning(pred, cfg_, rng_);
        stats_.record_stage("planning", stage_timer.elapsed_us());

        // 阶段 7：控制
        stage_timer.start();
        auto unused_ctrl = run_control(plan, cfg_);
        (void)unused_ctrl;
        stats_.record_stage("control", stage_timer.elapsed_us());

        int64_t e2e_ns = now_ns() - e2e_start;
        stats_.record_e2e(e2e_ns / 1000);

        if (cfg_.verbose) {
            std::cout << "帧 " << std::setw(4) << frame_id
                      << " | 端到端：" << std::setw(7) << (e2e_ns / 1000)
                      << " us\n";
        }
        if ((frame_id + 1) % cfg_.stats_interval == 0) {
            std::cout << "  已处理 " << (frame_id + 1) << "/"
                      << cfg_.num_frames << " 帧\n";
        }
    }
}

// ============================================================================
// 流水线执行器
// ============================================================================

void PipelinedExecutor::run() {
    int depth = cfg_.pipeline_depth;
    std::cout << "\n[模式：流水线执行] 流水线深度 = " << depth
              << "，阶段数 = 7\n\n";

    std::thread sensor_th(&PipelinedExecutor::sensor_worker, this);
    std::thread prep_th(&PipelinedExecutor::preprocess_worker, this);
    std::thread det_th(&PipelinedExecutor::detection_worker, this);
    std::thread track_th(&PipelinedExecutor::tracking_worker, this);
    std::thread pred_th(&PipelinedExecutor::prediction_worker, this);
    std::thread plan_th(&PipelinedExecutor::planning_worker, this);
    std::thread ctrl_th(&PipelinedExecutor::control_worker, this);

    // 将帧推入流水线
    for (int frame_id = 0; frame_id < cfg_.num_frames; frame_id++) {
        SensorFrame frame;
        frame.frame_id = frame_id;
        frame.timestamp_ns = now_ns();
        sensor_in_q_.push(frame);
    }

    // 发送输入结束信号
    stop_flag_ = true;
    {
        SensorFrame dummy;
        dummy.frame_id = -1;
        sensor_in_q_.push(dummy);
    }

    sensor_th.join();
    prep_th.join();
    det_th.join();
    track_th.join();
    pred_th.join();
    plan_th.join();
    ctrl_th.join();
}

void PipelinedExecutor::sensor_worker() {
    std::mt19937 rng(cfg_.seed + 100);
    while (true) {
        SensorFrame input;
        if (!sensor_in_q_.pop(input) || input.frame_id == -1) {
            SensorFrame sentinel;
            sentinel.frame_id = -1;
            prep_in_q_.push(sentinel);
            break;
        }
        int64_t e2e_ns = input.timestamp_ns;

        Timer st;
        st.start();
        auto result = run_sensor(input.frame_id, cfg_, rng);
        stats_.record_stage("sensor", st.elapsed_us());

        result.timestamp_ns = e2e_ns; // 传递端到端起始时间
        prep_in_q_.push(result);
    }
}

void PipelinedExecutor::preprocess_worker() {
    std::mt19937 rng(cfg_.seed + 200);
    while (true) {
        SensorFrame input;
        if (!prep_in_q_.pop(input) || input.frame_id == -1) {
            PreprocessedData sentinel;
            sentinel.frame_id = -1;
            det_in_q_.push(sentinel);
            break;
        }
        Timer st;
        st.start();
        auto result = run_preprocess(input, cfg_, rng);
        stats_.record_stage("preprocess", st.elapsed_us());

        result.timestamp_ns = input.timestamp_ns; // 传递端到端起始时间
        det_in_q_.push(result);
    }
}

void PipelinedExecutor::detection_worker() {
    std::mt19937 rng(cfg_.seed + 300);
    while (true) {
        PreprocessedData input;
        if (!det_in_q_.pop(input) || input.frame_id == -1) {
            DetectionResult sentinel;
            sentinel.frame_id = -1;
            tracking_in_q_.push(sentinel);
            break;
        }
        Timer st;
        st.start();
        auto result = run_detection(input, cfg_, rng);
        stats_.record_stage("detection", st.elapsed_us());

        result.timestamp_ns = input.timestamp_ns; // 传递端到端起始时间
        tracking_in_q_.push(result);
    }
}

void PipelinedExecutor::tracking_worker() {
    std::mt19937 rng(cfg_.seed + 400);
    while (true) {
        DetectionResult input;
        if (!tracking_in_q_.pop(input) || input.frame_id == -1) {
            TrackingResult sentinel;
            sentinel.frame_id = -1;
            pred_in_q_.push(sentinel);
            break;
        }
        Timer st;
        st.start();
        auto result = run_tracking(input, cfg_, rng);
        stats_.record_stage("tracking", st.elapsed_us());

        result.timestamp_ns = input.timestamp_ns; // 传递端到端起始时间
        pred_in_q_.push(result);
    }
}

void PipelinedExecutor::prediction_worker() {
    std::mt19937 rng(cfg_.seed + 500);
    while (true) {
        TrackingResult input;
        if (!pred_in_q_.pop(input) || input.frame_id == -1) {
            PredictionResult sentinel;
            sentinel.frame_id = -1;
            planning_in_q_.push(sentinel);
            break;
        }
        Timer st;
        st.start();
        auto result = run_prediction(input, cfg_, rng);
        stats_.record_stage("prediction", st.elapsed_us());

        result.timestamp_ns = input.timestamp_ns; // 传递端到端起始时间
        planning_in_q_.push(result);
    }
}

void PipelinedExecutor::planning_worker() {
    std::mt19937 rng(cfg_.seed + 600);
    while (true) {
        PredictionResult input;
        if (!planning_in_q_.pop(input) || input.frame_id == -1) {
            PlanningResult sentinel;
            sentinel.frame_id = -1;
            control_in_q_.push(sentinel);
            break;
        }
        Timer st;
        st.start();
        auto result = run_planning(input, cfg_, rng);
        stats_.record_stage("planning", st.elapsed_us());

        result.timestamp_ns = input.timestamp_ns; // 传递端到端起始时间
        control_in_q_.push(result);
    }
}

void PipelinedExecutor::control_worker() {
    while (true) {
        PlanningResult input;
        if (!control_in_q_.pop(input) || input.frame_id == -1) {
            break;
        }
        Timer st;
        st.start();
        auto result = run_control(input, cfg_);
        stats_.record_stage("control", st.elapsed_us());

        int64_t e2e_ns = now_ns() - input.timestamp_ns;
        stats_.record_e2e(e2e_ns / 1000);

        if (cfg_.verbose) {
            std::cout << "帧 " << std::setw(4) << result.frame_id
                      << " | 端到端：" << std::setw(7) << (e2e_ns / 1000)
                      << " us\n";
        }
        if ((result.frame_id + 1) % cfg_.stats_interval == 0) {
            std::cout << "  已处理 " << (result.frame_id + 1) << "/"
                      << cfg_.num_frames << " 帧\n";
        }
    }
}
