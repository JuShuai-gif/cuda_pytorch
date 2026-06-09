#include "joint_controller.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <thread>

namespace control {

// ============================================================================
// PIDController 实现
// ============================================================================

PIDController::PIDController() : prev_error_(0), integral_(0), prev_measurement_(0), first_update_(true) {
}

PIDController::PIDController(const PIDParams &params) : params_(params), prev_error_(0), integral_(0),
                                                        prev_measurement_(0), first_update_(true) {
}

double PIDController::update(double setpoint, double measurement, double dt,
                             double feedforward) {
    double error = setpoint - measurement;

    // 比例项
    double p_term = params_.kp * error;

    // 积分项（带抗饱和）
    if (dt > 1e-12 && !first_update_) {
        integral_ += error * dt;
        // 钳位
        if (integral_ > params_.max_integral)
            integral_ = params_.max_integral;
        else if (integral_ < -params_.max_integral)
            integral_ = -params_.max_integral;
    }
    double i_term = params_.ki * integral_;

    // 微分项（对测量值微分以减少"微分冲击"）
    double d_term = 0.0;
    if (dt > 1e-12 && !first_update_) {
        double derivative = (measurement - prev_measurement_) / dt;
        d_term = -params_.kd * derivative; // 负号：对测量值微分
    }

    double output = p_term + i_term + d_term + feedforward;

    // 输出钳位 + 反算积分（条件积分/back-calculation）
    if (output > params_.max_output) {
        output = params_.max_output;
        // 反算积分：如果积分项贡献导致饱和，则回退积分
        double excess = output - (p_term + d_term + feedforward);
        if (params_.ki > 1e-10 && dt > 1e-12) {
            integral_ -= excess / params_.ki * dt * 0.1; // 缓慢回退
        }
    } else if (output < -params_.max_output) {
        output = -params_.max_output;
        double excess = output - (p_term + d_term + feedforward);
        if (params_.ki > 1e-10 && dt > 1e-12) {
            integral_ -= excess / params_.ki * dt * 0.1;
        }
    }

    prev_error_ = error;
    prev_measurement_ = measurement;
    first_update_ = false;

    return output;
}

void PIDController::reset() {
    prev_error_ = 0;
    integral_ = 0;
    prev_measurement_ = 0;
    first_update_ = true;
}

void PIDController::set_params(const PIDParams &params) {
    params_ = params;
    reset();
}

// ============================================================================
// JointController 实现
// ============================================================================

JointController::JointController() {
}

JointController::JointController(const PIDParams &params) : pid_(params) {
}

double JointController::update(double target_pos, double current_pos,
                               double target_vel, double current_vel,
                               double dt, double feedforward_torque) {
    // 速度前馈：目标速度 × 速度增益（简化逆动力学前馈）
    double vel_ff = target_vel * 0.1; // 阻尼补偿前馈系数
    (void)current_vel;                // 保留供扩展使用

    return pid_.update(target_pos, current_pos, dt, feedforward_torque + vel_ff);
}

void JointController::reset() {
    pid_.reset();
}

void JointController::set_params(const PIDParams &params) {
    pid_.set_params(params);
}

// ============================================================================
// JointPlant 实现
// ============================================================================

JointPlant::JointPlant(double inertia, double damping) : inertia_(inertia), damping_(damping), position_(0), velocity_(0) {
}

double JointPlant::step(double torque, double dt) {
    // 双积分器 + 阻尼: I · θ̈ + b · θ̇ = τ
    // θ̈ = (τ - b · θ̇) / I
    // 使用半隐式 Euler 积分
    double acceleration = (torque - damping_ * velocity_) / inertia_;
    velocity_ += acceleration * dt;
    position_ += velocity_ * dt;
    return position_;
}

// ============================================================================
// RealTimeControlLoop 实现
// ============================================================================

RealTimeControlLoop::RealTimeControlLoop(int dof) : dof_(dof), joint_targets_(dof, 0.0) {
    joint_controllers_.resize(dof);
    plants_.resize(dof, JointPlant(0.1, 0.5));
    PIDParams default_params;
    for (auto &jc : joint_controllers_) {
        jc = JointController(default_params);
    }
    pid_params_ = default_params;
}

void RealTimeControlLoop::set_pid_params(const PIDParams &params) {
    pid_params_ = params;
    for (auto &jc : joint_controllers_) {
        jc.set_params(params);
    }
}

void RealTimeControlLoop::get_joint_positions(double *positions) const {
    for (int i = 0; i < dof_; ++i)
        positions[i] = plants_[i].position();
}

void RealTimeControlLoop::get_joint_velocities(double *velocities) const {
    for (int i = 0; i < dof_; ++i)
        velocities[i] = plants_[i].velocity();
}

void RealTimeControlLoop::set_target_trajectory(
    const std::vector<double> *times,
    const std::vector<double> *positions,
    const std::vector<double> *velocities,
    const std::vector<double> *accelerations,
    int num_waypoints) {
    // 将轨迹写入无锁共享状态
    FeedforwardData ff;
    ff.valid = true;

    // 初始目标
    for (int j = 0; j < dof_; ++j) {
        ff.target_pos[j] = (*positions)[j];
        ff.target_vel[j] = velocities ? (*velocities)[j] : 0.0;
        ff.target_accel[j] = accelerations ? (*accelerations)[j] : 0.0;
        // 简化的前馈力矩：基于目标加速度的惯性补偿
        ff.feedforward_torque[j] = ff.target_accel[j] * 0.1; // I * θ̈_des
    }

    ff_state_.write(ff);

    (void)times;
    (void)num_waypoints;
}

ControlLoopStats RealTimeControlLoop::run(double duration_ms,
                                          double control_period_us,
                                          bool use_feedforward) {
    ControlLoopStats stats;
    double dt = control_period_us / 1.0e6; // 转换为秒

    int total_cycles = static_cast<int>(duration_ms * 1000.0 / control_period_us);
    stats.total_cycles = total_cycles;

    // 收集所有循环时间用于计算 P99
    std::vector<double> loop_times;
    loop_times.reserve(total_cycles);

    // 初始化目标：所有关节目标为 0.5 rad 的阶跃
    for (int j = 0; j < dof_; ++j) {
        joint_targets_[j] = 0.5;
    }

    // 将初始目标写入共享状态
    {
        FeedforwardData ff;
        ff.valid = true;
        for (int j = 0; j < dof_; ++j) {
            ff.target_pos[j] = 0.5;
            ff.target_vel[j] = 0.0;
            ff.target_accel[j] = 0.0;
            ff.feedforward_torque[j] = 0.0;
        }
        ff_state_.write(ff);
    }

    for (int cycle = 0; cycle < total_cycles; ++cycle) {
        auto t_start = std::chrono::high_resolution_clock::now();

        // 1. 从无锁共享状态读取目标
        const FeedforwardData *ff = ff_state_.read();
        if (ff && ff->valid) {
            for (int j = 0; j < dof_; ++j) {
                joint_targets_[j] = ff->target_pos[j];
            }
        }

        // 2. 对每个关节计算 PID 输出力矩并模拟动力学
        for (int j = 0; j < dof_; ++j) {
            double current_pos = plants_[j].position();
            double current_vel = plants_[j].velocity();
            double target_pos = joint_targets_[j];

            double ff_torque = 0.0;
            if (use_feedforward && ff && ff->valid) {
                ff_torque = ff->feedforward_torque[j];
            }

            double torque = joint_controllers_[j].update(
                target_pos, current_pos,
                ff ? ff->target_vel[j] : 0.0,
                current_vel, dt, ff_torque);

            // 3. 模拟关节动力学
            plants_[j].step(torque, dt);
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

        loop_times.push_back(elapsed_us);

        stats.total_time_ms += elapsed_us / 1000.0;
        if (elapsed_us > stats.max_loop_time_us) stats.max_loop_time_us = elapsed_us;
        if (elapsed_us < stats.min_loop_time_us) stats.min_loop_time_us = elapsed_us;

        // 4. 检查是否超过 1ms 截止时间
        if (elapsed_us > control_period_us) {
            stats.deadline_misses++;
        }
    }

    // 计算统计量
    if (total_cycles > 0) {
        stats.avg_loop_time_us = stats.total_time_ms * 1000.0 / total_cycles;
    }
    stats.deadline_rate_pct = 100.0 * stats.deadline_misses / total_cycles;

    // 计算 P99
    if (!loop_times.empty()) {
        std::sort(loop_times.begin(), loop_times.end());
        int p99_idx = static_cast<int>(loop_times.size() * 0.99);
        if (p99_idx >= (int)loop_times.size()) p99_idx = (int)loop_times.size() - 1;
        stats.jitter_p99_us = loop_times[p99_idx];
    }

    return stats;
}

} // namespace control
