#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <vector>

// ============================================================================
// 实时关节控制：PID 控制器、前馈控制、1kHz 控制闭环仿真
// 无动态内存分配，锁自由共享状态
// ============================================================================

namespace control {

// ---- PID 控制参数 ----
struct PIDParams {
    double kp = 100.0;          // 比例增益 (Nm/rad)
    double ki = 10.0;           // 积分增益 (Nm/(rad·s))
    double kd = 5.0;            // 微分增益 (Nm·s/rad)
    double max_output = 200.0;  // 最大输出力矩 (Nm)
    double max_integral = 50.0; // 积分饱和上限（抗饱和）
};

// ---- PID 控制器（带抗饱和） ----
class PIDController {
public:
    PIDController();
    explicit PIDController(const PIDParams &params);

    // 计算控制输出
    // setpoint: 目标位置
    // measurement: 当前测量位置
    // dt: 时间步长 (s)
    // feedforward: 前馈力矩 (Nm)，可选
    double update(double setpoint, double measurement, double dt,
                  double feedforward = 0.0);

    void reset();
    void set_params(const PIDParams &params);
    const PIDParams &params() const {
        return params_;
    }

private:
    PIDParams params_;
    double prev_error_;
    double integral_;
    double prev_measurement_;
    bool first_update_;
};

// ---- 单关节 PID + 前馈控制器 ----
class JointController {
public:
    JointController();
    explicit JointController(const PIDParams &params);

    double update(double target_pos, double current_pos,
                  double target_vel, double current_vel,
                  double dt, double feedforward_torque = 0.0);

    void reset();
    void set_params(const PIDParams &params);

private:
    PIDController pid_;
};

// ---- 控制循环统计 ----
struct ControlLoopStats {
    int total_cycles = 0;
    int deadline_misses = 0;        // 超过 1ms 的周期数
    double total_time_ms = 0.0;     // 总执行时间
    double avg_loop_time_us = 0.0;  // 平均循环时间 (μs)
    double max_loop_time_us = 0.0;  // 最大循环时间 (μs)
    double min_loop_time_us = 1e9;  // 最小循环时间 (μs)
    double deadline_rate_pct = 0.0; // 错过截止时间比例
    double jitter_p99_us = 0.0;     // P99 抖动
};

// ---- 简化的关节动力学模型（双积分器 + 摩擦） ----
class JointPlant {
public:
    JointPlant(double inertia = 0.1, double damping = 0.5);

    // 施加力矩，模拟 dt 秒后的新状态
    // torque: 输入力矩 (Nm)
    // dt: 时间步长 (s)
    // 返回新位置
    double step(double torque, double dt);

    double position() const {
        return position_;
    }
    double velocity() const {
        return velocity_;
    }
    void set_state(double pos, double vel) {
        position_ = pos;
        velocity_ = vel;
    }
    void reset() {
        position_ = 0;
        velocity_ = 0;
    }

private:
    double inertia_;  // 转动惯量 (kg·m²)
    double damping_;  // 粘滞阻尼 (Nm·s/rad)
    double position_; // 当前角度 (rad)
    double velocity_; // 当前角速度 (rad/s)
};

// ---- 无锁共享状态（感知线程写入，控制线程读取） ----
template <typename T>
class LockFreeState {
public:
    LockFreeState() : front_idx_(0) {
        buffers_[0].store(nullptr, std::memory_order_relaxed);
        buffers_[1].store(nullptr, std::memory_order_relaxed);
    }

    // 写入端（感知线程）：写入非活跃缓冲，然后交换
    void write(const T &data) {
        int back = 1 - front_idx_.load(std::memory_order_relaxed);
        back_buffer_ = data;
        buffers_[back].store(&back_buffer_, std::memory_order_release);
        front_idx_.store(back, std::memory_order_release);
    }

    // 读取端（控制线程）：读取当前活跃缓冲
    const T *read() const {
        int front = front_idx_.load(std::memory_order_acquire);
        return buffers_[front].load(std::memory_order_acquire);
    }

private:
    T back_buffer_;
    std::atomic<T *> buffers_[2];
    std::atomic<int> front_idx_;
};

// ---- 实时控制循环（7-DOF） ----
class RealTimeControlLoop {
public:
    RealTimeControlLoop(int dof = 7);

    // 配置 PID 参数
    void set_pid_params(const PIDParams &params);

    // 运行控制循环
    // duration_ms: 运行时长 (ms)
    // control_period_us: 控制周期 (μs)，默认 1000μs = 1kHz
    // use_feedforward: 是否使用前馈力矩
    ControlLoopStats run(double duration_ms, double control_period_us = 1000.0,
                         bool use_feedforward = true);

    // 获取当前关节状态
    void get_joint_positions(double *positions) const;
    void get_joint_velocities(double *velocities) const;

    // 设置目标轨迹（外部线程调用）
    void set_target_trajectory(const std::vector<double> *times,
                               const std::vector<double> *positions, // 7*N 扁平
                               const std::vector<double> *velocities,
                               const std::vector<double> *accelerations,
                               int num_waypoints);

private:
    int dof_;
    std::vector<JointController> joint_controllers_;
    std::vector<JointPlant> plants_;
    std::vector<double> joint_targets_; // 当前目标位置
    PIDParams pid_params_;

    // 前馈数据（由感知线程更新）
    struct FeedforwardData {
        double target_pos[7];
        double target_vel[7];
        double target_accel[7];
        double feedforward_torque[7];
        bool valid = false;
    };
    LockFreeState<FeedforwardData> ff_state_;
};

} // namespace control
