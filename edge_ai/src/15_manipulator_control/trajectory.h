#pragma once

#include <array>
#include <vector>

// ============================================================================
// 轨迹生成：梯形速度曲线、S 曲线、多点样条插值
// 所有系数预计算一次，实时采样 O(1)，不使用动态内存分配
// ============================================================================

namespace trajectory {

// ---- 轨迹采样点 ----
struct TrajectoryPoint {
    double position;     // 关节位置 (rad)
    double velocity;     // 关节速度 (rad/s)
    double acceleration; // 关节加速度 (rad/s²)
    double time;         // 时间戳 (s)
};

// ============================================================================
// 梯形速度曲线：加速 → 匀速 → 减速
// ============================================================================

class TrapezoidalProfile {
public:
    TrapezoidalProfile();

    // 规划轨迹
    // start_pos, end_pos: 起止位置
    // max_vel: 最大速度（绝对值）
    // max_accel: 最大加速度（绝对值）
    void plan(double start_pos, double end_pos, double max_vel, double max_accel);

    // 在时间 t 处采样（t 范围：0 ~ total_duration()）
    TrajectoryPoint sample(double t) const;

    // 总持续时间
    double total_duration() const {
        return total_time_;
    }

    // 是否成功规划（三角形波形或三段波形）
    bool is_valid() const {
        return valid_;
    }

private:
    bool valid_;
    double start_pos_, end_pos_;
    double max_vel_, max_accel_;
    double t_accel_;       // 加速段时间
    double t_const_;       // 匀速段时间
    double t_decel_start_; // 减速开始时间
    double total_time_;
    double sign_; // +1 或 -1（正向/反向运动）

    // 内部：对给定位移规划时间
    void plan_ramp(double displacement);
};

// ============================================================================
// S 曲线（7 段 jerk-limited）：
// 段1: +jerk  (加加速度)   T1 = T_j
// 段2: 0 jerk (匀加速)     T2
// 段3: -jerk  (减加速度)   T3 = T_j
// 段4: 0 jerk (匀速)       T4
// 段5: -jerk  (加减速)     T5 = T_j
// 段6: 0 jerk (匀减速)     T6
// 段7: +jerk  (减减速)     T7 = T_j
// ============================================================================

class SCurveProfile {
public:
    SCurveProfile();

    // max_jerk: 最大加加速度 (rad/s³)
    void plan(double start_pos, double end_pos,
              double max_vel, double max_accel, double max_jerk);

    TrajectoryPoint sample(double t) const;

    double total_duration() const {
        return total_time_;
    }
    bool is_valid() const {
        return valid_;
    }

private:
    bool valid_;
    double start_pos_, end_pos_;
    double max_vel_, max_accel_, max_jerk_;
    double sign_;

    double T_[7];     // 各段持续时间
    double t_seg_[8]; // 各段结束时间（t_seg_[0]=0, t_seg_[k]=Σ(T_0..T_{k-1})）
    double total_time_;

    // 段 k 的初始条件：位置、速度、加速度
    double p0_[7], v0_[7], a0_[7];

    void compute_segments();
};

// ============================================================================
// 在线插补工具
// ============================================================================

// 从轨迹以 control_rate (Hz) 采样生成路径点
// 预分配：输出向量预先 reserve 足够容量
std::vector<TrajectoryPoint> generate_waypoints(
    const TrapezoidalProfile &profile, double control_rate_hz);

std::vector<TrajectoryPoint> generate_waypoints(
    const SCurveProfile &profile, double control_rate_hz);

// ============================================================================
// 三次样条（通过 N 个路径点，位置和速度连续）
// ============================================================================

class CubicSpline {
public:
    CubicSpline();

    // waypoints: (时间, 位置) 对
    // v_start, v_end: 起止速度边界条件
    void fit(const double *times, const double *positions, int n,
             double v_start, double v_end);

    // 在时间 t 处采样
    TrajectoryPoint sample(double t) const;

    double start_time() const {
        return t0_;
    }
    double end_time() const {
        return t_end_;
    }
    int num_segments() const {
        return n_seg_;
    }
    bool is_valid() const {
        return valid_;
    }

private:
    bool valid_;
    int n_seg_;
    double t0_, t_end_;

    // 每段 (i) 的系数: s(t) = a_i + b_i·τ + c_i·τ² + d_i·τ³，τ = t - t_i
    std::vector<double> a_, b_, c_, d_;
    std::vector<double> t_knots_; // 节点时间

    void solve_tridiagonal(int n, const double *lower,
                           const double *diag, const double *upper,
                           const double *rhs, double *x);
};

} // namespace trajectory
