#include "trajectory.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace trajectory {

// ============================================================================
// TrapezoidalProfile 实现
// ============================================================================

TrapezoidalProfile::TrapezoidalProfile() : valid_(false), start_pos_(0), end_pos_(0),
                                           max_vel_(0), max_accel_(0), t_accel_(0), t_const_(0),
                                           t_decel_start_(0), total_time_(0), sign_(1) {
}

void TrapezoidalProfile::plan(double start_pos, double end_pos,
                              double max_vel, double max_accel) {
    start_pos_ = start_pos;
    end_pos_ = end_pos;
    max_vel_ = max_vel;
    max_accel_ = max_accel;
    valid_ = true;

    double D = end_pos_ - start_pos_;
    sign_ = (D >= 0) ? 1.0 : -1.0;
    double abs_D = std::abs(D);

    // 临界位移：刚好加速到 Vmax 然后立即减速
    double D_crit = max_vel_ * max_vel_ / max_accel_;

    if (abs_D < 1e-12) {
        // 零位移
        t_accel_ = 0;
        t_const_ = 0;
        t_decel_start_ = 0;
        total_time_ = 0;
        valid_ = true;
        return;
    }

    if (abs_D >= D_crit) {
        // 三段式：加速 → 匀速 → 减速
        t_accel_ = max_vel_ / max_accel_;
        double d_accel = 0.5 * max_accel_ * t_accel_ * t_accel_;
        double d_const = abs_D - 2.0 * d_accel;
        t_const_ = d_const / max_vel_;
        t_decel_start_ = t_accel_ + t_const_;
        total_time_ = 2.0 * t_accel_ + t_const_;
    } else {
        // 三角形波形：无法达到 Vmax
        double v_peak = std::sqrt(abs_D * max_accel_);
        t_accel_ = v_peak / max_accel_;
        t_const_ = 0;
        t_decel_start_ = t_accel_;
        total_time_ = 2.0 * t_accel_;
    }
}

TrajectoryPoint TrapezoidalProfile::sample(double t) const {
    TrajectoryPoint pt;
    pt.time = t;

    if (!valid_ || total_time_ < 1e-12) {
        pt.position = end_pos_;
        pt.velocity = 0;
        pt.acceleration = 0;
        return pt;
    }

    t = std::max(0.0, std::min(t, total_time_));

    double p_raw, v_raw, a_raw;

    if (t <= t_accel_) {
        // 加速段
        a_raw = max_accel_;
        v_raw = max_accel_ * t;
        p_raw = 0.5 * max_accel_ * t * t;
    } else if (t <= t_decel_start_) {
        // 匀速段
        a_raw = 0;
        v_raw = max_vel_;
        double dt = t - t_accel_;
        p_raw = 0.5 * max_accel_ * t_accel_ * t_accel_ + max_vel_ * dt;
    } else {
        // 减速段
        double dt = t - t_decel_start_;
        a_raw = -max_accel_;
        v_raw = max_vel_ - max_accel_ * dt;
        double p_const = 0.5 * max_accel_ * t_accel_ * t_accel_
                         + max_vel_ * t_const_;
        p_raw = p_const + max_vel_ * dt - 0.5 * max_accel_ * dt * dt;
    }

    pt.position = start_pos_ + sign_ * p_raw;
    pt.velocity = sign_ * v_raw;
    pt.acceleration = sign_ * a_raw;

    return pt;
}

// ============================================================================
// SCurveProfile 实现
// ============================================================================

SCurveProfile::SCurveProfile() : valid_(false), start_pos_(0), end_pos_(0),
                                 max_vel_(0), max_accel_(0), max_jerk_(0), sign_(1), total_time_(0) {
    for (int i = 0; i < 7; ++i) {
        T_[i] = 0;
        p0_[i] = 0;
        v0_[i] = 0;
        a0_[i] = 0;
    }
    for (int i = 0; i < 8; ++i) t_seg_[i] = 0;
}

void SCurveProfile::plan(double start_pos, double end_pos,
                         double max_vel, double max_accel, double max_jerk) {
    start_pos_ = start_pos;
    end_pos_ = end_pos;
    max_vel_ = max_vel;
    max_accel_ = max_accel;
    max_jerk_ = max_jerk;
    valid_ = true;

    double D = end_pos_ - start_pos_;
    sign_ = (D >= 0) ? 1.0 : -1.0;
    double abs_D = std::abs(D);

    if (abs_D < 1e-12) {
        total_time_ = 0;
        for (int i = 0; i < 7; ++i) T_[i] = 0;
        t_seg_[0] = 0;
        for (int i = 0; i < 7; ++i) t_seg_[i + 1] = 0;
        return;
    }

    // 能否达到 A_max？
    double T_j = max_accel_ / max_jerk_; // 达到 A_max 所需的 jerk 时间

    // 临界位移：刚好达到 A_max 然后减速（无匀速、无匀加速段）
    // 加速度三角形：2*T_j 达到 A_max，2*T_j 减速
    // 速度：加速段面积 = A_max * T_j（梯形面积）= 2 * (1/2 * A_max * T_j) = A_max * T_j 不对
    // 重算：v_peak = A_max * T_j（纯 jerk 加速到 A_max 然后 jerk 减速到 0 的速度增量）
    // 实际：段1 jerk up → a 从 0 到 A_max，v 增量为 1/2 * A_max * T_j
    //      段2 const accel (T_a-T_j)，v 增量为 A_max * (T_a - T_j)
    //      段3 jerk down → a 从 A_max 到 0，v 增量为 1/2 * A_max * T_j
    // 总增速 = A_max * T_a
    // 同理减速段减速量 = A_max * T_d
    //
    // 先判断能否达到 V_max
    // 最小时间达到 V_max：
    //   到达 V_max 需要的匀速加速段时间 T_a：
    //   V_max = A_max * T_a（如果 T_a ≥ T_j，有匀加速段）
    //   若 V_max ≤ A_max * T_j：纯 jerk 三角，v_peak = A_max * T_j 达不到 V_max
    //      这里简化：若 V_max ≤ A_max * T_j，则 a_peak = sqrt(V_max * jerk)
    //      需要调整...

    double a_peak = max_accel_;
    double v_peak = max_vel_;

    // 判断能否达到 A_max（即 V_max 是否大于 A_max * T_j）
    if (v_peak > a_peak * T_j) {
        // 能达到 A_max，存在匀加速段
    } else {
        // 无法达到 A_max，jerk 三角形加速
        a_peak = std::sqrt(v_peak * max_jerk_);
        T_j = a_peak / max_jerk_;
    }

    // 加速段位移：
    // 段1: d1 = 1/6 * jerk * T_j^3
    // 段2: d2 = v_at_T_j * (T_a-2*T_j) + 1/2 * a_peak * (T_a-2*T_j)^2（若 T_a > 2*T_j）
    // 段3: d3 = (段1+段2的末速) * T_j - 1/6 * jerk * T_j^3 + a_peak^2 * T_j^2 / (2*jerk) - jerk*T_j^3/3
    // 简化：直接数值积分方式求得加速段总位移
    // 加速段总速度曲线下面积 = v_peak * T_a - (jerk 拐角补偿)
    // 对于对称 jerk 梯形：加速段面积 = v_peak * T_a / 2（平均速度 × 时间）
    //   不对。加速度对称：a(t) 在 [0,T_a] 下面积 = V_max
    //   p 增量 = ∫₀^T_a v(t) dt
    //   用梯形近似：加速段位移 ≈ V_max * T_a / 2（如果 a 全程线性变化）
    //   有匀加速段时更复杂。
    //
    // 实际采用分段积分数值法，离线计算，不影响实时采样

    // 对于 S 曲线，我们直接用分段解析
    compute_segments();
}

void SCurveProfile::compute_segments() {
    // 使用简化参数：T_j = A_max / J_max
    double abs_D = std::abs(end_pos_ - start_pos_);
    double T_j = max_accel_ / max_jerk_;

    if (max_vel_ > max_accel_ * T_j) {
        // 存在匀加速段 (T_a > T_j)
        T_[0] = T_j; // jerk up
        double T_a = max_vel_ / max_accel_;
        T_[1] = T_a - T_j; // const accel
        T_[2] = T_j;       // jerk down
        T_[3] = 0;         // const vel (待计算)
        T_[4] = T_j;       // jerk up (减速)
        T_[5] = T_a - T_j; // const decel
        T_[6] = T_j;       // jerk down (减速到零)
    } else {
        // 纯 jerk 三角：A_max 达不到
        double T_j_new = std::sqrt(max_vel_ / max_jerk_);
        T_j = T_j_new;
        T_[0] = T_j;
        T_[1] = 0;
        T_[2] = T_j;
        T_[3] = 0;
        T_[4] = T_j;
        T_[5] = 0;
        T_[6] = T_j;
    }

    // 计算加速段（段0-2）的位移
    auto integrate_accel_phase = [&](double /*t_start*/, double j_sign) -> double {
        double p = 0, v = 0, a = 0;
        double j = j_sign * max_jerk_;
        for (int seg = 0; seg < 3; ++seg) {
            double dt = T_[seg];
            if (dt <= 0) continue;
            // p(t) = p0 + v0·t + 1/2·a0·t² + 1/6·j·t³
            p += v * dt + 0.5 * a * dt * dt + (1.0 / 6.0) * j * dt * dt * dt;
            v += a * dt + 0.5 * j * dt * dt;
            a += j * dt;
            j = (seg == 0 && j_sign > 0) ? 0.0 : // 段1→段2
                    (seg == 1 && j_sign > 0) ? -max_jerk_ :
                                               // 段2→段3
                    (seg == 0 && j_sign < 0) ? 0.0 :
                (seg == 1 && j_sign < 0)     ? max_jerk_ :
                                               0.0;
        }
        return p;
    };

    double d_accel = integrate_accel_phase(0, 1.0);
    double d_decel = integrate_accel_phase(0, -1.0); // 对称

    double d_const_needed = abs_D - d_accel - d_decel;
    if (d_const_needed < 0) {
        // 无法达到 Vmax：缩小到三角波形
        // 均匀缩放所有段时间
        double scale = std::sqrt(abs_D / (d_accel + d_decel));
        for (int i = 0; i < 7; ++i) T_[i] *= scale;
        T_[3] = 0;
        d_accel *= scale * scale;
        d_decel *= scale * scale;
        d_const_needed = abs_D - d_accel - d_decel;
    }

    T_[3] = (max_vel_ > 1e-10) ? (d_const_needed / max_vel_) : 0;
    if (T_[3] < 0) T_[3] = 0;

    // 计算各段初始条件和累积时间
    t_seg_[0] = 0;
    double jerk_signs[7] = {1, 0, -1, 0, -1, 0, 1};

    double cur_p = 0, cur_v = 0, cur_a = 0;
    double cur_j = 0;

    for (int i = 0; i < 7; ++i) {
        p0_[i] = cur_p;
        v0_[i] = cur_v;
        a0_[i] = cur_a;

        double dt = T_[i];
        t_seg_[i + 1] = t_seg_[i] + dt;

        if (dt > 0) {
            cur_j = jerk_signs[i] * max_jerk_;
            // 更新状态到段末
            cur_p += cur_v * dt + 0.5 * cur_a * dt * dt + (1.0 / 6.0) * cur_j * dt * dt * dt;
            cur_v += cur_a * dt + 0.5 * cur_j * dt * dt;
            cur_a += cur_j * dt;
        }
    }

    total_time_ = t_seg_[7];
}

TrajectoryPoint SCurveProfile::sample(double t) const {
    TrajectoryPoint pt;
    pt.time = t;

    if (!valid_ || total_time_ < 1e-12) {
        pt.position = end_pos_;
        pt.velocity = 0;
        pt.acceleration = 0;
        return pt;
    }

    t = std::max(0.0, std::min(t, total_time_));

    // 找到所在段
    int seg = 0;
    for (int i = 0; i < 7; ++i) {
        if (t <= t_seg_[i + 1] + 1e-12) {
            seg = i;
            break;
        }
    }

    double tau = t - t_seg_[seg];
    double dt = T_[seg];
    if (dt < 1e-12) {
        pt.position = start_pos_ + sign_ * p0_[seg];
        pt.velocity = 0;
        pt.acceleration = 0;
        return pt;
    }

    double jerk_signs[7] = {1, 0, -1, 0, -1, 0, 1};
    double j = jerk_signs[seg] * max_jerk_;

    double p_raw = p0_[seg] + v0_[seg] * tau + 0.5 * a0_[seg] * tau * tau + (1.0 / 6.0) * j * tau * tau * tau;
    double v_raw = v0_[seg] + a0_[seg] * tau + 0.5 * j * tau * tau;
    double a_raw = a0_[seg] + j * tau;

    pt.position = start_pos_ + sign_ * p_raw;
    pt.velocity = sign_ * v_raw;
    pt.acceleration = sign_ * a_raw;

    return pt;
}

// ============================================================================
// generate_waypoints 实现
// ============================================================================

std::vector<TrajectoryPoint> generate_waypoints(
    const TrapezoidalProfile &profile, double control_rate_hz) {
    std::vector<TrajectoryPoint> waypoints;
    double T = profile.total_duration();
    double dt = 1.0 / control_rate_hz;
    int n = static_cast<int>(T / dt) + 2;

    waypoints.reserve(n);
    for (int i = 0; i < n; ++i) {
        double t = i * dt;
        if (t > T) t = T;
        waypoints.push_back(profile.sample(t));
        if (t >= T) break;
    }
    return waypoints;
}

std::vector<TrajectoryPoint> generate_waypoints(
    const SCurveProfile &profile, double control_rate_hz) {
    std::vector<TrajectoryPoint> waypoints;
    double T = profile.total_duration();
    double dt = 1.0 / control_rate_hz;
    int n = static_cast<int>(T / dt) + 2;

    waypoints.reserve(n);
    for (int i = 0; i < n; ++i) {
        double t = i * dt;
        if (t > T) t = T;
        waypoints.push_back(profile.sample(t));
        if (t >= T) break;
    }
    return waypoints;
}

// ============================================================================
// CubicSpline 实现
// ============================================================================

CubicSpline::CubicSpline() : valid_(false), n_seg_(0), t0_(0), t_end_(0) {
}

void CubicSpline::fit(const double *times, const double *positions, int n,
                      double v_start, double v_end) {
    n_seg_ = n - 1;
    if (n_seg_ < 1) {
        valid_ = false;
        return;
    }

    t0_ = times[0];
    t_end_ = times[n - 1];

    a_.resize(n_seg_);
    b_.resize(n_seg_);
    c_.resize(n_seg_);
    d_.resize(n_seg_);
    t_knots_.resize(n);

    for (int i = 0; i < n; ++i) t_knots_[i] = times[i];

    // 构建三对角系统解节点速度 v_i
    // 输入：h_i = t_{i+1} - t_i, p_i = positions[i]
    // 方程: h_{i-1}·v_{i-1} + 2(h_{i-1}+h_i)·v_i + h_i·v_{i+1} = 3(h_i·Δp_{i-1}/h_{i-1} + h_{i-1}·Δp_i/h_i)
    // 其中 Δp_i = (p_{i+1} - p_i)/h_i

    int m = n; // 节点数
    std::vector<double> h(n_seg_);
    for (int i = 0; i < n_seg_; ++i) {
        h[i] = times[i + 1] - times[i];
    }

    // 构建三对角矩阵（n 个节点速度 v_0..v_{n-1}）
    std::vector<double> lower(m), diag(m), upper(m), rhs(m);

    // 边界条件
    diag[0] = 1.0;
    upper[0] = 0.0;
    rhs[0] = v_start;

    for (int i = 1; i < m - 1; ++i) {
        lower[i] = h[i - 1];
        diag[i] = 2.0 * (h[i - 1] + h[i]);
        upper[i] = h[i];
        double dp_prev = (positions[i] - positions[i - 1]) / h[i - 1];
        double dp_curr = (positions[i + 1] - positions[i]) / h[i];
        rhs[i] = 3.0 * (h[i] * dp_prev + h[i - 1] * dp_curr);
    }

    diag[m - 1] = 1.0;
    lower[m - 1] = 0.0;
    rhs[m - 1] = v_end;

    std::vector<double> v(m);
    solve_tridiagonal(m, lower.data(), diag.data(), upper.data(), rhs.data(), v.data());

    // 从 v 和 p 计算每段系数
    for (int i = 0; i < n_seg_; ++i) {
        double hi = h[i];
        double pi = positions[i];
        double pi1 = positions[i + 1];
        double vi = v[i];
        double vi1 = v[i + 1];

        a_[i] = pi;
        b_[i] = vi;
        c_[i] = (3.0 * (pi1 - pi) / hi - 2.0 * vi - vi1) / hi;
        d_[i] = (vi + vi1 - 2.0 * (pi1 - pi) / hi) / (hi * hi);
    }

    valid_ = true;
}

TrajectoryPoint CubicSpline::sample(double t) const {
    TrajectoryPoint pt;
    pt.time = t;

    if (!valid_ || n_seg_ < 1) {
        pt.position = 0;
        pt.velocity = 0;
        pt.acceleration = 0;
        return pt;
    }

    t = std::max(t0_, std::min(t, t_end_));

    // 找到所在段
    int seg = 0;
    for (int i = 0; i < n_seg_; ++i) {
        if (t <= t_knots_[i + 1] + 1e-12) {
            seg = i;
            break;
        }
    }

    double tau = t - t_knots_[seg];
    double s = a_[seg] + b_[seg] * tau + c_[seg] * tau * tau + d_[seg] * tau * tau * tau;
    double s_dot = b_[seg] + 2.0 * c_[seg] * tau + 3.0 * d_[seg] * tau * tau;
    double s_ddot = 2.0 * c_[seg] + 6.0 * d_[seg] * tau;

    pt.position = s;
    pt.velocity = s_dot;
    pt.acceleration = s_ddot;

    return pt;
}

void CubicSpline::solve_tridiagonal(int n, const double *lower,
                                    const double *diag, const double *upper,
                                    const double *rhs, double *x) {
    // Thomas 算法
    std::vector<double> c_mod(n);
    std::vector<double> d_mod(n);

    c_mod[0] = upper[0] / diag[0];
    d_mod[0] = rhs[0] / diag[0];

    for (int i = 1; i < n; ++i) {
        double denom = diag[i] - lower[i] * c_mod[i - 1];
        c_mod[i] = (i < n - 1) ? upper[i] / denom : 0.0;
        d_mod[i] = (rhs[i] - lower[i] * d_mod[i - 1]) / denom;
    }

    x[n - 1] = d_mod[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        x[i] = d_mod[i] - c_mod[i] * x[i + 1];
    }
}

} // namespace trajectory
