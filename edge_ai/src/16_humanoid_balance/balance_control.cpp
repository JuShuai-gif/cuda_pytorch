#include "balance_control.h"
#include <cmath>
#include <stdexcept>

// ============================================================================
// LIPM 实现
// ============================================================================
LIPM::LIPM(double com_height, double gravity) : zc_(com_height), g_(gravity) {
    if (zc_ <= 0.0) {
        throw std::invalid_argument("CoM 高度必须为正数");
    }
    omega_ = std::sqrt(g_ / zc_);
}

Eigen::Vector2d LIPM::computeZMP(const Eigen::Vector2d &com_pos,
                                 const Eigen::Vector2d &com_acc) const {
    // ZMP 公式：p_zmp = x - (zc/g) * ẍ = x - ẍ / ω²
    return com_pos - com_acc / (omega_ * omega_);
}

Eigen::Vector2d LIPM::computeCoMAccel(const Eigen::Vector2d &com_pos,
                                      const Eigen::Vector2d &zmp) const {
    // ẍ = ω² (x - p_zmp)
    return omega_ * omega_ * (com_pos - zmp);
}

void LIPM::step(State &state, const Eigen::Vector2d &zmp, double dt) const {
    // 半隐式欧拉积分（速度先更新，再用新速度更新位置）
    Eigen::Vector2d accel = computeCoMAccel(state.pos, zmp);
    state.vel += accel * dt;
    state.pos += state.vel * dt;
}

std::vector<Eigen::Vector2d> LIPM::generateCoMTrajectory(
    const std::vector<Eigen::Vector2d> &footstep_plan,
    double dt, double T_step) const {
    // 简化的预览控制：将 ZMP 参考设为每步的足底中心，用 LIPM 前向推演 CoM
    int steps_per_phase = static_cast<int>(T_step / dt);
    int total_steps = steps_per_phase * static_cast<int>(footstep_plan.size());

    std::vector<Eigen::Vector2d> com_traj;
    com_traj.reserve(total_steps);

    // 初始状态：CoM 在第一个足部中心
    State state;
    state.pos = footstep_plan.front();
    state.vel = Eigen::Vector2d::Zero();

    // ZMP 参考轨迹：在支撑多边形中心之间线性插值
    for (size_t fp = 0; fp < footstep_plan.size(); ++fp) {
        // 当前阶段的 ZMP 目标
        Eigen::Vector2d zmp_target = footstep_plan[fp];

        // 如果是最后一个落点，ZMP 固定
        if (fp == footstep_plan.size() - 1) {
            for (int s = 0; s < steps_per_phase; ++s) {
                step(state, zmp_target, dt);
                com_traj.push_back(state.pos);
            }
        } else {
            // ZMP 从当前目标线性过渡到下一个目标
            Eigen::Vector2d zmp_next = footstep_plan[fp + 1];
            for (int s = 0; s < steps_per_phase; ++s) {
                double alpha = static_cast<double>(s) / steps_per_phase;
                Eigen::Vector2d zmp = (1.0 - alpha) * zmp_target + alpha * zmp_next;
                step(state, zmp, dt);
                com_traj.push_back(state.pos);
            }
        }
    }

    return com_traj;
}

// ============================================================================
// SupportPolygon 实现
// ============================================================================
SupportPolygon::SupportPolygon(const std::vector<Eigen::Vector2d> &vertices) {
    setVertices(vertices);
}

void SupportPolygon::setVertices(const std::vector<Eigen::Vector2d> &vertices) {
    if (vertices.size() < 3) {
        throw std::invalid_argument("支撑多边形至少需要 3 个顶点");
    }
    vertices_ = vertices;
}

bool SupportPolygon::contains(const Eigen::Vector2d &point) const {
    // 射线法（Ray Casting）：从点向右发射水平射线，计算与多边形边的交点数
    // 奇数次交点 = 在内部，偶数次 = 在外部
    int count = 0;
    int n = static_cast<int>(vertices_.size());

    for (int i = 0; i < n; ++i) {
        const Eigen::Vector2d &v1 = vertices_[i];
        const Eigen::Vector2d &v2 = vertices_[(i + 1) % n];

        // 检查水平射线是否穿过此边
        // 边必须在点的 y 范围之间（一端在上一端在下）
        if ((v1.y() > point.y()) != (v2.y() > point.y())) {
            // 计算射线与边的交点 x 坐标
            double x_intersect = v1.x() + (v2.x() - v1.x()) * (point.y() - v1.y()) / (v2.y() - v1.y());

            if (point.x() < x_intersect) {
                ++count;
            }
        }
    }

    return (count % 2) == 1;
}

double SupportPolygon::area() const {
    // Shoelace 公式（鞋带公式）
    double a = 0.0;
    int n = static_cast<int>(vertices_.size());
    for (int i = 0; i < n; ++i) {
        const auto &v1 = vertices_[i];
        const auto &v2 = vertices_[(i + 1) % n];
        a += v1.x() * v2.y() - v2.x() * v1.y();
    }
    return std::abs(a) * 0.5;
}

Eigen::Vector2d SupportPolygon::centroid() const {
    Eigen::Vector2d c(0.0, 0.0);
    for (const auto &v : vertices_) {
        c += v;
    }
    return c / static_cast<double>(vertices_.size());
}

// ============================================================================
// FrictionCone 实现
// ============================================================================
FrictionCone::FrictionCone(double mu, double fz_min, double fz_max) : mu_(mu), fz_min_(fz_min), fz_max_(fz_max) {
    if (mu_ <= 0.0) {
        throw std::invalid_argument("摩擦系数必须为正数");
    }
}

bool FrictionCone::isForceValid(const Eigen::Vector3d &force) const {
    double fx = force.x();
    double fy = force.y();
    double fz = force.z();

    // 法向力检查
    if (fz < fz_min_ || fz > fz_max_) return false;

    // 摩擦力检查：|f_t| ≤ μ f_n（库仑摩擦）
    double f_tangential = std::sqrt(fx * fx + fy * fy);
    return f_tangential <= mu_ * std::abs(fz);
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd>
FrictionCone::getLinearConstraints() const {
    // 线性化摩擦锥：4 条平面 + 法向力上下界
    // 约束形式为 A·f ≤ b
    // f = [fx, fy, fz]ᵀ

    Eigen::MatrixXd A(6, 3);
    Eigen::VectorXd b(6);

    // 4 条摩擦锥线性化平面
    // [ 1,  0, -μ] f ≤ 0
    // [-1,  0, -μ] f ≤ 0
    // [ 0,  1, -μ] f ≤ 0
    // [ 0, -1, -μ] f ≤ 0
    double mu = mu_;
    A << 1.0, 0.0, -mu,
        -1.0, 0.0, -mu,
        0.0, 1.0, -mu,
        0.0, -1.0, -mu,
        0.0, 0.0, -1.0, // -fz ≤ -fz_min  → fz ≥ fz_min
        0.0, 0.0, 1.0;  //  fz ≤ fz_max

    b << 0.0, 0.0, 0.0, 0.0, -fz_min_, fz_max_;

    return {A, b};
}
