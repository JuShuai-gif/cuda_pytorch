#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <cmath>

// ============================================================================
// 线性倒立摆模型（LIPM）：用于实时平衡控制
// ============================================================================
class LIPM {
public:
    // com_height: 质心恒定高度（m）
    // gravity: 重力加速度（m/s²）
    LIPM(double com_height = 0.80, double gravity = 9.81);

    // 从 CoM 位置和加速度计算 ZMP 位置（2D 水平面）
    // com_pos: (x, y) CoM 水平位置
    // com_acc: (ẍ, ÿ) CoM 水平加速度
    Eigen::Vector2d computeZMP(const Eigen::Vector2d &com_pos,
                               const Eigen::Vector2d &com_acc) const;

    // 从 ZMP 位置计算期望的 CoM 加速度
    Eigen::Vector2d computeCoMAccel(const Eigen::Vector2d &com_pos,
                                    const Eigen::Vector2d &zmp) const;

    // LIPM 状态前推一步（欧拉积分）
    // state_pos/vel: 会被原地更新
    struct State {
        Eigen::Vector2d pos; // CoM 水平位置
        Eigen::Vector2d vel; // CoM 水平速度
    };
    void step(State &state, const Eigen::Vector2d &zmp, double dt) const;

    // 生成步行质心轨迹（简化预览控制）
    // footstep_plan: 各步的 ZMP 参考点序列（足底中心）
    // dt: 离散时间步长
    // T_step: 每步持续时间
    std::vector<Eigen::Vector2d> generateCoMTrajectory(
        const std::vector<Eigen::Vector2d> &footstep_plan,
        double dt, double T_step) const;

    double omega() const {
        return omega_;
    }
    double comHeight() const {
        return zc_;
    }

private:
    double zc_;    // 质心恒定高度
    double g_;     // 重力加速度
    double omega_; // ω = √(g/zc)
};

// ============================================================================
// 支撑多边形：由一系列 2D 顶点定义（顺时针或逆时针均可）
// ============================================================================
class SupportPolygon {
public:
    SupportPolygon() = default;

    // 由顶点列表构造（顶点需按顺序排列形成闭合凸包）
    explicit SupportPolygon(const std::vector<Eigen::Vector2d> &vertices);

    // 设置顶点
    void setVertices(const std::vector<Eigen::Vector2d> &vertices);

    // 判定点 point 是否在多边形内部（射线法）
    bool contains(const Eigen::Vector2d &point) const;

    // 获取多边形面积（Shoelace 公式）
    double area() const;

    // 获取多边形中心
    Eigen::Vector2d centroid() const;

    const std::vector<Eigen::Vector2d> &vertices() const {
        return vertices_;
    }
    int nVertices() const {
        return static_cast<int>(vertices_.size());
    }

private:
    std::vector<Eigen::Vector2d> vertices_;
};

// ============================================================================
// 线性化摩擦锥约束
// ============================================================================
class FrictionCone {
public:
    // mu: 摩擦系数（典型值 0.5-0.8）
    // fz_min: 最小法向力（防止离地）
    // fz_max: 最大法向力（防止过载）
    FrictionCone(double mu = 0.6, double fz_min = 10.0, double fz_max = 1000.0);

    // 检查接触力是否满足摩擦锥约束
    // force: (fx, fy, fz) 世界坐标系下的接触力
    bool isForceValid(const Eigen::Vector3d &force) const;

    // 获取线性化约束矩阵 A 和上界 b：A·f ≤ b
    // 返回 (A_mat, b_vec)
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> getLinearConstraints() const;

    double mu() const {
        return mu_;
    }
    double fzMin() const {
        return fz_min_;
    }
    double fzMax() const {
        return fz_max_;
    }

private:
    double mu_;     // 摩擦系数
    double fz_min_; // 最小法向力
    double fz_max_; // 最大法向力
};

// ============================================================================
// 步行模式生成器辅助结构
// ============================================================================
struct WalkingPattern {
    std::vector<Eigen::Vector2d> com_trajectory; // 质心轨迹（x,y）序列
    std::vector<Eigen::Vector2d> zmp_trajectory; // ZMP 轨迹（x,y）序列
    std::vector<bool> zmp_in_polygon;            // 每步 ZMP 是否在支撑多边形内
    double step_duration;                        // 每步时长
    int n_steps;                                 // 总步数
};
