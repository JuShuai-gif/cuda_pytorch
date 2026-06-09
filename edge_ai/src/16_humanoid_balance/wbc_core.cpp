#include "wbc_core.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <iomanip>

// ============================================================================
// 构造与任务管理
// ============================================================================
WBCSolver::WBCSolver(int n_dof) : n_dof_(n_dof) {
    last_info_ = {};
}

void WBCSolver::addTask(const Task &task) {
    tasks_.push_back(task);
    // 按 priority 升序排列（数值越小优先级越高）
    std::sort(tasks_.begin(), tasks_.end(),
              [](const Task &a, const Task &b) {
                  return a.priority < b.priority;
              });
}

void WBCSolver::clear() {
    tasks_.clear();
}

// ============================================================================
// 阻尼最小二乘伪逆：J⁺ = Jᵀ (J Jᵀ + λ²I)⁻¹
// 适用于 m < n（任务维度 < 关节维度）的典型 WBC 场景
// ============================================================================
Eigen::MatrixXd WBCSolver::computePseudoinverse(const Eigen::MatrixXd &J,
                                                double damping) {
    int m = J.rows();
    int n = J.cols();

    if (m == 0 || n == 0) {
        return Eigen::MatrixXd::Zero(n, m);
    }

    if (damping > 0.0) {
        // 阻尼最小二乘：J⁺ = Jᵀ (J Jᵀ + λ² I_m)⁻¹
        Eigen::MatrixXd JJt = J * J.transpose();     // m×m
        JJt.diagonal().array() += damping * damping; // 正则化
        Eigen::MatrixXd JJt_inv = JJt.inverse();     // m×m
        return J.transpose() * JJt_inv;              // n×m
    } else {
        // 标准伪逆：通过完全正交分解（CompleteOrthogonalDecomposition）
        // 内部使用列选主元 QR，对秩亏矩阵比 SVD 更快
        return J.completeOrthogonalDecomposition().pseudoInverse();
    }
}

// ============================================================================
// 零空间投影矩阵：N = I_n - J⁺J
// J_pinv 是 n×m 的伪逆
// ============================================================================
Eigen::MatrixXd WBCSolver::computeNullspaceProjector(
    const Eigen::MatrixXd &J_pinv, const Eigen::MatrixXd &J) {
    int n = J_pinv.rows();
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
    return I - J_pinv * J; // N = I - J⁺J
}

// ============================================================================
// 层级零空间投影求解
// 算法：
//   N = I (n×n)
//   q̇ = 0 (n)
//   for each task (sorted by priority ascending):
//       Ĵ = J_k · N               // 增广雅可比（投影到当前零空间）
//       Ĵ⁺ = pinv(Ĵ)               // 增广伪逆
//       Δq̇ = Ĵ⁺ (v_k - J_k q̇)     // 零空间内的速度增量
//       q̇ += Δq̇
//       N = N - Ĵ⁺ Ĵ               // 更新零空间
// ============================================================================
Eigen::VectorXd WBCSolver::solve() {
    auto t_start = std::chrono::high_resolution_clock::now();

    if (tasks_.empty()) {
        last_info_ = {};
        return Eigen::VectorXd::Zero(n_dof_);
    }

    // 初始化零空间和累积速度
    Eigen::MatrixXd N = Eigen::MatrixXd::Identity(n_dof_, n_dof_);
    Eigen::VectorXd q_dot = Eigen::VectorXd::Zero(n_dof_);

    last_info_.n_tasks = static_cast<int>(tasks_.size());
    last_info_.task_residuals.clear();
    last_info_.nullspace_ranks.clear();

    for (const auto &task : tasks_) {
        int m = static_cast<int>(task.J.rows());

        if (m == 0) continue;

        // 步骤 1：将当前任务的雅可比投影到零空间
        Eigen::MatrixXd J_aug = task.J * N; // m×n（增广雅可比）

        // 步骤 2：计算增广雅可比的伪逆
        Eigen::MatrixXd J_aug_pinv = computePseudoinverse(J_aug, task.damping); // n×m

        // 步骤 3：计算零空间内的速度增量
        // Δq̇ = Ĵ⁺ (v_des - J q̇_current)
        Eigen::VectorXd residual = task.v_des - task.J * q_dot;
        Eigen::VectorXd dq = J_aug_pinv * residual; // n×1

        q_dot += dq;

        // 步骤 4：更新零空间 N = N - Ĵ⁺ Ĵ
        N = N - J_aug_pinv * J_aug;

        // 记录诊断信息
        double task_residual = (task.J * q_dot - task.v_des).norm();
        last_info_.task_residuals.push_back(task_residual);

        // 估计零空间的秩（通过迹近似）
        double nullspace_trace = N.trace();
        last_info_.nullspace_ranks.push_back(nullspace_trace);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    last_info_.time_us = std::chrono::duration<double, std::micro>(
                             t_end - t_start)
                             .count();

    return q_dot;
}
