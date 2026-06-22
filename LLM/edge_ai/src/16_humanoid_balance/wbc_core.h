#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <vector>
#include <algorithm>
#include <string>

// ============================================================================
// 任务定义：包含雅可比、期望速度、优先级、权重
// ============================================================================
struct Task {
    Eigen::MatrixXd J;     // 雅可比矩阵 m×n，映射关节速度到任务空间速度
    Eigen::VectorXd v_des; // 期望任务空间速度（m 维）
    int priority;          // 优先级数值（越小越高，0 = 最高优先级 P0）
    double damping;        // 阻尼因子 λ（0 = 标准伪逆，>0 = 阻尼最小二乘）
    std::string name;      // 任务名称（用于调试和输出）

    Task() : priority(0), damping(0.01) {
    }

    Task(const Eigen::MatrixXd &jac, const Eigen::VectorXd &vel,
         int prio, double damp, const std::string &n) : J(jac), v_des(vel), priority(prio), damping(damp), name(n) {
    }
};

// ============================================================================
// WBCSolver：层级零空间投影全身控制器
// ============================================================================
class WBCSolver {
public:
    // 构造函数：n_dof = 关节自由度数量
    explicit WBCSolver(int n_dof);

    // 添加一个任务（内部按 priority 排序）
    void addTask(const Task &task);

    // 移除所有任务
    void clear();

    // 求解：返回关节速度 q̇ ∈ R^{n_dof}
    // 按优先级从高到低依次投影
    Eigen::VectorXd solve();

    // 获取求解的详细诊断信息
    struct SolveInfo {
        double time_us;                      // 求解耗时（微秒）
        std::vector<double> task_residuals;  // 每个任务的速度残差 ‖Jq̇ - v_des‖
        std::vector<double> nullspace_ranks; // 每个零空间投影矩阵的秩
        int n_tasks;                         // 任务数量
    };
    SolveInfo lastSolveInfo() const {
        return last_info_;
    }

    int nDof() const {
        return n_dof_;
    }
    int nTasks() const {
        return static_cast<int>(tasks_.size());
    }

private:
    int n_dof_;
    std::vector<Task> tasks_;
    SolveInfo last_info_;

    // 计算（阻尼）伪逆矩阵
    // J 是 m×n，返回 n×m
    Eigen::MatrixXd computePseudoinverse(const Eigen::MatrixXd &J,
                                         double damping);

    // 计算零空间投影矩阵 N = I - J⁺J
    // J_pinv 是 J 的伪逆（n×m 矩阵）
    Eigen::MatrixXd computeNullspaceProjector(const Eigen::MatrixXd &J_pinv,
                                              const Eigen::MatrixXd &J);
};
