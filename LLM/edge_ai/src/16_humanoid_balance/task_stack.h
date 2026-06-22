#pragma once

#include "wbc_core.h"
#include "balance_control.h"
#include <Eigen/Dense>
#include <vector>
#include <random>

// ============================================================================
// 任务栈构建器：创建 34-DOF 人形机器人 WBC 优先级任务栈
// ============================================================================
class TaskStackBuilder {
public:
    // 构造函数
    // n_dof: 自由度数量（默认 34）
    // dt: 控制周期（秒）
    // seed: 随机数种子（用于生成可复现的物理模型参数）
    TaskStackBuilder(int n_dof = 34, double dt = 0.001, unsigned seed = 42);

    // 构建完整的优先级任务栈（P0-P4）
    // 返回已排序的任务列表（priority 从小到大）
    std::vector<Task> buildFullStack();

    // 单独获取各任务（用于自定义优先级排列）
    Task jointLimitTask() const;
    Task balanceTask() const;
    Task contactTask() const;
    Task swingFootTask() const;
    Task handTask() const;
    Task postureTask() const;

    // 从关节速度计算力矩指令（简化动力学）
    // q: 当前关节位置
    // q_dot: WBC 求解得到的关节速度
    // q_des: 目标关节位置
    // 返回：力矩 τ
    Eigen::VectorXd computeTorqueCommands(
        const Eigen::VectorXd &q,
        const Eigen::VectorXd &q_dot,
        const Eigen::VectorXd &q_des) const;

    // 获取用于验证的参考关节状态
    Eigen::VectorXd getNominalPosture() const {
        return q_nominal_;
    }
    Eigen::VectorXd getCurrentJointPos() const {
        return q_current_;
    }
    Eigen::VectorXd getJointLowerLimits() const {
        return q_min_;
    }
    Eigen::VectorXd getJointUpperLimits() const {
        return q_max_;
    }

    // 更新当前关节位置（模拟传感器读数）
    void setCurrentJointPos(const Eigen::VectorXd &q) {
        q_current_ = q;
    }

    int nDof() const {
        return n_dof_;
    }
    double dt() const {
        return dt_;
    }

private:
    int n_dof_;
    double dt_;
    std::mt19937 rng_;

    // 参考姿态（直立，膝关节微屈 5°）
    Eigen::VectorXd q_nominal_;

    // 当前关节位置（模拟状态）
    Eigen::VectorXd q_current_;

    // 关节限位
    Eigen::VectorXd q_min_;
    Eigen::VectorXd q_max_;

    // CoM 雅可比（3×n，映射关节速度到 CoM 速度）
    Eigen::MatrixXd J_com_;

    // 右手雅可比（6×n，映射关节速度到右手末端速度）
    Eigen::MatrixXd J_right_hand_;

    // 右脚雅可比（6×n）
    Eigen::MatrixXd J_right_foot_;

    // Kp, Kd 增益
    double Kp_;
    double Kd_;

    // 生成物理上合理的雅可比矩阵（基于简化的运动学参数）
    void generateKinematicModel();
};
