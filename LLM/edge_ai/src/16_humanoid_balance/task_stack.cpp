#include "task_stack.h"
#include <cmath>
#include <algorithm>

// ============================================================================
// 关节名称辅助（用于调试）
// ============================================================================
[[maybe_unused]] static const char *joint_names_34[] = {
    // 左腿 (6 DOF)
    "L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle_pitch", "L_ankle_roll",
    // 右腿 (6 DOF)
    "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle_pitch", "R_ankle_roll",
    // 腰部 (3 DOF)
    "waist_yaw", "waist_roll", "waist_pitch",
    // 躯干 (3 DOF)
    "torso_roll", "torso_pitch", "torso_yaw",
    // 左臂 (7 DOF)
    "L_sho_pitch", "L_sho_roll", "L_sho_yaw", "L_elbow", "L_wri_roll", "L_wri_pitch", "L_wri_yaw",
    // 右臂 (7 DOF)
    "R_sho_pitch", "R_sho_roll", "R_sho_yaw", "R_elbow", "R_wri_roll", "R_wri_pitch", "R_wri_yaw",
    // 头部 (2 DOF)
    "neck_pitch", "neck_yaw"};

// ============================================================================
// 构造：初始化运动学模型和参考姿态
// ============================================================================
TaskStackBuilder::TaskStackBuilder(int n_dof, double dt, unsigned seed) : n_dof_(n_dof), dt_(dt), rng_(seed),
                                                                          Kp_(150.0), Kd_(10.0) {
    // 名义站立姿态：全部关节初始化为 0（直立），膝关节微屈
    q_nominal_ = Eigen::VectorXd::Zero(n_dof_);
    q_current_ = Eigen::VectorXd::Zero(n_dof_);

    // 膝关节微屈 5°（防止奇异）
    // 注意：不同 DOF 配置的膝关节索引不同
    // 34-DOF: 左膝=3, 右膝=9
    double knee_angle = 5.0 * M_PI / 180.0;     // 5°
    if (n_dof_ > 3) q_nominal_(3) = knee_angle; // 左膝（至少4个关节）
    if (n_dof_ > 9) q_nominal_(9) = knee_angle; // 右膝（至少10个关节）

    q_current_ = q_nominal_;

    // 关节限位（典型人形机器人范围，弧度）
    q_min_ = Eigen::VectorXd::Constant(n_dof_, -2.0);
    q_max_ = Eigen::VectorXd::Constant(n_dof_, 2.0);

    // 膝关节限位特殊设置（仅当关节存在时）
    if (n_dof_ > 3) {
        q_min_(3) = 0.0;
        q_max_(3) = 2.5;
    } // 左膝
    if (n_dof_ > 9) {
        q_min_(9) = 0.0;
        q_max_(9) = 2.5;
    } // 右膝

    // 生成物理上合理的雅可比矩阵
    generateKinematicModel();
}

// ============================================================================
// 生成运动学模型（基于简化的人形机器人运动学参数）
// ============================================================================
void TaskStackBuilder::generateKinematicModel() {
    // 实际系统中雅可比矩阵从机器人运动学/动力学库（如 Pinocchio、RBDL、Drake）获取
    // 此处使用物理上合理的随机雅可比，模拟真实系统特征：
    // - 腿部关节对 CoM 影响最大
    // - 臂部关节对手末端影响最大
    // - 躯干关节影响全局姿态

    std::uniform_real_distribution<double> small_noise(-0.05, 0.05);
    std::uniform_real_distribution<double> link_length(0.2, 0.5);

    // -- CoM 雅可比 (3 × n_dof) --
    // 腿部关节对 CoM 有显著影响，上肢影响较小
    J_com_ = Eigen::MatrixXd::Zero(3, n_dof_);
    for (int j = 0; j < n_dof_; ++j) {
        double influence = 0.0;
        if (j < 12) {
            // 腿部关节（索引 0-11）：对 CoM 影响大
            influence = 0.15 + std::abs(small_noise(rng_));
        } else if (j < 18) {
            // 腰部+躯干（索引 12-17）：中等等影响
            influence = 0.08 + std::abs(small_noise(rng_)) * 0.5;
        } else if (j < 32) {
            // 手臂（索引 18-31）：对 CoM 影响小
            influence = 0.02 + std::abs(small_noise(rng_)) * 0.1;
        } else {
            // 头部（索引 32-33）：几乎不影响
            influence = 0.005;
        }

        // x 方向：主要受前后倾关节影响
        J_com_(0, j) = influence * (j % 3 == 0 ? 1.0 : small_noise(rng_));
        // y 方向：主要受侧倾关节影响
        J_com_(1, j) = influence * (j % 3 == 1 ? 1.0 : small_noise(rng_));
        // z 方向：主要受 pitch 关节影响
        J_com_(2, j) = influence * (j % 3 == 2 ? 1.0 : small_noise(rng_));
    }

    // -- 右手雅可比 (6 × n_dof) --
    // 位置部分 (3 rows)：手臂关节影响大，腿部无影响
    J_right_hand_ = Eigen::MatrixXd::Zero(6, n_dof_);
    for (int j = 18; j < n_dof_ && j < 32; ++j) {   // 所有手臂关节
        double influence = (j >= 25) ? 0.12 : 0.08; // 右臂（25-31）影响更大
        for (int r = 0; r < 3; ++r) {
            J_right_hand_(r, j) = influence * (1.0 + small_noise(rng_));
        }
    }
    // 加上躯干和腰部对末端位置的间接影响
    for (int j = 12; j < n_dof_ && j < 18; ++j) {
        double influence = 0.03;
        for (int r = 0; r < 3; ++r) {
            J_right_hand_(r, j) = influence * (1.0 + small_noise(rng_));
        }
    }
    // 方向部分（3 rows）：类似结构
    for (int j = 18; j < n_dof_ && j < 32; ++j) {
        double influence = (j >= 25) ? 0.10 : 0.06;
        for (int r = 3; r < 6; ++r) {
            J_right_hand_(r, j) = influence * (1.0 + small_noise(rng_));
        }
    }

    // -- 右脚雅可比 (6 × n_dof) --
    J_right_foot_ = Eigen::MatrixXd::Zero(6, n_dof_);
    for (int j = 6; j < n_dof_ && j < 12; ++j) { // 右腿关节
        double influence = 0.15;
        for (int r = 0; r < 3; ++r) {
            J_right_foot_(r, j) = influence * (1.0 + small_noise(rng_));
        }
    }
    for (int j = 12; j < n_dof_ && j < 18; ++j) { // 腰部和躯干
        double influence = 0.04;
        for (int r = 0; r < 3; ++r) {
            J_right_foot_(r, j) = influence * (1.0 + small_noise(rng_));
        }
    }
    for (int j = 6; j < n_dof_ && j < 12; ++j) {
        double influence = 0.12;
        for (int r = 3; r < 6; ++r) {
            J_right_foot_(r, j) = influence * (1.0 + small_noise(rng_));
        }
    }
}

// ============================================================================
// 关节限位任务（P0 最高优先级）
// ============================================================================
Task TaskStackBuilder::jointLimitTask() const {
    // 遍历所有关节，对接近限位的关节添加一维约束
    const double margin = 0.15; // 限位余量（弧度，约 8.6°）
    std::vector<int> active_joints;
    Eigen::VectorXd v_des_active;

    for (int i = 0; i < n_dof_; ++i) {
        double dist_to_min = q_current_(i) - q_min_(i);
        double dist_to_max = q_max_(i) - q_current_(i);

        if (dist_to_min < margin) {
            // 接近下限，需要正速度（远离下限）
            active_joints.push_back(i);
            double urgency = 1.0 - dist_to_min / margin;
            v_des_active.conservativeResize(v_des_active.size() + 1);
            v_des_active(v_des_active.size() - 1) = urgency * 0.5; // 远离速度
        } else if (dist_to_max < margin) {
            // 接近上限，需要负速度（远离上限）
            active_joints.push_back(i);
            double urgency = 1.0 - dist_to_max / margin;
            v_des_active.conservativeResize(v_des_active.size() + 1);
            v_des_active(v_des_active.size() - 1) = -urgency * 0.5;
        }
    }

    int n_active = static_cast<int>(active_joints.size());
    Eigen::MatrixXd J(n_active, n_dof_);
    J.setZero();

    for (int k = 0; k < n_active; ++k) {
        J(k, active_joints[k]) = 1.0;
    }

    if (n_active == 0) {
        // 无关节接近限位，返回空任务（0 行雅可比）
        return Task(Eigen::MatrixXd(0, n_dof_),
                    Eigen::VectorXd(0),
                    0, 0.0, "关节限位（无活跃约束）");
    }

    return Task(J, v_des_active, 0, 0.0, "关节限位");
}

// ============================================================================
// 平衡任务（P1）：通过 CoM 雅可比控制 ZMP
// ============================================================================
Task TaskStackBuilder::balanceTask() const {
    // 使用 CoM 雅可比（3D），控制 CoM 水平位置到目标
    // 实际中目标 CoM 位置由步态规划器给出
    Eigen::MatrixXd J = J_com_; // 3×n

    // 期望 CoM 速度：将 CoM 拉回支撑多边形中心
    Eigen::Vector3d v_des(0.0, 0.0, 0.0);

    // 当前 CoM 位置（简化：从 q_current 反推）
    // 在实际系统中，CoM 由运动学正解计算
    // 此处用 q_current 前 12 个关节（腿部）近似
    double com_x = 0.0, com_y = 0.0;
    int max_leg_joints = std::min(12, n_dof_);
    for (int i = 0; i < max_leg_joints; ++i) {
        com_x += 0.02 * q_current_(i);
        com_y += 0.01 * q_current_(i);
    }

    // 期望 CoM 在脚底中心 (0, 0) 上方
    double Kp_com = 3.0;
    v_des(0) = -Kp_com * com_x;
    v_des(1) = -Kp_com * com_y;
    v_des(2) = 0.0; // 保持高度不变

    return Task(J, v_des, 1, 0.05, "平衡（CoM/ZMP）");
}

// ============================================================================
// 接触任务（P2）：保持支撑脚不动 + 满足摩擦锥
// ============================================================================
Task TaskStackBuilder::contactTask() const {
    // 支撑脚（右脚）必须保持静止：J_foot · q̇ = 0
    Eigen::MatrixXd J = J_right_foot_; // 6×n

    // 期望速度为零（脚不能动）
    Eigen::VectorXd v_des = Eigen::VectorXd::Zero(6);

    return Task(J, v_des, 2, 0.01, "足底接触（右脚静止）");
}

// ============================================================================
// 摆动脚任务（P3）：跟踪预设的摆动脚轨迹
// ============================================================================
Task TaskStackBuilder::swingFootTask() const {
    // 模拟左脚向前迈步的摆动轨迹
    // 在实际系统中，摆动脚轨迹由步态规划器生成
    // 此处简化为：左脚抬升 5cm + 向前移动
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, n_dof_);

    // 左脚雅可比：左腿关节（0-5）+ 腰部（12-14）
    double influence = 0.12;
    int max_left_leg = std::min(6, n_dof_);
    for (int j = 0; j < max_left_leg; ++j) {
        for (int r = 0; r < 3; ++r) {
            J(r, j) = influence * (1.0 + 0.1 * (j % 3));
        }
    }
    for (int j = 12; j < n_dof_ && j < 15; ++j) {
        for (int r = 0; r < 3; ++r) {
            J(r, j) = 0.03;
        }
    }
    for (int j = 0; j < max_left_leg; ++j) {
        for (int r = 3; r < 6; ++r) {
            J(r, j) = 0.08;
        }
    }

    // 摆动轨迹：向前上方移动（模拟步态）
    Eigen::VectorXd v_des(6);
    v_des << 0.15, 0.0, 0.05, // 线性速度：前向 + 抬升
        0.0, 0.0, 0.0;        // 角速度：保持方向

    return Task(J, v_des, 3, 0.1, "摆动脚（左脚迈步）");
}

// ============================================================================
// 手末端任务（P3）：跟踪右手目标姿态
// ============================================================================
Task TaskStackBuilder::handTask() const {
    Eigen::MatrixXd J = J_right_hand_; // 6×n

    // 期望的末端速度：手向前伸展 10cm/s
    Eigen::VectorXd v_des(6);
    v_des << 0.10, 0.0, 0.0, // 前向伸展
        0.0, 0.0, 0.0;       // 保持方向

    return Task(J, v_des, 3, 0.1, "右手末端跟踪");
}

// ============================================================================
// 姿态任务（P4）：维持名义站立姿态
// ============================================================================
Task TaskStackBuilder::postureTask() const {
    // J = I（控制全部关节）
    Eigen::MatrixXd J = Eigen::MatrixXd::Identity(n_dof_, n_dof_);

    // 期望速度：将当前姿态拉回名义姿态
    // v_des = Kp * (q_nominal - q_current)
    double Kp = 5.0;
    Eigen::VectorXd v_des = Kp * (q_nominal_ - q_current_);

    // 使用阻尼伪逆，避免完全消耗零空间
    return Task(J, v_des, 4, 0.5, "姿态（站立）");
}

// ============================================================================
// 构建完整的优先级任务栈
// ============================================================================
std::vector<Task> TaskStackBuilder::buildFullStack() {
    std::vector<Task> tasks;

    Task t0 = jointLimitTask();
    if (t0.J.rows() > 0) {
        tasks.push_back(t0);
    }

    tasks.push_back(balanceTask());
    tasks.push_back(contactTask());
    tasks.push_back(swingFootTask());
    tasks.push_back(handTask());
    tasks.push_back(postureTask());

    // 按 priority 排序
    std::sort(tasks.begin(), tasks.end(),
              [](const Task &a, const Task &b) {
                  return a.priority < b.priority;
              });

    return tasks;
}

// ============================================================================
// 从关节速度计算力矩指令（简化逆动力学）
// ============================================================================
Eigen::VectorXd TaskStackBuilder::computeTorqueCommands(
    const Eigen::VectorXd &q,
    const Eigen::VectorXd &q_dot,
    const Eigen::VectorXd &q_des) const {
    // 简化的 PD 控制器 + 重力补偿模型
    // τ = Kp (q_des - q + q̇·dt) - Kd (q̇ - q̇_des) + 重力项
    // 完整的多体动力学为 τ = M(q)q̈ + C(q,q̇)q̇ + g(q)
    // 此处使用简化形式，因为完整的 M/C/g 计算需要 RNEA 递推

    Eigen::VectorXd q_des_next = q_des + q_dot * dt_; // 一步后的期望位置
    Eigen::VectorXd q_des_vel = (q_des - q) / dt_;    // 期望速度

    Eigen::VectorXd tau = Kp_ * (q_des_next - q) - Kd_ * (q_dot - q_des_vel);

    // 简化的重力补偿：仅对膝关节施加常值补偿力矩
    // 实际系统中重力矩由 RNEA 递推计算
    double gravity_torque_knee = 25.0;                 // Nm（典型膝关节重力矩）
    if (tau.size() > 3) tau(3) += gravity_torque_knee; // 左膝
    if (tau.size() > 9) tau(9) += gravity_torque_knee; // 右膝

    // 力矩限幅（防止损伤电机）
    const double MAX_TORQUE = 150.0; // Nm
    for (int i = 0; i < tau.size(); ++i) {
        tau(i) = std::clamp(tau(i), -MAX_TORQUE, MAX_TORQUE);
    }

    return tau;
}
