#pragma once

#include <array>
#include <cmath>
#include <cstddef>

// ============================================================================
// 机械臂运动学：DH 参数、正运动学、雅可比矩阵、数值逆运动学
// 支持 7-DOF 机械臂（KUKA iiwa 类似构型）
// ============================================================================

namespace manipulator {

// ---- 基础类型 ----

struct DHParam {
    double a;     // 连杆长度 (m)
    double alpha; // 连杆扭转角 (rad)
    double d;     // 关节偏移 (m)
    double theta; // 关节角度 (rad) — 变量
};

struct Pose {
    std::array<double, 3> position; // x, y, z (m)
    std::array<double, 3> rpy;      // roll, pitch, yaw (rad)
};

// 4×4 齐次变换矩阵（行优先存储）
struct Mat4 {
    double m[4][4];

    Mat4();
    static Mat4 identity();
    static Mat4 dh_transform(const DHParam &dh);
    Mat4 operator*(const Mat4 &other) const;

    // 从变换矩阵提取位置
    void get_position(double &x, double &y, double &z) const;
    // 从变换矩阵提取 z 轴方向
    void get_z_axis(double &zx, double &zy, double &zz) const;
};

// ---- 运动学类 ----

class Kinematics {
public:
    // 初始化 KUKA iiwa 式 7-DOF DH 参数
    Kinematics();

    // 使用自定义 DH 参数初始化
    explicit Kinematics(const std::array<DHParam, 7> &dh_params);

    int dof() const {
        return n_dof_;
    }

    // 正运动学：给定关节角，计算末端 4×4 位姿矩阵
    Mat4 forward_kinematics(const double *joint_angles) const;

    // 计算末端位姿（position + RPY）
    Pose forward_kinematics_pose(const double *joint_angles) const;

    // 计算 6×N 几何雅可比矩阵（N = n_dof_）
    // J 按行优先存储：J[6*N]，前 3 行为线速度，后 3 行为角速度
    void compute_jacobian(const double *joint_angles, double *J) const;

    // DLS 逆运动学：最小化 ||JΔθ - e||² + λ²||Δθ||²
    // 返回是否收敛，joint_angles 作为初值传入并更新
    // max_iters: 最大迭代次数
    // tol_pos: 位置收敛容差 (m)
    // tol_rot: 旋转收敛容差 (rad)
    // lambda: 阻尼因子（自适应范围内变化）
    bool inverse_kinematics_dls(
        const Pose &target,
        double *joint_angles,
        int max_iters = 50,
        double tol_pos = 1e-4,
        double tol_rot = 1e-4,
        double lambda = 0.1) const;

    // Newton-Raphson（伪逆）逆运动学用于对比
    bool inverse_kinematics_nr(
        const Pose &target,
        double *joint_angles,
        int max_iters = 50,
        double tol_pos = 1e-4,
        double tol_rot = 1e-4) const;

    // 奇异点检测：返回最小奇异值（雅可比矩阵的）
    // 若 min_sigma < threshold，则认为处于奇异构型
    double detect_singularity(const double *joint_angles, double *min_sigma = nullptr) const;

    // 获取 DH 参数（只读）
    const std::array<DHParam, 7> &dh_params() const {
        return dh_params_;
    }

    // 关节限制
    std::array<double, 7> joint_min() const {
        return joint_min_;
    }
    std::array<double, 7> joint_max() const {
        return joint_max_;
    }

private:
    int n_dof_;
    std::array<DHParam, 7> dh_params_;
    std::array<double, 7> joint_min_;
    std::array<double, 7> joint_max_;

    // 计算位姿误差 e = [Δp; Δω] (6x1)
    void compute_pose_error(const Mat4 &current, const Pose &target, double *error) const;

    // 旋转矩阵 → RPY（ZYX 欧拉角）
    void rotmat_to_rpy(const double R[3][3], double *rpy) const;

    // RPY → 旋转矩阵
    void rpy_to_rotmat(double roll, double pitch, double yaw, double R[3][3]) const;

    // 夹关节角度到限制范围内
    void clamp_joints(double *angles) const;

    // Jacobi SVD 用于小型矩阵（N≤7）
    // A: m×n 矩阵（行优先），返回 U(m×m), S(min(m,n)), V(n×n)
    // 使用 one-sided Jacobi 方法处理 A^T A
    void svd(const double *A, int m, int n,
             double *U, double *S, double *V) const;

    // Cholesky 分解 A = L L^T（A 为 n×n 对称正定矩阵）
    // L 下三角（列优先对角线含 1 的变换不可用，使用标准 Cholesky）
    bool cholesky_decompose(const double *A, int n, double *L) const;

    // 前代/回代解 L L^T x = b
    void cholesky_solve(const double *L, int n, const double *b, double *x) const;

    // 矩阵×向量
    void matvec(const double *A, int rows, int cols, const double *x, double *y, bool transpose = false) const;

    // 矩阵×矩阵
    void matmul(const double *A, const double *B, double *C,
                int A_rows, int A_cols, int B_cols, bool transposeB = false) const;
};

} // namespace manipulator
