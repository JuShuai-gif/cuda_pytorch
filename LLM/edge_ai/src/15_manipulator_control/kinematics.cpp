#include "kinematics.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace manipulator {

// ============================================================================
// Mat4 实现
// ============================================================================

Mat4::Mat4() {
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            m[i][j] = (i == j) ? 1.0 : 0.0;
}

Mat4 Mat4::identity() {
    Mat4 result;
    return result;
}

Mat4 Mat4::dh_transform(const DHParam &dh) {
    double ct = std::cos(dh.theta);
    double st = std::sin(dh.theta);
    double ca = std::cos(dh.alpha);
    double sa = std::sin(dh.alpha);

    Mat4 T;
    T.m[0][0] = ct;
    T.m[0][1] = -st * ca;
    T.m[0][2] = st * sa;
    T.m[0][3] = dh.a * ct;

    T.m[1][0] = st;
    T.m[1][1] = ct * ca;
    T.m[1][2] = -ct * sa;
    T.m[1][3] = dh.a * st;

    T.m[2][0] = 0.0;
    T.m[2][1] = sa;
    T.m[2][2] = ca;
    T.m[2][3] = dh.d;

    T.m[3][0] = 0.0;
    T.m[3][1] = 0.0;
    T.m[3][2] = 0.0;
    T.m[3][3] = 1.0;

    return T;
}

Mat4 Mat4::operator*(const Mat4 &other) const {
    Mat4 result;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            result.m[i][j] = 0.0;
            for (int k = 0; k < 4; ++k) {
                result.m[i][j] += m[i][k] * other.m[k][j];
            }
        }
    }
    return result;
}

void Mat4::get_position(double &x, double &y, double &z) const {
    x = m[0][3];
    y = m[1][3];
    z = m[2][3];
}

void Mat4::get_z_axis(double &zx, double &zy, double &zz) const {
    zx = m[0][2];
    zy = m[1][2];
    zz = m[2][2];
}

// ============================================================================
// Kinematics 实现
// ============================================================================

Kinematics::Kinematics() : n_dof_(7) {
    dh_params_[0] = {0.0, -M_PI / 2.0, 0.36, 0.0};
    dh_params_[1] = {0.0, M_PI / 2.0, 0.0, 0.0};
    dh_params_[2] = {0.0, M_PI / 2.0, 0.42, 0.0};
    dh_params_[3] = {0.0, -M_PI / 2.0, 0.0, 0.0};
    dh_params_[4] = {0.0, -M_PI / 2.0, 0.40, 0.0};
    dh_params_[5] = {0.0, M_PI / 2.0, 0.0, 0.0};
    dh_params_[6] = {0.0, 0.0, 0.126, 0.0};

    double limit = 170.0 * M_PI / 180.0;
    for (int i = 0; i < 7; ++i) {
        joint_min_[i] = -limit;
        joint_max_[i] = limit;
    }
    joint_min_[2] = -160.0 * M_PI / 180.0;
    joint_max_[2] = 160.0 * M_PI / 180.0;
}

Kinematics::Kinematics(const std::array<DHParam, 7> &dh_params) : n_dof_(7), dh_params_(dh_params) {
    double limit = 170.0 * M_PI / 180.0;
    for (int i = 0; i < 7; ++i) {
        joint_min_[i] = -limit;
        joint_max_[i] = limit;
    }
}

Mat4 Kinematics::forward_kinematics(const double *joint_angles) const {
    Mat4 T = Mat4::identity();
    for (int i = 0; i < n_dof_; ++i) {
        DHParam dh = dh_params_[i];
        dh.theta = joint_angles[i];
        T = T * Mat4::dh_transform(dh);
    }
    return T;
}

Pose Kinematics::forward_kinematics_pose(const double *joint_angles) const {
    Mat4 T = forward_kinematics(joint_angles);
    Pose pose;
    T.get_position(pose.position[0], pose.position[1], pose.position[2]);

    double R[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R[i][j] = T.m[i][j];

    rotmat_to_rpy(R, pose.rpy.data());
    return pose;
}

void Kinematics::compute_jacobian(const double *joint_angles, double *J) const {
    // 注意：关节 i 的旋转轴是 z_{i-1}（前一个坐标系的 z 轴）
    // 所以要在乘以第 i 个 DH 变换之前取 z 轴和原点
    Mat4 T_accum = Mat4::identity();

    double z_axes[7][3];
    double origins[7][3];

    for (int i = 0; i < n_dof_; ++i) {
        // 先取当前累积变换的 z 轴和原点作为关节 i 的参数
        z_axes[i][0] = T_accum.m[0][2];
        z_axes[i][1] = T_accum.m[1][2];
        z_axes[i][2] = T_accum.m[2][2];

        origins[i][0] = T_accum.m[0][3];
        origins[i][1] = T_accum.m[1][3];
        origins[i][2] = T_accum.m[2][3];

        // 再乘以关节 i 的 DH 变换
        DHParam dh = dh_params_[i];
        dh.theta = joint_angles[i];
        T_accum = T_accum * Mat4::dh_transform(dh);
    }

    // 末端位置 = 最终累积变换的原点
    double pe[3] = {T_accum.m[0][3], T_accum.m[1][3], T_accum.m[2][3]};

    // 填充雅可比矩阵 J (6×N，行优先)
    for (int i = 0; i < n_dof_; ++i) {
        double dx = pe[0] - origins[i][0];
        double dy = pe[1] - origins[i][1];
        double dz = pe[2] - origins[i][2];

        double zx = z_axes[i][0];
        double zy = z_axes[i][1];
        double zz = z_axes[i][2];

        // Jv_i = z_i × (p_ee - p_i)
        J[0 * n_dof_ + i] = zy * dz - zz * dy;
        J[1 * n_dof_ + i] = zz * dx - zx * dz;
        J[2 * n_dof_ + i] = zx * dy - zy * dx;

        // Jω_i = z_i
        J[3 * n_dof_ + i] = zx;
        J[4 * n_dof_ + i] = zy;
        J[5 * n_dof_ + i] = zz;
    }
}

void Kinematics::compute_pose_error(const Mat4 &current, const Pose &target, double *error) const {
    // 位置误差
    double cx, cy, cz;
    current.get_position(cx, cy, cz);
    error[0] = target.position[0] - cx;
    error[1] = target.position[1] - cy;
    error[2] = target.position[2] - cz;

    // 旋转误差：从 R_err = R_target * R_current^T 提取旋转向量
    double Rc[3][3], Rt[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            Rc[i][j] = current.m[i][j];

    rpy_to_rotmat(target.rpy[0], target.rpy[1], target.rpy[2], Rt);

    // Re = Rt * Rc^T
    double Re[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            Re[i][j] = 0.0;
            for (int k = 0; k < 3; ++k)
                Re[i][j] += Rt[i][k] * Rc[j][k];
        }

    // 旋转矩阵 → 旋转向量 (axis-angle)
    double trace = Re[0][0] + Re[1][1] + Re[2][2];
    double cos_theta = (trace - 1.0) * 0.5;
    if (cos_theta > 1.0) cos_theta = 1.0;
    if (cos_theta < -1.0) cos_theta = -1.0;
    double theta = std::acos(cos_theta);

    if (std::abs(theta) < 1e-10) {
        error[3] = 0.0;
        error[4] = 0.0;
        error[5] = 0.0;
    } else {
        double k = theta / (2.0 * std::sin(theta));
        error[3] = k * (Re[2][1] - Re[1][2]);
        error[4] = k * (Re[0][2] - Re[2][0]);
        error[5] = k * (Re[1][0] - Re[0][1]);
    }
}

bool Kinematics::inverse_kinematics_dls(
    const Pose &target,
    double *joint_angles,
    int max_iters,
    double tol_pos,
    double tol_rot,
    double lambda) const {
    // 工作数组（栈分配，无动态内存）
    double J[6 * 7];
    double error[6];
    double JTJ[7 * 7];
    double JTe[7];
    double L[7 * 7];
    double delta_theta[7];

    for (int iter = 0; iter < max_iters; ++iter) {
        Mat4 T_current = forward_kinematics(joint_angles);
        compute_pose_error(T_current, target, error);

        double pos_err = std::sqrt(error[0] * error[0] + error[1] * error[1] + error[2] * error[2]);
        double rot_err = std::sqrt(error[3] * error[3] + error[4] * error[4] + error[5] * error[5]);

        if (pos_err < tol_pos && rot_err < tol_rot) {
            return true;
        }

        compute_jacobian(joint_angles, J);

        // J^T J (7×7)
        for (int i = 0; i < 7; ++i) {
            for (int j = 0; j < 7; ++j) {
                JTJ[i * 7 + j] = 0.0;
                for (int k = 0; k < 6; ++k) {
                    JTJ[i * 7 + j] += J[k * 7 + i] * J[k * 7 + j];
                }
            }
        }

        // J^T e (7×1)
        for (int i = 0; i < 7; ++i) {
            JTe[i] = 0.0;
            for (int k = 0; k < 6; ++k) {
                JTe[i] += J[k * 7 + i] * error[k];
            }
        }

        // 自适应阻尼
        double lambda_sq = lambda * lambda;
        if (pos_err < 0.01 && rot_err < 0.01) {
            lambda_sq *= 0.01;
        }

        // A = J^T J + λ²I
        double A[7 * 7];
        for (int i = 0; i < 7; ++i) {
            for (int j = 0; j < 7; ++j) {
                A[i * 7 + j] = JTJ[i * 7 + j];
            }
            A[i * 7 + i] += lambda_sq;
        }

        if (!cholesky_decompose(A, 7, L)) {
            // 增大 λ 重试
            lambda_sq = lambda * lambda * 10.0;
            for (int i = 0; i < 7; ++i) {
                for (int j = 0; j < 7; ++j)
                    A[i * 7 + j] = JTJ[i * 7 + j];
                A[i * 7 + i] += lambda_sq;
            }
            if (!cholesky_decompose(A, 7, L)) {
                return false;
            }
        }

        cholesky_solve(L, 7, JTe, delta_theta);

        // 自适应步长：如果步长太大则缩放
        double max_delta = 0.0;
        for (int i = 0; i < 7; ++i) {
            if (std::abs(delta_theta[i]) > max_delta)
                max_delta = std::abs(delta_theta[i]);
        }
        if (max_delta > 1.0) {
            double scale = 1.0 / max_delta;
            for (int i = 0; i < 7; ++i)
                delta_theta[i] *= scale;
        }

        for (int i = 0; i < 7; ++i) {
            joint_angles[i] += delta_theta[i];
        }
        clamp_joints(joint_angles);
    }

    Mat4 T_final = forward_kinematics(joint_angles);
    compute_pose_error(T_final, target, error);
    double pos_err = std::sqrt(error[0] * error[0] + error[1] * error[1] + error[2] * error[2]);
    double rot_err = std::sqrt(error[3] * error[3] + error[4] * error[4] + error[5] * error[5]);
    return (pos_err < tol_pos * 10.0 && rot_err < tol_rot * 10.0);
}

bool Kinematics::inverse_kinematics_nr(
    const Pose &target,
    double *joint_angles,
    int max_iters,
    double tol_pos,
    double tol_rot) const {
    double J[6 * 7];
    double error[6];
    double delta_theta[7];

    for (int iter = 0; iter < max_iters; ++iter) {
        Mat4 T_current = forward_kinematics(joint_angles);
        compute_pose_error(T_current, target, error);

        double pos_err = std::sqrt(error[0] * error[0] + error[1] * error[1] + error[2] * error[2]);
        double rot_err = std::sqrt(error[3] * error[3] + error[4] * error[4] + error[5] * error[5]);
        if (pos_err < tol_pos && rot_err < tol_rot) {
            return true;
        }

        compute_jacobian(joint_angles, J);

        // 构建 J^T (7×6) 用于 SVD
        double JT[7 * 6];
        for (int i = 0; i < 7; ++i)
            for (int k = 0; k < 6; ++k)
                JT[i * 6 + k] = J[k * 7 + i];

        // SVD of J^T (7×6): J^T = U · Σ · V^T
        // U: 7×7, S: 6×1, V: 6×6
        double U_svd[49], S_svd[7], V_svd[36];
        svd(JT, 7, 6, U_svd, S_svd, V_svd);

        // 计算 e_v = V^T · error (6×1)
        double Vt_e[6] = {0};
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                Vt_e[i] += V_svd[j * 6 + i] * error[j]; // V_svd is 6×6, V^T[i][j] = V[j][i]
            }
        }

        // s_i = Vt_e[i] / σ_i (对于 σ_i > threshold)
        double sv_threshold = 1e-6;
        double s_tmp[7] = {0};
        for (int i = 0; i < 6; ++i) {
            if (S_svd[i] > sv_threshold) {
                s_tmp[i] = Vt_e[i] / S_svd[i];
            }
        }

        // Δθ = U · s_tmp (7×1)
        for (int i = 0; i < 7; ++i) {
            delta_theta[i] = 0.0;
            for (int j = 0; j < 6; ++j) {
                delta_theta[i] += U_svd[i * 7 + j] * s_tmp[j];
            }
        }

        // 步长限制
        double max_delta = 0.0;
        for (int i = 0; i < 7; ++i)
            if (std::abs(delta_theta[i]) > max_delta)
                max_delta = std::abs(delta_theta[i]);
        if (max_delta > 0.5) {
            double scale = 0.5 / max_delta;
            for (int i = 0; i < 7; ++i)
                delta_theta[i] *= scale;
        }

        for (int i = 0; i < 7; ++i) {
            joint_angles[i] += delta_theta[i];
        }
        clamp_joints(joint_angles);
    }

    Mat4 T_final = forward_kinematics(joint_angles);
    compute_pose_error(T_final, target, error);
    double pos_err = std::sqrt(error[0] * error[0] + error[1] * error[1] + error[2] * error[2]);
    double rot_err = std::sqrt(error[3] * error[3] + error[4] * error[4] + error[5] * error[5]);
    return (pos_err < tol_pos * 10.0 && rot_err < tol_rot * 10.0);
}

double Kinematics::detect_singularity(const double *joint_angles, double *min_sigma) const {
    double J[6 * 7];
    compute_jacobian(joint_angles, J);

    double JT[7 * 6];
    for (int i = 0; i < 7; ++i)
        for (int k = 0; k < 6; ++k)
            JT[i * 6 + k] = J[k * 7 + i];

    double U[49], S[7], V[36];
    svd(JT, 7, 6, U, S, V);

    double m = S[5]; // 第 6 个奇异值（rank ≤ 6）
    if (min_sigma) *min_sigma = m;
    return m;
}

// ============================================================================
// 内部工具函数
// ============================================================================

void Kinematics::rotmat_to_rpy(const double R[3][3], double *rpy) const {
    double sy = std::sqrt(R[0][0] * R[0][0] + R[1][0] * R[1][0]);
    bool singular = sy < 1e-6;

    if (!singular) {
        rpy[0] = std::atan2(R[2][1], R[2][2]);
        rpy[1] = std::atan2(-R[2][0], sy);
        rpy[2] = std::atan2(R[1][0], R[0][0]);
    } else {
        rpy[0] = std::atan2(-R[1][2], R[1][1]);
        rpy[1] = std::atan2(-R[2][0], sy);
        rpy[2] = 0.0;
    }
}

void Kinematics::rpy_to_rotmat(double roll, double pitch, double yaw, double R[3][3]) const {
    double cr = std::cos(roll), sr = std::sin(roll);
    double cp = std::cos(pitch), sp = std::sin(pitch);
    double cy = std::cos(yaw), sy = std::sin(yaw);

    R[0][0] = cy * cp;
    R[0][1] = cy * sp * sr - sy * cr;
    R[0][2] = cy * sp * cr + sy * sr;
    R[1][0] = sy * cp;
    R[1][1] = sy * sp * sr + cy * cr;
    R[1][2] = sy * sp * cr - cy * sr;
    R[2][0] = -sp;
    R[2][1] = cp * sr;
    R[2][2] = cp * cr;
}

void Kinematics::clamp_joints(double *angles) const {
    for (int i = 0; i < n_dof_; ++i) {
        if (angles[i] < joint_min_[i]) angles[i] = joint_min_[i];
        if (angles[i] > joint_max_[i]) angles[i] = joint_max_[i];
    }
}

// ============================================================================
// Jacobi SVD（用于小型矩阵 N≤10）
// ============================================================================

void Kinematics::svd(const double *A, int m, int n,
                     double *U, double *S, double *V) const {
    int r = std::min(m, n);

    // 1. A^T A (n×n)
    double ATA[49]; // max 7×7
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            ATA[i * n + j] = 0.0;
            for (int k = 0; k < m; ++k) {
                ATA[i * n + j] += A[k * n + i] * A[k * n + j];
            }
        }
    }

    // 2. Jacobi 特征分解 A^T A = V · Λ · V^T
    for (int i = 0; i < n * n; ++i) V[i] = 0.0;
    for (int i = 0; i < n; ++i) V[i * n + i] = 1.0;

    const int max_sweeps = 30;
    const double eps = 1e-12;

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        double max_off = 0.0;
        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                double apq = ATA[p * n + q];
                double app = ATA[p * n + p];
                double aqq = ATA[q * n + q];
                max_off = std::max(max_off, std::abs(apq));

                if (std::abs(apq) > eps * std::sqrt(app * aqq)) {
                    double theta = 0.5 * std::atan2(2.0 * apq, aqq - app);
                    double c = std::cos(theta);
                    double s = std::sin(theta);

                    double old_pp = app;
                    double old_qq = aqq;
                    double old_pq = apq;

                    ATA[p * n + p] = c * c * old_pp - 2.0 * c * s * old_pq + s * s * old_qq;
                    ATA[q * n + q] = s * s * old_pp + 2.0 * c * s * old_pq + c * c * old_qq;
                    ATA[p * n + q] = (c * c - s * s) * old_pq + c * s * (old_pp - old_qq);
                    ATA[q * n + p] = ATA[p * n + q];

                    for (int k = 0; k < n; ++k) {
                        if (k == p || k == q) continue;
                        double akp = ATA[k * n + p];
                        double akq = ATA[k * n + q];
                        ATA[k * n + p] = c * akp - s * akq;
                        ATA[p * n + k] = ATA[k * n + p];
                        ATA[k * n + q] = s * akp + c * akq;
                        ATA[q * n + k] = ATA[k * n + q];
                    }

                    for (int k = 0; k < n; ++k) {
                        double vkp = V[k * n + p];
                        double vkq = V[k * n + q];
                        V[k * n + p] = c * vkp - s * vkq;
                        V[k * n + q] = s * vkp + c * vkq;
                    }
                }
            }
        }
        if (max_off < eps) break;
    }

    // 3. 提取奇异值并排序（降序）
    struct {
        double val;
        int idx;
    } sv_pairs[7];
    for (int i = 0; i < n; ++i) {
        double val = ATA[i * n + i];
        sv_pairs[i].val = (val > 0) ? std::sqrt(val) : 0.0;
        sv_pairs[i].idx = i;
    }
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (sv_pairs[j].val > sv_pairs[i].val) {
                auto tmp = sv_pairs[i];
                sv_pairs[i] = sv_pairs[j];
                sv_pairs[j] = tmp;
            }
        }
    }

    double V_sorted[49];
    for (int j = 0; j < n; ++j) {
        int col = sv_pairs[j].idx;
        for (int i = 0; i < n; ++i)
            V_sorted[i * n + j] = V[i * n + col];
        S[j] = sv_pairs[j].val;
    }
    for (int i = 0; i < n * n; ++i) V[i] = V_sorted[i];

    // 4. 计算 U: u_j = A · v_j / σ_j (j < r)
    for (int i = 0; i < m * m; ++i) U[i] = 0.0;
    for (int j = 0; j < r; ++j) {
        if (S[j] > 1e-12) {
            for (int i = 0; i < m; ++i) {
                double sum = 0.0;
                for (int k = 0; k < n; ++k)
                    sum += A[i * n + k] * V[k * n + j];
                U[i * m + j] = sum / S[j];
            }
        }
    }
    // Gram-Schmidt 补全 U 的剩余列
    for (int j = r; j < m; ++j) {
        U[j * m + j] = 1.0;
        for (int k = 0; k < j; ++k) {
            double dot = 0.0;
            for (int i = 0; i < m; ++i)
                dot += U[i * m + k] * U[i * m + j];
            for (int i = 0; i < m; ++i)
                U[i * m + j] -= dot * U[i * m + k];
        }
        double norm = 0.0;
        for (int i = 0; i < m; ++i)
            norm += U[i * m + j] * U[i * m + j];
        norm = std::sqrt(norm);
        if (norm > 1e-12) {
            for (int i = 0; i < m; ++i)
                U[i * m + j] /= norm;
        }
    }
}

// ============================================================================
// Cholesky 分解
// ============================================================================

bool Kinematics::cholesky_decompose(const double *A, int n, double *L) const {
    for (int i = 0; i < n * n; ++i) L[i] = 0.0;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = A[i * n + j];
            for (int k = 0; k < j; ++k) {
                sum -= L[i * n + k] * L[j * n + k];
            }

            if (i == j) {
                if (sum <= 0.0) return false;
                L[i * n + i] = std::sqrt(sum);
            } else {
                L[i * n + j] = sum / L[j * n + j];
            }
        }
    }
    return true;
}

void Kinematics::cholesky_solve(const double *L, int n, const double *b, double *x) const {
    double y[7];

    // 前代: L y = b
    for (int i = 0; i < n; ++i) {
        double sum = b[i];
        for (int j = 0; j < i; ++j) {
            sum -= L[i * n + j] * y[j];
        }
        y[i] = sum / L[i * n + i];
    }

    // 回代: L^T x = y
    for (int i = n - 1; i >= 0; --i) {
        double sum = y[i];
        for (int j = i + 1; j < n; ++j) {
            sum -= L[j * n + i] * x[j];
        }
        x[i] = sum / L[i * n + i];
    }
}

void Kinematics::matvec(const double *A, int rows, int cols, const double *x, double *y, bool transpose) const {
    if (!transpose) {
        for (int i = 0; i < rows; ++i) {
            y[i] = 0.0;
            for (int j = 0; j < cols; ++j) {
                y[i] += A[i * cols + j] * x[j];
            }
        }
    } else {
        for (int i = 0; i < rows; ++i) {
            y[i] = 0.0;
            for (int j = 0; j < cols; ++j) {
                y[i] += A[j * cols + i] * x[j];
            }
        }
    }
}

void Kinematics::matmul(const double *A, const double *B, double *C,
                        int A_rows, int A_cols, int B_cols, bool transposeB) const {
    for (int i = 0; i < A_rows; ++i) {
        for (int j = 0; j < B_cols; ++j) {
            C[i * B_cols + j] = 0.0;
            for (int k = 0; k < A_cols; ++k) {
                double bkj = transposeB ? B[j * A_cols + k] : B[k * B_cols + j];
                C[i * B_cols + j] += A[i * A_cols + k] * bkj;
            }
        }
    }
}

} // namespace manipulator
