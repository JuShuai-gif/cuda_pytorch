#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <curand_kernel.h>
#include <math.h>

#define STATE_DIM 4    // x, y, vx, vy
#define MEAS_DIM 2     // range, bearing (or x, y for KF)

// ─── Linear algebra helpers (small fixed-size matrices) ─────────────────────

__device__ void mat_vec_mul_4x4(const float* M, const float* v, float* out) {
    out[0] = M[0]*v[0] + M[1]*v[1] + M[2]*v[2]  + M[3]*v[3];
    out[1] = M[4]*v[0] + M[5]*v[1] + M[6]*v[2]  + M[7]*v[3];
    out[2] = M[8]*v[0] + M[9]*v[1] + M[10]*v[2] + M[11]*v[3];
    out[3] = M[12]*v[0]+ M[13]*v[1]+ M[14]*v[2] + M[15]*v[3];
}

__device__ void mat_mul_4x4(const float* A, const float* B, float* C) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++)
                sum += A[i*4 + k] * B[k*4 + j];
            C[i*4 + j] = sum;
        }
    }
}

__device__ void mat_transpose_4x4(const float* A, float* AT) {
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            AT[j*4 + i] = A[i*4 + j];
}

__device__ void mat_add_4x4(const float* A, const float* B, float* C) {
    for (int i = 0; i < 16; i++) C[i] = A[i] + B[i];
}

__device__ void mat_sub_4x4(const float* A, const float* B, float* C) {
    for (int i = 0; i < 16; i++) C[i] = A[i] - B[i];
}

__device__ void mat_mul_2x4_4x4(const float* A, const float* B, float* C) {
    // A: 2x4, B: 4x4, C: 2x4
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++)
                sum += A[i*4 + k] * B[k*4 + j];
            C[i*4 + j] = sum;
        }
    }
}

__device__ void mat_mul_4x4_4x2(const float* A, const float* B, float* C) {
    // A: 4x4, B: 4x2, C: 4x2
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++)
                sum += A[i*4 + k] * B[k*2 + j];
            C[i*2 + j] = sum;
        }
    }
}

__device__ void mat_mul_2x4_4x2(const float* A, const float* B, float* C) {
    // A: 2x4, B: 4x2, C: 2x2
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++)
                sum += A[i*4 + k] * B[k*2 + j];
            C[i*2 + j] = sum;
        }
    }
}

__device__ void mat_mul_4x2_2x4(const float* A, const float* B, float* C) {
    // A: 4x2, B: 2x4, C: 4x4
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 2; k++)
                sum += A[i*2 + k] * B[k*4 + j];
            C[i*4 + j] = sum;
        }
    }
}

__device__ float mat_det_2x2(const float* M) {
    return M[0]*M[3] - M[1]*M[2];
}

__device__ void mat_inv_2x2(const float* M, float* inv) {
    float det = mat_det_2x2(M);
    if (fabsf(det) < 1e-12f) det = 1e-12f;
    float inv_det = 1.0f / det;
    inv[0] =  M[3] * inv_det;
    inv[1] = -M[1] * inv_det;
    inv[2] = -M[2] * inv_det;
    inv[3] =  M[0] * inv_det;
}

__device__ void mat_add_2x2(const float* A, const float* B, float* C) {
    C[0]=A[0]+B[0]; C[1]=A[1]+B[1];
    C[2]=A[2]+B[2]; C[3]=A[3]+B[3];
}

// ─── Standard Kalman Filter (batch) ─────────────────────────────────────────
// Each thread handles one KF predict + update cycle

__global__ void kalman_filter_kernel(
    float* states,              // [N, STATE_DIM]  x_k|k
    float* covariances,         // [N, STATE_DIM*STATE_DIM]  P_k|k
    const float* measurements,  // [N, MEAS_DIM]  z_k
    const float* F,             // [STATE_DIM*STATE_DIM] state transition matrix
    const float* H,             // [MEAS_DIM*STATE_DIM] measurement matrix
    const float* Q,             // [STATE_DIM*STATE_DIM] process noise covariance
    const float* R,             // [MEAS_DIM*MEAS_DIM] measurement noise covariance
    float* innovations,         // [N, MEAS_DIM] output: innovation y_k
    float* kalman_gains,        // [N, STATE_DIM*MEAS_DIM] output: Kalman gain K
    int N,
    float dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float* x = &states[idx * STATE_DIM];
    float* P = &covariances[idx * STATE_DIM * STATE_DIM];
    const float* z = &measurements[idx * MEAS_DIM];

    // ── Predict ─────────────────────────────────────────────────────────
    float x_pred[STATE_DIM];
    float P_pred[STATE_DIM * STATE_DIM];
    float FP[STATE_DIM * STATE_DIM];
    float FPT_FT[STATE_DIM * STATE_DIM];

    // x_pred = F * x
    mat_vec_mul_4x4(F, x, x_pred);

    // P_pred = F * P * F^T + Q
    mat_mul_4x4(F, P, FP);
    float FT[STATE_DIM * STATE_DIM];
    mat_transpose_4x4(F, FT);
    mat_mul_4x4(FP, FT, FPT_FT);
    mat_add_4x4(FPT_FT, Q, P_pred);

    // ── Update ──────────────────────────────────────────────────────────
    float Hx_pred[MEAS_DIM];
    float y[MEAS_DIM];  // innovation = z - H*x_pred
    float HP_pred[MEAS_DIM * STATE_DIM];
    float P_pred_HT[STATE_DIM * MEAS_DIM];
    float S[MEAS_DIM * MEAS_DIM];
    float S_inv[MEAS_DIM * MEAS_DIM];
    float K[STATE_DIM * MEAS_DIM];
    float K_y[STATE_DIM];
    float KH[STATE_DIM * STATE_DIM];
    float KHP[STATE_DIM * STATE_DIM];

    // y = z - H * x_pred
    mat_mul_2x4_4x4(H, (float*)&x_pred, Hx_pred);  // H*x_pred (H is 2x4, x_pred treated as 4x4 but only first col used)
    // Actually H is MEAS_DIM x STATE_DIM, x_pred is STATE_DIM x 1
    // Simpler: direct computation
    Hx_pred[0] = H[0]*x_pred[0] + H[1]*x_pred[1] + H[2]*x_pred[2] + H[3]*x_pred[3];
    Hx_pred[1] = H[4]*x_pred[0] + H[5]*x_pred[1] + H[6]*x_pred[2] + H[7]*x_pred[3];
    y[0] = z[0] - Hx_pred[0];
    y[1] = z[1] - Hx_pred[1];

    // S = H * P_pred * H^T + R
    // H is 2x4, H^T is 4x2: H^T[i][j] = H[j][i]
    float HT[STATE_DIM * MEAS_DIM];  // 4x2
    HT[0] = H[0]; HT[1] = H[4];  // column 0: H^T[0] = H[0,0], H^T[1] = H[1,0]
    HT[2] = H[1]; HT[3] = H[5];
    HT[4] = H[2]; HT[5] = H[6];
    HT[6] = H[3]; HT[7] = H[7];

    mat_mul_2x4_4x4(H, P_pred, HP_pred);  // H*P_pred: 2x4
    // HP_pred * H^T: 2x4 * 4x2 = 2x2
    S[0] = HP_pred[0]*HT[0] + HP_pred[1]*HT[1] + HP_pred[2]*HT[2] + HP_pred[3]*HT[3];
    S[1] = HP_pred[0]*HT[4] + HP_pred[1]*HT[5] + HP_pred[2]*HT[6] + HP_pred[3]*HT[7];
    S[2] = HP_pred[4]*HT[0] + HP_pred[5]*HT[1] + HP_pred[6]*HT[2] + HP_pred[7]*HT[3];
    S[3] = HP_pred[4]*HT[4] + HP_pred[5]*HT[5] + HP_pred[6]*HT[6] + HP_pred[7]*HT[7];
    mat_add_2x2(S, R, S);

    // K = P_pred * H^T * S^{-1}
    mat_mul_4x4_4x2(P_pred, HT, P_pred_HT);  // P_pred * H^T: 4x2
    mat_inv_2x2(S, S_inv);
    // P_pred_HT (4x2) * S_inv (2x2) = K (4x2)
    for (int i = 0; i < 4; i++) {
        K[i*2+0] = P_pred_HT[i*2+0]*S_inv[0] + P_pred_HT[i*2+1]*S_inv[2];
        K[i*2+1] = P_pred_HT[i*2+0]*S_inv[1] + P_pred_HT[i*2+1]*S_inv[3];
    }

    // x = x_pred + K * y
    K_y[0] = K[0]*y[0] + K[1]*y[1];
    K_y[1] = K[2]*y[0] + K[3]*y[1];
    K_y[2] = K[4]*y[0] + K[5]*y[1];
    K_y[3] = K[6]*y[0] + K[7]*y[1];
    x[0] = x_pred[0] + K_y[0];
    x[1] = x_pred[1] + K_y[1];
    x[2] = x_pred[2] + K_y[2];
    x[3] = x_pred[3] + K_y[3];

    // P = (I - K*H) * P_pred
    float KH_temp[STATE_DIM * STATE_DIM];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            KH_temp[i*4+j] = K[i*2+0]*H[j] + K[i*2+1]*H[4+j];
        }
    }
    float I_KH[STATE_DIM * STATE_DIM];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            I_KH[i*4+j] = (i==j ? 1.0f : 0.0f) - KH_temp[i*4+j];
        }
    }
    mat_mul_4x4(I_KH, P_pred, P);

    // Store innovations and gains for analysis
    int innov_base = idx * MEAS_DIM;
    innovations[innov_base + 0] = y[0];
    innovations[innov_base + 1] = y[1];

    int gain_base = idx * STATE_DIM * MEAS_DIM;
    for (int i = 0; i < STATE_DIM * MEAS_DIM; i++)
        kalman_gains[gain_base + i] = K[i];
}


// ─── Extended Kalman Filter (EKF) - radar measurement model ─────────────────
// Radar: z = [range, bearing]
// h(x) = [sqrt(x^2+y^2), atan2(y,x)]
// H = Jacobian of h(x)

__global__ void ekf_batch_kernel(
    float* states,
    float* covariances,
    const float* measurements,
    const float* F, const float* Q, const float* R,
    float* innovations,
    int N, float dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float* x = &states[idx * STATE_DIM];
    float* P = &covariances[idx * STATE_DIM * STATE_DIM];
    const float* z = &measurements[idx * MEAS_DIM];

    // ── Predict (same as KF) ────────────────────────────────────────────
    float x_pred[STATE_DIM];
    float P_pred[STATE_DIM * STATE_DIM];

    mat_vec_mul_4x4(F, x, x_pred);

    float FP[STATE_DIM * STATE_DIM];
    float FT[STATE_DIM * STATE_DIM];
    mat_mul_4x4(F, P, FP);
    mat_transpose_4x4(F, FT);
    mat_mul_4x4(FP, FT, P_pred);
    mat_add_4x4(P_pred, Q, P_pred);

    // ── Update with nonlinear measurement ───────────────────────────────
    float px = x_pred[0], py = x_pred[1];
    float rho = sqrtf(px*px + py*py);
    float phi = atan2f(py, px);

    // Nonlinear measurement prediction
    float z_pred[2] = {rho, phi};
    float y[2] = {z[0] - z_pred[0], z[1] - z_pred[1]};

    // Normalize bearing innovation to [-pi, pi]
    while (y[1] > M_PI)  y[1] -= 2.0f * M_PI;
    while (y[1] < -M_PI) y[1] += 2.0f * M_PI;

    // Jacobian H of h(x) = [sqrt(x^2+y^2), atan2(y,x)]
    // H = [[px/rho, py/rho, 0, 0],
    //      [-py/rho^2, px/rho^2, 0, 0]]
    float H_jac[MEAS_DIM * STATE_DIM];
    float rho_inv = 1.0f / (rho + 1e-10f);
    float rho2_inv = rho_inv * rho_inv;

    H_jac[0] = px * rho_inv;  H_jac[1] = py * rho_inv;  H_jac[2] = 0.0f;  H_jac[3] = 0.0f;
    H_jac[4] = -py * rho2_inv; H_jac[5] = px * rho2_inv; H_jac[6] = 0.0f;  H_jac[7] = 0.0f;

    // S = H * P_pred * H^T + R
    float S[4];
    float HP[MEAS_DIM * STATE_DIM];
    // HP = H (2x4) * P_pred (4x4)
    for (int i = 0; i < MEAS_DIM; i++) {
        for (int j = 0; j < STATE_DIM; j++) {
            float sum = 0.0f;
            for (int k = 0; k < STATE_DIM; k++)
                sum += H_jac[i*STATE_DIM + k] * P_pred[k*STATE_DIM + j];
            HP[i*STATE_DIM + j] = sum;
        }
    }
    // S = HP * H^T + R
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            float sum = 0.0f;
            for (int k = 0; k < STATE_DIM; k++)
                sum += HP[i*STATE_DIM + k] * H_jac[j*STATE_DIM + k];
            S[i*2+j] = sum;
        }
    }
    mat_add_2x2(S, R, S);

    // K = P_pred * H^T * S^{-1}
    float HT[STATE_DIM * MEAS_DIM];
    for (int i = 0; i < STATE_DIM; i++) {
        HT[i*MEAS_DIM + 0] = H_jac[i];        // column 0 of H^T
        HT[i*MEAS_DIM + 1] = H_jac[4 + i];    // column 1 of H^T
    }
    float PHT[STATE_DIM * MEAS_DIM];
    mat_mul_4x4_4x2(P_pred, HT, PHT);
    float S_inv[4];
    mat_inv_2x2(S, S_inv);

    float K[STATE_DIM * MEAS_DIM];
    for (int i = 0; i < 4; i++) {
        K[i*2+0] = PHT[i*2+0]*S_inv[0] + PHT[i*2+1]*S_inv[2];
        K[i*2+1] = PHT[i*2+0]*S_inv[1] + PHT[i*2+1]*S_inv[3];
    }

    // x = x_pred + K*y
    float Ky[4];
    Ky[0] = K[0]*y[0] + K[1]*y[1];
    Ky[1] = K[2]*y[0] + K[3]*y[1];
    Ky[2] = K[4]*y[0] + K[5]*y[1];
    Ky[3] = K[6]*y[0] + K[7]*y[1];
    x[0] = x_pred[0] + Ky[0];
    x[1] = x_pred[1] + Ky[1];
    x[2] = x_pred[2] + Ky[2];
    x[3] = x_pred[3] + Ky[3];

    // P = (I - K*H) * P_pred
    float KH[STATE_DIM * STATE_DIM];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            KH[i*4+j] = K[i*2+0]*H_jac[j] + K[i*2+1]*H_jac[4+j];
        }
    }
    float I_KH[STATE_DIM * STATE_DIM];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            I_KH[i*4+j] = (i==j ? 1.0f : 0.0f) - KH[i*4+j];
    mat_mul_4x4(I_KH, P_pred, P);

    int innov_base = idx * MEAS_DIM;
    innovations[innov_base + 0] = y[0];
    innovations[innov_base + 1] = y[1];
}


// ─── UKF Sigma point propagation kernel ─────────────────────────────────────
// Each block: one filter. Each thread within block: one sigma point.
// 2*STATE_DIM+1 = 9 sigma points per filter

__global__ void ukf_sigma_predict_kernel(
    float* sigma_points_pred,   // [N, 9, STATE_DIM]
    const float* sigma_points,  // [N, 9, STATE_DIM]
    const float* F,
    int N
) {
    int filter_idx = blockIdx.x;
    int sigma_idx = threadIdx.x;
    if (filter_idx >= N || sigma_idx >= 9) return;

    int base = (filter_idx * 9 + sigma_idx) * STATE_DIM;
    float sp[4] = {sigma_points[base], sigma_points[base+1],
                   sigma_points[base+2], sigma_points[base+3]};

    float pred[4];
    mat_vec_mul_4x4(F, sp, pred);

    sigma_points_pred[base+0] = pred[0];
    sigma_points_pred[base+1] = pred[1];
    sigma_points_pred[base+2] = pred[2];
    sigma_points_pred[base+3] = pred[3];
}

// UKF mean + covariance recomposition
__global__ void ukf_recompose_kernel(
    float* states,
    float* covariances,
    const float* sigma_points_pred,
    const float* measurements,
    const float* R,
    float* innovations,
    int N,
    float alpha, float beta, float kappa_val
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float* x = &states[idx * STATE_DIM];
    float* P = &covariances[idx * STATE_DIM * STATE_DIM];
    const float* z = &measurements[idx * MEAS_DIM];

    int n = STATE_DIM;
    float lambda = alpha*alpha * (n + kappa_val) - n;
    float w_m0 = lambda / (n + lambda);
    float w_c0 = w_m0 + (1.0f - alpha*alpha + beta);
    float w_i = 0.5f / (n + lambda);

    // Compute predicted mean: x = Σ w_i * sigma_i
    float x_mean[4] = {0, 0, 0, 0};
    for (int s = 0; s < 9; s++) {
        int base = (idx * 9 + s) * STATE_DIM;
        float weight = (s == 0) ? w_m0 : w_i;
        x_mean[0] += weight * sigma_points_pred[base+0];
        x_mean[1] += weight * sigma_points_pred[base+1];
        x_mean[2] += weight * sigma_points_pred[base+2];
        x_mean[3] += weight * sigma_points_pred[base+3];
    }

    // Compute predicted covariance
    float P_new[16] = {0};
    for (int s = 0; s < 9; s++) {
        int base = (idx * 9 + s) * STATE_DIM;
        float weight = (s == 0) ? w_c0 : w_i;
        float dx[4] = {sigma_points_pred[base+0]-x_mean[0],
                       sigma_points_pred[base+1]-x_mean[1],
                       sigma_points_pred[base+2]-x_mean[2],
                       sigma_points_pred[base+3]-x_mean[3]};
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                P_new[i*4+j] += weight * dx[i] * dx[j];
    }

    // Measurement prediction: for each sigma point, h(sigma) = [range, bearing]
    float z_mean[2] = {0, 0};
    float z_sigma[9 * 2];
    float z_diff[9 * 2];
    float Pxz[8] = {0};   // 4x2 cross-covariance
    float Pzz[4] = {0};   // 2x2 innovation covariance

    for (int s = 0; s < 9; s++) {
        int base = (idx * 9 + s) * STATE_DIM;
        float px = sigma_points_pred[base+0];
        float py = sigma_points_pred[base+1];
        float range = sqrtf(px*px + py*py);
        float bearing = atan2f(py, px);

        float weight = (s == 0) ? w_m0 : w_i;
        z_sigma[s*2+0] = range;
        z_sigma[s*2+1] = bearing;
        z_mean[0] += weight * range;
        z_mean[1] += weight * bearing;
    }

    for (int s = 0; s < 9; s++) {
        float wc = (s == 0) ? w_c0 : w_i;
        z_diff[s*2+0] = z_sigma[s*2+0] - z_mean[0];
        z_diff[s*2+1] = z_sigma[s*2+1] - z_mean[1];
        // Normalize bearing
        while (z_diff[s*2+1] > M_PI)  z_diff[s*2+1] -= 2*M_PI;
        while (z_diff[s*2+1] < -M_PI) z_diff[s*2+1] += 2*M_PI;

        float dx[4] = {sigma_points_pred[(idx*9+s)*STATE_DIM+0]-x_mean[0],
                       sigma_points_pred[(idx*9+s)*STATE_DIM+1]-x_mean[1],
                       sigma_points_pred[(idx*9+s)*STATE_DIM+2]-x_mean[2],
                       sigma_points_pred[(idx*9+s)*STATE_DIM+3]-x_mean[3]};

        for (int i = 0; i < 4; i++) {
            Pxz[i*2+0] += wc * dx[i] * z_diff[s*2+0];
            Pxz[i*2+1] += wc * dx[i] * z_diff[s*2+1];
        }
        for (int i = 0; i < 2; i++) {
            Pzz[i*2+0] += wc * z_diff[s*2+i] * z_diff[s*2+0];
            Pzz[i*2+1] += wc * z_diff[s*2+i] * z_diff[s*2+1];
        }
    }

    // Add measurement noise
    mat_add_2x2(Pzz, R, Pzz);

    // Kalman gain K = Pxz * Pzz^{-1}
    float Pzz_inv[4];
    mat_inv_2x2(Pzz, Pzz_inv);
    float K[8];
    for (int i = 0; i < 4; i++) {
        K[i*2+0] = Pxz[i*2+0]*Pzz_inv[0] + Pxz[i*2+1]*Pzz_inv[2];
        K[i*2+1] = Pxz[i*2+0]*Pzz_inv[1] + Pxz[i*2+1]*Pzz_inv[3];
    }

    // Innovation
    float y[2] = {z[0] - z_mean[0], z[1] - z_mean[1]};
    while (y[1] > M_PI)  y[1] -= 2*M_PI;
    while (y[1] < -M_PI) y[1] += 2*M_PI;

    innovations[idx*2+0] = y[0];
    innovations[idx*2+1] = y[1];

    // Update state: x = x_mean + K*y
    x[0] = x_mean[0] + K[0]*y[0] + K[1]*y[1];
    x[1] = x_mean[1] + K[2]*y[0] + K[3]*y[1];
    x[2] = x_mean[2] + K[4]*y[0] + K[5]*y[1];
    x[3] = x_mean[3] + K[6]*y[0] + K[7]*y[1];

    // Update covariance: P = P_new - K * Pzz * K^T
    float KPzz[8];
    for (int i = 0; i < 4; i++) {
        KPzz[i*2+0] = K[i*2+0]*Pzz[0] + K[i*2+1]*Pzz[2];
        KPzz[i*2+1] = K[i*2+0]*Pzz[1] + K[i*2+1]*Pzz[3];
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float kp_kt = KPzz[i*2+0]*K[j*2+0] + KPzz[i*2+1]*K[j*2+1];
            P[i*4+j] = P_new[i*4+j] - kp_kt;
        }
    }
}

// ─── Host wrappers ──────────────────────────────────────────────────────────

void kalman_filter_batch_cuda(
    torch::Tensor states,
    torch::Tensor covariances,
    torch::Tensor measurements,
    torch::Tensor F, torch::Tensor H,
    torch::Tensor Q, torch::Tensor R,
    torch::Tensor innovations,
    torch::Tensor kalman_gains,
    int N, float dt
) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    kalman_filter_kernel<<<blocks, threads, 0,
        torch::cuda::getCurrentCUDAStream()>>>(
        states.data_ptr<float>(),
        covariances.data_ptr<float>(),
        measurements.data_ptr<float>(),
        F.data_ptr<float>(),
        H.data_ptr<float>(),
        Q.data_ptr<float>(),
        R.data_ptr<float>(),
        innovations.data_ptr<float>(),
        kalman_gains.data_ptr<float>(),
        N, dt);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ekf_batch_cuda(
    torch::Tensor states,
    torch::Tensor covariances,
    torch::Tensor measurements,
    torch::Tensor F, torch::Tensor Q, torch::Tensor R,
    torch::Tensor innovations,
    int N, float dt
) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    ekf_batch_kernel<<<blocks, threads, 0,
        torch::cuda::getCurrentCUDAStream()>>>(
        states.data_ptr<float>(),
        covariances.data_ptr<float>(),
        measurements.data_ptr<float>(),
        F.data_ptr<float>(), Q.data_ptr<float>(), R.data_ptr<float>(),
        innovations.data_ptr<float>(),
        N, dt);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ukf_batch_cuda(
    torch::Tensor states,
    torch::Tensor covariances,
    torch::Tensor sigma_points_pred,
    torch::Tensor sigma_points,
    torch::Tensor measurements,
    torch::Tensor F, torch::Tensor R,
    torch::Tensor innovations,
    int N
) {
    int threads = 9;  // 2*STATE_DIM+1 sigma points
    int blocks = N;

    ukf_sigma_predict_kernel<<<blocks, threads, 0,
        torch::cuda::getCurrentCUDAStream()>>>(
        sigma_points_pred.data_ptr<float>(),
        sigma_points.data_ptr<float>(),
        F.data_ptr<float>(),
        N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int t2 = 256;
    int b2 = (N + t2 - 1) / t2;
    ukf_recompose_kernel<<<b2, t2, 0,
        torch::cuda::getCurrentCUDAStream()>>>(
        states.data_ptr<float>(),
        covariances.data_ptr<float>(),
        sigma_points_pred.data_ptr<float>(),
        measurements.data_ptr<float>(),
        R.data_ptr<float>(),
        innovations.data_ptr<float>(),
        N, 1.0f, 2.0f, 0.0f);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
