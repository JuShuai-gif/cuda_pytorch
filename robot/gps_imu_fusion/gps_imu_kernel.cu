#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <math.h>

// State: [x, y, vx, vy], 4 dims
// Measurement: [x, y] from GPS, 2 dims
#define NX 4
#define NZ 2

__global__ void gps_imu_predict_kernel(
    float* states,              // [N, NX]
    float* covariances,         // [N, NX*NX]
    const float* imu_accel,     // [N, 2]  world-frame acceleration (ax, ay) for this IMU step
    const float* Q,             // [NX*NX] process noise
    int N,
    float dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float* x = &states[idx * NX];
    float* P = &covariances[idx * NX * NX];
    float ax = imu_accel[idx * 2];
    float ay = imu_accel[idx * 2 + 1];

    // ── Motion model: constant acceleration from IMU ────────────────────
    // x_k+1 = x_k + vx*dt + 0.5*ax*dt^2
    // y_k+1 = y_k + vy*dt + 0.5*ay*dt^2
    // vx_k+1 = vx_k + ax*dt
    // vy_k+1 = vy_k + ay*dt
    // F = [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]
    // B*u = [0.5*ax*dt^2, 0.5*ay*dt^2, ax*dt, ay*dt]

    float dt2 = 0.5f * dt * dt;

    float x_new[4];
    x_new[0] = x[0] + x[2]*dt + ax*dt2;
    x_new[1] = x[1] + x[3]*dt + ay*dt2;
    x_new[2] = x[2] + ax*dt;
    x_new[3] = x[3] + ay*dt;

    // ── Covariance predict: P = F*P*F^T + Q ────────────────────────────
    // F = [1, 0, dt, 0; 0, 1, 0, dt; 0, 0, 1, 0; 0, 0, 0, 1]
    // P_pred = F*P*F^T
    // We compute directly: P_pred = F*P*F^T element by element
    float P_pred[16];

    // P * F^T  (row 0 and 1 involve dt, rows 2 and 3 are identity)
    // P(i,j) * F^T(k,j) = sum over j:
    // For row i of result (P*F^T):
    //   col 0: P[i][0]*1 + P[i][2]*dt
    //   col 1: P[i][1]*1 + P[i][3]*dt
    //   col 2: P[i][2]*1
    //   col 3: P[i][3]*1
    float PFt[16];
    for (int r = 0; r < 4; r++) {
        PFt[r*4+0] = P[r*4+0] + P[r*4+2]*dt;
        PFt[r*4+1] = P[r*4+1] + P[r*4+3]*dt;
        PFt[r*4+2] = P[r*4+2];
        PFt[r*4+3] = P[r*4+3];
    }

    // F * (P*F^T): row 0 = 1*PFt[0] + dt*PFt[2]; row 1 = 1*PFt[1] + dt*PFt[3]; rows 2,3 unchanged
    P_pred[0]  = PFt[0]  + PFt[8]*dt;
    P_pred[1]  = PFt[1]  + PFt[9]*dt;
    P_pred[2]  = PFt[2]  + PFt[10]*dt;
    P_pred[3]  = PFt[3]  + PFt[11]*dt;
    P_pred[4]  = PFt[4]  + PFt[12]*dt;
    P_pred[5]  = PFt[5]  + PFt[13]*dt;
    P_pred[6]  = PFt[6]  + PFt[14]*dt;
    P_pred[7]  = PFt[7]  + PFt[15]*dt;
    P_pred[8]  = PFt[8];
    P_pred[9]  = PFt[9];
    P_pred[10] = PFt[10];
    P_pred[11] = PFt[11];
    P_pred[12] = PFt[12];
    P_pred[13] = PFt[13];
    P_pred[14] = PFt[14];
    P_pred[15] = PFt[15];

    // Add Q
    for (int i = 0; i < 16; i++) {
        P[i] = P_pred[i] + Q[i];
    }
    x[0] = x_new[0];
    x[1] = x_new[1];
    x[2] = x_new[2];
    x[3] = x_new[3];
}

// GPS update kernel: called only when GPS measurement is available
__global__ void gps_update_kernel(
    float* states,              // [N, NX]
    float* covariances,         // [N, NX*NX]
    const float* gps_meas,      // [N, NZ]  GPS position measurement
    const float* R_gps,         // [NZ*NZ] GPS measurement noise
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float* x = &states[idx * NX];
    float* P = &covariances[idx * NX * NX];
    float mx = gps_meas[idx * 2];
    float my = gps_meas[idx * 2 + 1];

    // ── Innovation ─────────────────────────────────────────────────────
    float y[2];
    y[0] = mx - x[0];
    y[1] = my - x[1];

    // ── Innovation covariance S = H*P*H^T + R ──────────────────────────
    // H = [[1,0,0,0], [0,1,0,0]]  →  S = P[0:2, 0:2] + R
    float S[4];
    S[0] = P[0]  + R_gps[0];
    S[1] = P[1]  + R_gps[1];
    S[2] = P[4]  + R_gps[2];
    S[3] = P[5]  + R_gps[3];

    // ── Invert S (2x2) ─────────────────────────────────────────────────
    float det = S[0]*S[3] - S[1]*S[2];
    if (fabsf(det) < 1e-12f) det = 1e-12f;
    float inv_det = 1.0f / det;
    float S_inv[4];
    S_inv[0] =  S[3] * inv_det;
    S_inv[1] = -S[1] * inv_det;
    S_inv[2] = -S[2] * inv_det;
    S_inv[3] =  S[0] * inv_det;

    // ── Kalman gain K = P * H^T * S^{-1} ───────────────────────────────
    // H^T = [1,0; 0,1; 0,0; 0,0] → P*H^T = P[:, 0:2]
    // K = P[:,0:2] * S_inv
    float K[8];
    for (int r = 0; r < 4; r++) {
        K[r*2+0] = P[r*4+0]*S_inv[0] + P[r*4+1]*S_inv[2];
        K[r*2+1] = P[r*4+0]*S_inv[1] + P[r*4+1]*S_inv[3];
    }

    // ── State update: x += K*y ─────────────────────────────────────────
    x[0] += K[0]*y[0] + K[1]*y[1];
    x[1] += K[2]*y[0] + K[3]*y[1];
    x[2] += K[4]*y[0] + K[5]*y[1];
    x[3] += K[6]*y[0] + K[7]*y[1];

    // ── Covariance update: P = (I - K*H) * P ───────────────────────────
    // K*H: K is 4x2, H is 2x4
    // KH[r][c] = K[r][0]*H[0][c] + K[r][1]*H[1][c] = K[r][0]*(c==0) + K[r][1]*(c==1)
    // I - KH: subtract KH from first 2 columns
    float KH[16] = {0};
    for (int r = 0; r < 4; r++) {
        KH[r*4+0] = K[r*2+0];  // K[r][0] * 1
        KH[r*4+1] = K[r*2+1];  // K[r][1] * 1
    }

    // P_new = (I - KH) * P
    float P_new[16];
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) {
                float i_kh = (r == k ? 1.0f : 0.0f) - KH[r*4+k];
                sum += i_kh * P[k*4+c];
            }
            P_new[r*4+c] = sum;
        }
    }
    for (int i = 0; i < 16; i++) P[i] = P_new[i];
}

// ─── Host wrappers ──────────────────────────────────────────────────────────

void gps_imu_predict_batch_cuda(
    torch::Tensor states,
    torch::Tensor covariances,
    torch::Tensor imu_accel_world,
    torch::Tensor Q,
    int N, float dt
) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    gps_imu_predict_kernel<<<blocks, threads, 0,
        torch::cuda::getCurrentCUDAStream()>>>(
        states.data_ptr<float>(),
        covariances.data_ptr<float>(),
        imu_accel_world.data_ptr<float>(),
        Q.data_ptr<float>(),
        N, dt);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gps_update_batch_cuda(
    torch::Tensor states,
    torch::Tensor covariances,
    torch::Tensor gps_meas,
    torch::Tensor R_gps,
    int N
) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    gps_update_kernel<<<blocks, threads, 0,
        torch::cuda::getCurrentCUDAStream()>>>(
        states.data_ptr<float>(),
        covariances.data_ptr<float>(),
        gps_meas.data_ptr<float>(),
        R_gps.data_ptr<float>(),
        N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
