"""
Batch Kalman Filter - CUDA Parallel State Estimation
======================================================
Three variants for multi-object tracking:
  - KF:   Standard Kalman Filter (linear measurement: position)
  - EKF:  Extended Kalman Filter (nonlinear radar: range + bearing)
  - UKF:  Unscented Kalman Filter (sigma-point, most accurate for nonlinear)

CUDA parallelization: each thread = one filter predict + update cycle.
Typical use case: tracking thousands of objects simultaneously (autonomous
driving, radar tracking, multi-sensor fusion).
"""

from pathlib import Path
import math
import time
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline


# ─── Simulation ──────────────────────────────────────────────────────────────


def simulate_objects(N, T, dt=0.1, seed=42):
    """Simulate N objects with constant velocity + process noise."""
    rng = np.random.default_rng(seed)
    # state: [x, y, vx, vy] for each object
    true_states = np.zeros((T, N, 4), dtype=np.float32)

    # Initial states: random positions, random velocities
    true_states[0, :, 0] = rng.uniform(-50, 50, N)
    true_states[0, :, 1] = rng.uniform(-50, 50, N)
    true_states[0, :, 2] = rng.uniform(-3, 3, N)
    true_states[0, :, 3] = rng.uniform(-3, 3, N)

    q_pos = 0.01  # position process noise std
    q_vel = 0.05  # velocity process noise std

    for t in range(1, T):
        true_states[t, :, 0] = (
            true_states[t - 1, :, 0]
            + dt * true_states[t - 1, :, 2]
            + rng.normal(0, q_pos, N)
        )
        true_states[t, :, 1] = (
            true_states[t - 1, :, 1]
            + dt * true_states[t - 1, :, 3]
            + rng.normal(0, q_pos, N)
        )
        true_states[t, :, 2] = true_states[t - 1, :, 2] + rng.normal(0, q_vel, N)
        true_states[t, :, 3] = true_states[t - 1, :, 3] + rng.normal(0, q_vel, N)

    return true_states


def generate_linear_measurements(true_states, r_std=1.0, seed=123):
    """Linear measurement: directly observe x, y with Gaussian noise."""
    rng = np.random.default_rng(seed)
    T, N, _ = true_states.shape
    measurements = np.zeros((T, N, 2), dtype=np.float32)
    for t in range(T):
        measurements[t, :, 0] = true_states[t, :, 0] + rng.normal(0, r_std, N)
        measurements[t, :, 1] = true_states[t, :, 1] + rng.normal(0, r_std, N)
    return measurements


def generate_radar_measurements(true_states, range_std=0.5, bearing_std=0.02, seed=123):
    """Nonlinear radar measurement: range + bearing with noise."""
    rng = np.random.default_rng(seed)
    T, N, _ = true_states.shape
    measurements = np.zeros((T, N, 2), dtype=np.float32)
    for t in range(T):
        x, y = true_states[t, :, 0], true_states[t, :, 1]
        rho = np.sqrt(x**2 + y**2) + rng.normal(0, range_std, N)
        phi = np.arctan2(y, x) + rng.normal(0, bearing_std, N)
        measurements[t, :, 0] = rho
        measurements[t, :, 1] = phi
    return measurements


# ─── CPU Kalman Filters ──────────────────────────────────────────────────────


def kf_cpu(true_states, measurements, dt, Q, R):
    T, N, _ = true_states.shape
    est_states = np.zeros((T, N, 4), dtype=np.float32)
    P = np.tile(np.eye(4, dtype=np.float32) * 0.1, (N, 1, 1))

    F = np.array(
        [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)

    est_states[0, :, 0:2] = measurements[0]  # init with first measurement
    est_states[0, :, 2:4] = 0  # unknown velocity

    for t in range(1, T):
        for i in range(N):
            x = est_states[t - 1, i]
            # Predict
            x_pred = F @ x
            P_pred = F @ P[i] @ F.T + Q
            # Update
            y = measurements[t, i] - H @ x_pred
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            x_new = x_pred + K @ y
            P_new = (np.eye(4) - K @ H) @ P_pred
            est_states[t, i] = x_new
            P[i] = P_new
    return est_states


def ekf_cpu(true_states, measurements, dt, Q, R):
    T, N, _ = true_states.shape
    est_states = np.zeros((T, N, 4), dtype=np.float32)
    P = np.tile(np.eye(4, dtype=np.float32) * 0.1, (N, 1, 1))

    F = np.array(
        [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )

    est_states[0, :, 0] = measurements[0, :, 0] * np.cos(measurements[0, :, 1])
    est_states[0, :, 1] = measurements[0, :, 0] * np.sin(measurements[0, :, 1])

    for t in range(1, T):
        for i in range(N):
            x = est_states[t - 1, i]
            x_pred = F @ x
            P_pred = F @ P[i] @ F.T + Q

            px, py = x_pred[0], x_pred[1]
            rho = math.sqrt(px**2 + py**2) + 1e-10
            z_pred = np.array([rho, math.atan2(py, px)], dtype=np.float32)

            y = measurements[t, i] - z_pred
            while y[1] > math.pi:
                y[1] -= 2 * math.pi
            while y[1] < -math.pi:
                y[1] += 2 * math.pi

            H_jac = np.array(
                [[px / rho, py / rho, 0, 0], [-py / (rho**2), px / (rho**2), 0, 0]],
                dtype=np.float32,
            )

            S = H_jac @ P_pred @ H_jac.T + R
            K = P_pred @ H_jac.T @ np.linalg.inv(S)
            x_new = x_pred + K @ y
            P_new = (np.eye(4) - K @ H_jac) @ P_pred
            est_states[t, i] = x_new
            P[i] = P_new
    return est_states


def ukf_cpu(true_states, measurements, dt, Q, R, alpha=1.0, beta=2.0, kappa=0.0):
    T, N, _ = true_states.shape
    n = 4
    lam = alpha**2 * (n + kappa) - n
    w_m = np.full(2 * n + 1, 0.5 / (n + lam), dtype=np.float32)
    w_c = np.full(2 * n + 1, 0.5 / (n + lam), dtype=np.float32)
    w_m[0] = lam / (n + lam)
    w_c[0] = w_m[0] + (1 - alpha**2 + beta)

    est_states = np.zeros((T, N, 4), dtype=np.float32)
    P = np.tile(np.eye(4, dtype=np.float32) * 0.1, (N, 1, 1))

    F = np.array(
        [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )

    est_states[0, :, 0] = measurements[0, :, 0] * np.cos(measurements[0, :, 1])
    est_states[0, :, 1] = measurements[0, :, 0] * np.sin(measurements[0, :, 1])

    for t in range(1, T):
        for i in range(N):
            x = est_states[t - 1, i]
            P_i = P[i]

            # Generate sigma points
            L = np.linalg.cholesky((n + lam) * P_i)
            sigma = np.zeros((2 * n + 1, 4), dtype=np.float32)
            sigma[0] = x
            for j in range(n):
                sigma[j + 1] = x + L[:, j]
                sigma[n + j + 1] = x - L[:, j]

            # Predict sigma points
            sigma_pred = np.zeros_like(sigma)
            for j in range(2 * n + 1):
                sigma_pred[j] = F @ sigma[j]

            # Predicted mean
            x_pred = np.sum(w_m[:, None] * sigma_pred, axis=0)

            # Predicted covariance
            P_pred = Q.copy()
            for j in range(2 * n + 1):
                dx = (sigma_pred[j] - x_pred).reshape(-1, 1)
                P_pred += w_c[j] * (dx @ dx.T)

            # Measurement prediction
            z_sigma = np.zeros((2 * n + 1, 2), dtype=np.float32)
            for j in range(2 * n + 1):
                px, py = sigma_pred[j, 0], sigma_pred[j, 1]
                z_sigma[j, 0] = math.sqrt(px**2 + py**2)
                z_sigma[j, 1] = math.atan2(py, px)

            z_mean = np.sum(w_m[:, None] * z_sigma, axis=0)

            y = measurements[t, i] - z_mean
            while y[1] > math.pi:
                y[1] -= 2 * math.pi
            while y[1] < -math.pi:
                y[1] += 2 * math.pi

            # Cross covariance and innovation covariance
            Pxz = np.zeros((4, 2), dtype=np.float32)
            Pzz = np.zeros((2, 2), dtype=np.float32)
            for j in range(2 * n + 1):
                dx = (sigma_pred[j] - x_pred).reshape(-1, 1)
                dz = (z_sigma[j] - z_mean).reshape(-1, 1)
                dz[1, 0] = math.atan2(math.sin(dz[1, 0]), math.cos(dz[1, 0]))
                Pxz += w_c[j] * (dx @ dz.T)
                Pzz += w_c[j] * (dz @ dz.T)

            Pzz += R
            K = Pxz @ np.linalg.inv(Pzz)
            x_new = x_pred + K @ y
            P_new = P_pred - K @ Pzz @ K.T
            est_states[t, i] = x_new
            P[i] = P_new
    return est_states


# ─── CUDA Extension ──────────────────────────────────────────────────────────


def compile_kalman_extension():
    cuda_source = (Path(__file__).parent / "kalman_kernel.cu").read_text()
    cpp_source = """
void kalman_filter_batch_cuda(
    torch::Tensor states, torch::Tensor covariances,
    torch::Tensor measurements,
    torch::Tensor F, torch::Tensor H,
    torch::Tensor Q, torch::Tensor R,
    torch::Tensor innovations, torch::Tensor kalman_gains,
    int N, float dt
);
void ekf_batch_cuda(
    torch::Tensor states, torch::Tensor covariances,
    torch::Tensor measurements,
    torch::Tensor F, torch::Tensor Q, torch::Tensor R,
    torch::Tensor innovations, int N, float dt
);
void ukf_batch_cuda(
    torch::Tensor states, torch::Tensor covariances,
    torch::Tensor sigma_points_pred, torch::Tensor sigma_points,
    torch::Tensor measurements,
    torch::Tensor F, torch::Tensor R,
    torch::Tensor innovations, int N
);
"""
    return load_inline(
        name="kalman_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["kalman_filter_batch_cuda", "ekf_batch_cuda", "ukf_batch_cuda"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
    )


# ─── CUDA Batch KF ───────────────────────────────────────────────────────────


def kf_cuda(true_states, measurements, dt, Q, R, ext):
    DEVICE = "cuda"
    T, N, _ = true_states.shape

    F = np.array(
        [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)

    F_t = torch.from_numpy(F).to(DEVICE)
    H_t = torch.from_numpy(H).to(DEVICE)
    Q_t = torch.from_numpy(Q).to(DEVICE)
    R_t = torch.from_numpy(R).to(DEVICE)

    states = torch.zeros(N, 4, dtype=torch.float32, device=DEVICE)
    states[:, 0:2] = torch.from_numpy(measurements[0]).to(DEVICE)
    cov = (
        torch.eye(4, dtype=torch.float32, device=DEVICE).unsqueeze(0).repeat(N, 1, 1)
        * 0.1
    )
    cov = cov.reshape(N, 16)

    est_states = np.zeros((T, N, 4), dtype=np.float32)
    est_states[0] = states.cpu().numpy()

    innovations_t = torch.empty(N, 2, dtype=torch.float32, device=DEVICE)
    gains_t = torch.empty(N, 8, dtype=torch.float32, device=DEVICE)

    torch.cuda.synchronize()

    for t in range(1, T):
        meas_t = torch.from_numpy(measurements[t]).to(DEVICE)
        ext.kalman_filter_batch_cuda(
            states, cov, meas_t, F_t, H_t, Q_t, R_t, innovations_t, gains_t, N, dt
        )
        est_states[t] = states.cpu().numpy()

    torch.cuda.synchronize()
    return est_states


def ekf_cuda(true_states, measurements, dt, Q, R, ext):
    DEVICE = "cuda"
    T, N, _ = true_states.shape

    F = np.array(
        [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    F_t = torch.from_numpy(F).to(DEVICE)
    Q_t = torch.from_numpy(Q).to(DEVICE)
    R_t = torch.from_numpy(R).to(DEVICE)

    states = torch.zeros(N, 4, dtype=torch.float32, device=DEVICE)
    states[:, 0] = torch.from_numpy(
        measurements[0, :, 0] * np.cos(measurements[0, :, 1])
    ).to(DEVICE)
    states[:, 1] = torch.from_numpy(
        measurements[0, :, 0] * np.sin(measurements[0, :, 1])
    ).to(DEVICE)
    cov = (
        torch.eye(4, dtype=torch.float32, device=DEVICE)
        .unsqueeze(0)
        .repeat(N, 1, 1)
        .reshape(N, 16)
        * 0.1
    )

    est_states = np.zeros((T, N, 4), dtype=np.float32)
    est_states[0] = states.cpu().numpy()

    innovations_t = torch.empty(N, 2, dtype=torch.float32, device=DEVICE)

    torch.cuda.synchronize()

    for t in range(1, T):
        meas_t = torch.from_numpy(measurements[t]).to(DEVICE)
        ext.ekf_batch_cuda(states, cov, meas_t, F_t, Q_t, R_t, innovations_t, N, dt)
        est_states[t] = states.cpu().numpy()

    torch.cuda.synchronize()
    return est_states


def ukf_cuda(true_states, measurements, dt, Q, R, ext):
    DEVICE = "cuda"
    T, N, _ = true_states.shape
    n = 4
    lam = 1.0**2 * (n + 0.0) - n  # alpha=1, kappa=0 -> lambda=0
    # Use alpha=1.0, kappa=0.0, beta=2.0

    # Adjust: for UKF, we need lambda > -n. With alpha=1, kappa=0: lambda=0
    # But Cholesky needs (n+lambda)*P = n*P which is fine

    F = np.array(
        [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    F_t = torch.from_numpy(F).to(DEVICE)
    R_t = torch.from_numpy(R).to(DEVICE)

    states = torch.zeros(N, 4, dtype=torch.float32, device=DEVICE)
    states[:, 0] = torch.from_numpy(
        measurements[0, :, 0] * np.cos(measurements[0, :, 1])
    ).to(DEVICE)
    states[:, 1] = torch.from_numpy(
        measurements[0, :, 0] * np.sin(measurements[0, :, 1])
    ).to(DEVICE)
    cov = (
        torch.eye(4, dtype=torch.float32, device=DEVICE)
        .unsqueeze(0)
        .repeat(N, 1, 1)
        .reshape(N, 16)
        * 0.1
    )

    # Sigma points: N filters, each has 9 sigma points
    sigma_points = torch.zeros(N, 9, 4, dtype=torch.float32, device=DEVICE)
    sigma_pred = torch.zeros(N, 9, 4, dtype=torch.float32, device=DEVICE)

    est_states = np.zeros((T, N, 4), dtype=np.float32)
    est_states[0] = states.cpu().numpy()
    innovations_t = torch.empty(N, 2, dtype=torch.float32, device=DEVICE)

    torch.cuda.synchronize()

    for t in range(1, T):
        # Generate sigma points on CPU (Cholesky), copy to GPU
        states_cpu = states.cpu().numpy()
        cov_cpu = cov.cpu().numpy().reshape(N, 4, 4)
        sigma_np = np.zeros((N, 9, 4), dtype=np.float32)

        for i in range(N):
            x = states_cpu[i]
            P_i = cov_cpu[i]
            L = np.linalg.cholesky((n + lam) * P_i)
            sigma_np[i, 0] = x
            for j in range(n):
                sigma_np[i, j + 1] = x + L[:, j]
                sigma_np[i, n + j + 1] = x - L[:, j]

        sigma_points.copy_(
            torch.from_numpy(sigma_np.reshape(N * 9, 4)).to(DEVICE).view(N, 9, 4)
        )

        meas_t = torch.from_numpy(measurements[t]).to(DEVICE)
        ext.ukf_batch_cuda(
            states, cov, sigma_pred, sigma_points, meas_t, F_t, R_t, innovations_t, N
        )
        est_states[t] = states.cpu().numpy()

    torch.cuda.synchronize()
    return est_states


# ─── Main ────────────────────────────────────────────────────────────────────


def compute_rmse(est, true):
    pos_err = np.sqrt(
        (est[:, :, 0] - true[:, :, 0]) ** 2 + (est[:, :, 1] - true[:, :, 1]) ** 2
    )
    vel_err = np.sqrt(
        (est[:, :, 2] - true[:, :, 2]) ** 2 + (est[:, :, 3] - true[:, :, 3]) ** 2
    )
    return pos_err.mean(), vel_err.mean()


def main():
    print("=" * 70)
    print("Batch Kalman Filter - CUDA Parallel Multi-Object Tracking")
    print("Variants: KF (linear) | EKF (nonlinear radar) | UKF (sigma-point)")
    print("=" * 70)

    N_OBJECTS = 5000
    T_STEPS = 100
    DT = 0.1

    print(f"\nSimulating {N_OBJECTS} objects over {T_STEPS} time steps (dt={DT}s)...")
    true_states = simulate_objects(N_OBJECTS, T_STEPS, DT)

    lin_meas = generate_linear_measurements(true_states)
    radar_meas = generate_radar_measurements(true_states)

    # Noise covariances
    Q = np.eye(4, dtype=np.float32) * 0.01
    R_linear = np.eye(2, dtype=np.float32) * 1.0
    R_radar = np.diag([0.25, 0.0004]).astype(np.float32)  # range_std^2, bearing_std^2

    ext = compile_kalman_extension()

    filters = [
        ("KF  (linear)", kf_cpu, kf_cuda, lin_meas, R_linear),
        ("EKF (radar)", ekf_cpu, ekf_cuda, radar_meas, R_radar),
        ("UKF (radar)", ukf_cpu, ukf_cuda, radar_meas, R_radar),
    ]

    for name, cpu_fn, cuda_fn, meas, R in filters:
        print(f"\n{'─' * 70}")
        print(f"[{name}]")
        print(f"{'─' * 70}")

        print("  CUDA...", end=" ", flush=True)
        start = time.perf_counter()
        est_cuda = cuda_fn(true_states, meas, DT, Q, R, ext)
        cuda_time = time.perf_counter() - start
        pos_rmse_c, vel_rmse_c = compute_rmse(est_cuda, true_states)
        print(f"{cuda_time:.3f}s  pos_RMSE={pos_rmse_c:.3f}  vel_RMSE={vel_rmse_c:.3f}")

        print("  CPU... ", end=" ", flush=True)
        start = time.perf_counter()
        est_cpu = cpu_fn(true_states, meas, DT, Q, R)
        cpu_time = time.perf_counter() - start
        pos_rmse_cpu, vel_rmse_cpu = compute_rmse(est_cpu, true_states)
        print(
            f"{cpu_time:.3f}s  pos_RMSE={pos_rmse_cpu:.3f}  vel_RMSE={vel_rmse_cpu:.3f}"
        )

        print(
            f"  Speedup: {cpu_time / cuda_time:.1f}x  "
            f"(pos RMSE ratio: {pos_rmse_c / pos_rmse_cpu:.4f})"
        )

    # Summary
    print(f"\n{'=' * 70}")
    print("Summary: GPU batch Kalman Filter accelerates multi-object tracking")
    print("by running N filters in parallel (one thread per filter per step).")
    print("UKF is most accurate for nonlinear radar but heaviest computationally.")
    print("=" * 70)


if __name__ == "__main__":
    main()
