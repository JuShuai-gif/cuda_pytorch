"""
GPS + IMU Sensor Fusion - Loosely-Coupled Kalman Filter
=========================================================
IMU provides high-frequency (100Hz) acceleration measurements but drifts over time.
GPS provides low-frequency (1Hz) absolute position with bounded error.
The Kalman Filter fuses both: IMU for smooth high-rate prediction,
GPS for periodic drift correction.

CUDA batch: run the fusion for N vehicles/drones simultaneously.
"""

from pathlib import Path
import math
import time
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline


# ─── Trajectory simulation ───────────────────────────────────────────────────


def generate_trajectory(T_total, dt_imu=0.01, seed=42):
    """Generate a figure-8 trajectory with velocity for a drone."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, T_total, dt_imu)
    n = len(t)

    # Figure-8 pattern
    omega = 0.5  # rad/s
    scale = 30.0
    x_true = scale * np.sin(omega * t)
    y_true = scale * np.sin(2.0 * omega * t) * 0.5
    vx_true = scale * omega * np.cos(omega * t)
    vy_true = scale * omega * np.cos(2.0 * omega * t)

    # True acceleration (derivative of velocity)
    ax_true = -scale * omega**2 * np.sin(omega * t)
    ay_true = -2.0 * scale * omega**2 * np.sin(2.0 * omega * t)

    return (
        t.astype(np.float32),
        x_true.astype(np.float32),
        y_true.astype(np.float32),
        vx_true.astype(np.float32),
        vy_true.astype(np.float32),
        ax_true.astype(np.float32),
        ay_true.astype(np.float32),
    )


def generate_imu_measurements(
    ax_true, ay_true, accel_noise_std=0.3, accel_bias=0.02, seed=123
):
    """Generate noisy IMU accelerometer measurements (body frame = world frame simplified)."""
    rng = np.random.default_rng(seed)
    ax_meas = ax_true + accel_bias + rng.normal(0, accel_noise_std, len(ax_true))
    ay_meas = ay_true + accel_bias + rng.normal(0, accel_noise_std, len(ay_true))
    return ax_meas.astype(np.float32), ay_meas.astype(np.float32)


def generate_gps_measurements(
    x_true, y_true, t, gps_hz=1.0, pos_noise_std=3.0, seed=456
):
    """Generate noisy GPS measurements at lower frequency."""
    rng = np.random.default_rng(seed)
    dt_gps = 1.0 / gps_hz
    total_t = t[-1]

    gps_times = []
    gps_x = []
    gps_y = []
    tg = 0.0
    while tg <= total_t:
        idx = np.searchsorted(t, tg)
        if idx >= len(t):
            idx = len(t) - 1
        gps_times.append(tg)
        gps_x.append(x_true[idx] + rng.normal(0, pos_noise_std))
        gps_y.append(y_true[idx] + rng.normal(0, pos_noise_std))
        tg += dt_gps

    return (
        np.array(gps_times, dtype=np.float32),
        np.array(gps_x, dtype=np.float32),
        np.array(gps_y, dtype=np.float32),
    )


# ─── GPS-only baseline ───────────────────────────────────────────────────────


def gps_only_estimate(gps_times, gps_x, gps_y, t_eval):
    """Linear interpolation between GPS samples."""
    x_est = np.interp(t_eval, gps_times, gps_x)
    y_est = np.interp(t_eval, gps_times, gps_y)
    return x_est, y_est


# ─── IMU-only dead reckoning ─────────────────────────────────────────────────


def imu_only_estimate(ax_meas, ay_meas, dt):
    """Double-integrate IMU acceleration (dead reckoning)."""
    n = len(ax_meas)
    vx = np.zeros(n, dtype=np.float32)
    vy = np.zeros(n, dtype=np.float32)
    x = np.zeros(n, dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)

    for i in range(1, n):
        vx[i] = vx[i - 1] + ax_meas[i - 1] * dt
        vy[i] = vy[i - 1] + ay_meas[i - 1] * dt
        x[i] = x[i - 1] + vx[i - 1] * dt + 0.5 * ax_meas[i - 1] * dt**2
        y[i] = y[i - 1] + vy[i - 1] * dt + 0.5 * ay_meas[i - 1] * dt**2

    return x, y, vx, vy


# ─── CPU GPS+IMU Kalman Filter ───────────────────────────────────────────────


def gps_imu_fusion_cpu(
    ax_meas, ay_meas, gps_times, gps_x, gps_y, t_imu, dt_imu, Q, R_gps
):
    """Loosely-coupled GPS+IMU fusion using Kalman Filter."""
    n = len(t_imu)
    x_est = np.zeros((n, 4), dtype=np.float32)  # [x, y, vx, vy]
    P = np.eye(4, dtype=np.float32) * 10.0

    if len(gps_times) > 0:
        x_est[0, 0] = gps_x[0]
        x_est[0, 1] = gps_y[0]
    gps_idx = 1

    for i in range(1, n):
        ax = ax_meas[i - 1]
        ay = ay_meas[i - 1]
        dt = dt_imu

        # ----- Predict (IMU-driven motion model) -----
        dt2 = 0.5 * dt * dt
        x_pred = np.array(
            [
                x_est[i - 1, 0] + x_est[i - 1, 2] * dt + ax * dt2,
                x_est[i - 1, 1] + x_est[i - 1, 3] * dt + ay * dt2,
                x_est[i - 1, 2] + ax * dt,
                x_est[i - 1, 3] + ay * dt,
            ],
            dtype=np.float32,
        )

        F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        P = F @ P @ F.T + Q

        x_est[i] = x_pred

        # ----- Update (if GPS measurement available at this time) -----
        if gps_idx < len(gps_times) and t_imu[i] >= gps_times[gps_idx] - dt_imu * 0.5:
            H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
            z = np.array([gps_x[gps_idx], gps_y[gps_idx]], dtype=np.float32)

            y_innov = z - H @ x_est[i]
            S = H @ P @ H.T + R_gps
            K = P @ H.T @ np.linalg.inv(S)

            x_est[i] = x_est[i] + K @ y_innov
            P = (np.eye(4) - K @ H) @ P
            gps_idx += 1

    return x_est


# ─── CUDA Extension ──────────────────────────────────────────────────────────


def compile_gps_imu_extension():
    cuda_source = (Path(__file__).parent / "gps_imu_kernel.cu").read_text()
    cpp_source = """
void gps_imu_predict_batch_cuda(
    torch::Tensor states, torch::Tensor covariances,
    torch::Tensor imu_accel_world,
    torch::Tensor Q, int N, float dt
);
void gps_update_batch_cuda(
    torch::Tensor states, torch::Tensor covariances,
    torch::Tensor gps_meas, torch::Tensor R_gps, int N
);
"""
    return load_inline(
        name="gps_imu_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["gps_imu_predict_batch_cuda", "gps_update_batch_cuda"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
    )


# ─── CUDA Batch GPS+IMU Fusion ───────────────────────────────────────────────


def gps_imu_fusion_cuda(
    ax_meas_list,
    ay_meas_list,
    gps_times_list,
    gps_x_list,
    gps_y_list,
    t_imu,
    dt_imu,
    Q_np,
    R_gps_np,
    ext,
):
    """Run fusion for N vehicles in batch on GPU."""
    DEVICE = "cuda"
    N = len(ax_meas_list)
    n_steps = len(t_imu)

    Q = torch.from_numpy(Q_np).to(DEVICE)
    R_gps = torch.from_numpy(R_gps_np).to(DEVICE)

    states = torch.zeros(N, 4, dtype=torch.float32, device=DEVICE)
    cov = torch.eye(4, dtype=torch.float32, device=DEVICE).unsqueeze(0).repeat(N, 1, 1)
    cov = cov.reshape(N, 16) * 10.0

    # Init with first GPS measurement (t=0)
    states_np = states.cpu().numpy()
    for v in range(N):
        if len(gps_x_list[v]) > 0:
            states_np[v, 0] = float(gps_x_list[v][0])
            states_np[v, 1] = float(gps_y_list[v][0])
    states.copy_(torch.from_numpy(states_np).to(DEVICE))

    # Pre-load all IMU data
    imu_data = np.zeros((n_steps - 1, N, 2), dtype=np.float32)
    for v in range(N):
        for i in range(n_steps - 1):
            imu_data[i, v, 0] = ax_meas_list[v][i]
            imu_data[i, v, 1] = ay_meas_list[v][i]

    # Pre-process GPS: map each IMU timestep to nearest GPS measurement
    gps_meas = np.zeros((n_steps, N, 2), dtype=np.float32)
    gps_available = np.zeros(n_steps, dtype=bool)
    for v in range(N):
        gps_idx = 1  # skip t=0 GPS (already used for init)
        for i in range(n_steps):
            if (
                gps_idx < len(gps_times_list[v])
                and t_imu[i] >= gps_times_list[v][gps_idx] - dt_imu * 0.5
            ):
                gps_meas[i, v, 0] = gps_x_list[v][gps_idx]
                gps_meas[i, v, 1] = gps_y_list[v][gps_idx]
                gps_available[i] = True
                gps_idx += 1

    est_states_np = np.zeros((n_steps, N, 4), dtype=np.float32)
    est_states_np[0] = states.cpu().numpy()

    torch.cuda.synchronize()

    for i in range(1, n_steps):
        # Predict step (every IMU sample)
        imu_t = torch.from_numpy(imu_data[i - 1]).to(DEVICE)
        ext.gps_imu_predict_batch_cuda(states, cov, imu_t, Q, N, dt_imu)

        # Update step (if GPS available)
        if gps_available[i]:
            gps_t = torch.from_numpy(gps_meas[i]).to(DEVICE)
            ext.gps_update_batch_cuda(states, cov, gps_t, R_gps, N)

        est_states_np[i] = states.cpu().numpy()

    torch.cuda.synchronize()
    return est_states_np


# ─── Main ────────────────────────────────────────────────────────────────────


def compute_rmse_2d(est_x, est_y, true_x, true_y):
    err = np.sqrt((est_x - true_x) ** 2 + (est_y - true_y) ** 2)
    return np.mean(err), np.max(err)


def main():
    print("=" * 70)
    print("GPS + IMU Sensor Fusion - Loosely-Coupled Kalman Filter")
    print("IMU: 100Hz (drifting)  |  GPS: 1Hz (noisy but absolute)")
    print("=" * 70)

    # ── Simulation parameters ────────────────────────────────────────────
    T_TOTAL = 60.0
    DT_IMU = 0.01  # 100Hz IMU
    GPS_HZ = 1.0  # 1Hz GPS
    N_VEHICLES = 2000  # batch size for CUDA

    print(
        f"\nSimulating {T_TOTAL}s trajectory, IMU={1 / DT_IMU:.0f}Hz, GPS={GPS_HZ}Hz..."
    )

    # Generate master trajectory (all vehicles follow the same path for fair comparison)
    t, x_true, y_true, vx_true, vy_true, ax_true, ay_true = generate_trajectory(
        T_TOTAL, DT_IMU
    )
    ax_meas, ay_meas = generate_imu_measurements(
        ax_true, ay_true, accel_noise_std=0.5, accel_bias=0.0
    )
    gps_times, gps_x, gps_y = generate_gps_measurements(x_true, y_true, t, GPS_HZ)

    # Noise covariances
    Q = (
        np.eye(4, dtype=np.float32) * 0.5
    )  # process noise (tuned to balance GPS vs IMU trust)
    R_gps = (
        np.eye(2, dtype=np.float32) * 9.0
    )  # GPS measurement noise (3m std -> 9m^2 var)

    # ── Single-vehicle comparison ────────────────────────────────────────

    print("\n─── Single Vehicle: Method Comparison ───\n")

    # 1. GPS-only
    x_gps, y_gps = gps_only_estimate(gps_times, gps_x, gps_y, t)
    gps_rmse, gps_max = compute_rmse_2d(x_gps, y_gps, x_true, y_true)

    # 2. IMU-only dead reckoning
    x_imu, y_imu, _, _ = imu_only_estimate(ax_meas, ay_meas, DT_IMU)
    imu_rmse, imu_max = compute_rmse_2d(x_imu, y_imu, x_true, y_true)

    # 3. GPS+IMU fusion (CPU)
    x_est_cpu = gps_imu_fusion_cpu(
        ax_meas, ay_meas, gps_times, gps_x, gps_y, t, DT_IMU, Q, R_gps
    )
    fusion_rmse, fusion_max = compute_rmse_2d(
        x_est_cpu[:, 0], x_est_cpu[:, 1], x_true, y_true
    )

    print(f"  {'Method':<22} {'RMSE(m)':>10} {'Max Error(m)':>14}")
    print(f"  {'─' * 46}")
    print(f"  {'GPS-only (1Hz)':<22} {gps_rmse:>10.3f} {gps_max:>14.3f}")
    print(f"  {'IMU-only (dead reck.)':<22} {imu_rmse:>10.3f} {imu_max:>14.3f}")
    print(f"  {'GPS+IMU Fusion (KF)':<22} {fusion_rmse:>10.3f} {fusion_max:>14.3f}")

    improvement_vs_gps = (gps_rmse - fusion_rmse) / gps_rmse * 100
    improvement_vs_imu = (imu_rmse - fusion_rmse) / imu_rmse * 100
    print(
        f"\n  Fusion improvement: {improvement_vs_gps:.1f}% vs GPS-only, "
        f"{improvement_vs_imu:.1f}% vs IMU-only"
    )

    # ── CUDA batch comparison ────────────────────────────────────────────

    print(f"\n─── CUDA Batch: {N_VEHICLES} Vehicles ───\n")

    ext = compile_gps_imu_extension()

    # Generate per-vehicle noisy measurements (different noise seeds)
    ax_list, ay_list = [], []
    gps_t_list, gps_x_list, gps_y_list = [], [], []
    for v in range(N_VEHICLES):
        ax_v, ay_v = generate_imu_measurements(
            ax_true, ay_true, accel_noise_std=0.5, accel_bias=0.0, seed=123 + v
        )
        gt_v, gx_v, gy_v = generate_gps_measurements(
            x_true, y_true, t, GPS_HZ, seed=456 + v
        )
        ax_list.append(ax_v)
        ay_list.append(ay_v)
        gps_t_list.append(gt_v)
        gps_x_list.append(gx_v)
        gps_y_list.append(gy_v)

    print("  CUDA batch fusion...", end=" ", flush=True)
    start = time.perf_counter()
    est_cuda = gps_imu_fusion_cuda(
        ax_list, ay_list, gps_t_list, gps_x_list, gps_y_list, t, DT_IMU, Q, R_gps, ext
    )
    cuda_time = time.perf_counter() - start
    print(f"{cuda_time:.3f}s")

    # Average RMSE across vehicles
    cuda_rmse_total = 0.0
    for v in range(N_VEHICLES):
        rmse_v, _ = compute_rmse_2d(
            est_cuda[:, v, 0], est_cuda[:, v, 1], x_true, y_true
        )
        cuda_rmse_total += rmse_v
    cuda_rmse_avg = cuda_rmse_total / N_VEHICLES

    # CPU single-vehicle time (scale estimate)
    print("  CPU single-vehicle fusion...", end=" ", flush=True)
    start = time.perf_counter()
    gps_imu_fusion_cpu(ax_meas, ay_meas, gps_times, gps_x, gps_y, t, DT_IMU, Q, R_gps)
    cpu_single_time = time.perf_counter() - start
    cpu_est_time = cpu_single_time * N_VEHICLES
    print(f"{cpu_single_time:.3f}s (×{N_VEHICLES} = ~{cpu_est_time:.1f}s)")

    print(f"\n  {'Metric':<22} {'CUDA Batch':>12} {'CPU (est.)':>12}")
    print(f"  {'─' * 46}")
    print(f"  {'Time':<22} {cuda_time:>11.3f}s {cpu_est_time:>11.1f}s")
    print(f"  {'Speedup':<22} {cpu_est_time / cuda_time:>11.1f}x {'':>12}")
    print(f"  {'Avg RMSE':<22} {cuda_rmse_avg:>11.3f}m {'':>12}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("GPS+IMU fusion provides smooth, high-rate state estimates")
    print("by combining IMU's high frequency with GPS's absolute reference.")
    print("CUDA enables running this fusion for thousands of vehicles in parallel.")
    print("=" * 70)


if __name__ == "__main__":
    main()
