"""
Spatio-Temporal Joint Trajectory Optimization
=============================================
Frenet-frame trajectory planning with parallel CUDA cost evaluation.

Generates candidate trajectories by varying:
  - Lateral target offset (d1)
  - Time horizon (T)
  - Target cruise velocity (v_target)

Each candidate is a quintic lateral polynomial + quartic longitudinal polynomial.
Cost functions: jerk, acceleration, reference deviation, obstacle avoidance,
                curvature, centripetal acceleration, velocity deviation, time.

CUDA kernel evaluates all candidates in parallel (one thread per candidate).

Reference: "Optimal Trajectory Generation for Dynamic Street Scenarios
            in a Frenet Frame" (Werling et al., ICRA 2010)
"""

from pathlib import Path
import math
import time
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline


# ─── Polynomial coefficient solvers ──────────────────────────────────────────


def solve_quintic_coeffs(t0, t1, x0, x0_dot, x0_ddot, x1, x1_dot, x1_ddot):
    T = t1 - t0
    if T < 1e-6:
        return np.array([x0, x0_dot, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    A = np.array(
        [
            [1.0, t0, t0**2, t0**3, t0**4, t0**5],
            [0.0, 1.0, 2 * t0, 3 * t0**2, 4 * t0**3, 5 * t0**4],
            [0.0, 0.0, 2.0, 6 * t0, 12 * t0**2, 20 * t0**3],
            [1.0, t1, t1**2, t1**3, t1**4, t1**5],
            [0.0, 1.0, 2 * t1, 3 * t1**2, 4 * t1**3, 5 * t1**4],
            [0.0, 0.0, 2.0, 6 * t1, 12 * t1**2, 20 * t1**3],
        ],
        dtype=np.float64,
    )
    b = np.array([x0, x0_dot, x0_ddot, x1, x1_dot, x1_ddot], dtype=np.float64)
    return np.linalg.solve(A, b).astype(np.float32)


def solve_quartic_coeffs(t0, t1, x0, x0_dot, x0_ddot, x1_dot, x1_ddot):
    T = t1 - t0
    if T < 1e-6:
        return np.array([x0, x0_dot, 0.0, 0.0, 0.0], dtype=np.float32)

    A = np.array(
        [
            [1.0, t0, t0**2, t0**3, t0**4],
            [0.0, 1.0, 2 * t0, 3 * t0**2, 4 * t0**3],
            [0.0, 0.0, 2.0, 6 * t0, 12 * t0**2],
            [0.0, 1.0, 2 * t1, 3 * t1**2, 4 * t1**3],
            [0.0, 0.0, 2.0, 6 * t1, 12 * t1**2],
        ],
        dtype=np.float64,
    )
    b = np.array([x0, x0_dot, x0_ddot, x1_dot, x1_ddot], dtype=np.float64)
    return np.linalg.solve(A, b).astype(np.float32)


# ─── Polynomial evaluation (CPU) ─────────────────────────────────────────────


def eval_poly5(c, t):
    return c[0] + t * (c[1] + t * (c[2] + t * (c[3] + t * (c[4] + t * c[5]))))


def eval_poly5_dot(c, t):
    return c[1] + t * (2 * c[2] + t * (3 * c[3] + t * (4 * c[4] + t * 5 * c[5])))


def eval_poly5_ddot(c, t):
    return 2 * c[2] + t * (6 * c[3] + t * (12 * c[4] + t * 20 * c[5]))


def eval_poly5_dddot(c, t):
    return 6 * c[3] + t * (24 * c[4] + t * 60 * c[5])


def eval_poly4(c, t):
    return c[0] + t * (c[1] + t * (c[2] + t * (c[3] + t * c[4])))


def eval_poly4_dot(c, t):
    return c[1] + t * (2 * c[2] + t * (3 * c[3] + t * 4 * c[4]))


def eval_poly4_ddot(c, t):
    return 2 * c[2] + t * (6 * c[3] + t * 12 * c[4])


def eval_poly4_dddot(c, t):
    return 6 * c[3] + t * 24 * c[4]


# ─── Cost function evaluation (CPU) ──────────────────────────────────────────


def eval_trajectory_cost_cpu(lat_c, lon_c, T, v_target, obstacles, num_steps, weights):
    jerk_int = lat_accel_int = lon_accel_int = ref_dev_int = obstacle_cost = 0.0
    max_curvature = max_centrip = 0.0

    for k in range(num_steps):
        t = T * k / (num_steps - 1)
        d = eval_poly5(lat_c, t)
        d_dot = eval_poly5_dot(lat_c, t)
        d_ddot = eval_poly5_ddot(lat_c, t)
        d_dddot = eval_poly5_dddot(lat_c, t)
        s = eval_poly4(lon_c, t)
        s_dot = eval_poly4_dot(lon_c, t)
        s_ddot = eval_poly4_ddot(lon_c, t)
        s_dddot = eval_poly4_dddot(lon_c, t)

        jerk_int += d_dddot**2 + s_dddot**2
        lat_accel_int += d_ddot**2
        lon_accel_int += s_ddot**2
        ref_dev_int += d**2

        denom = 1.0 + d_dot**2
        kappa = abs(d_ddot) / (denom * math.sqrt(denom) + 1e-6)
        max_curvature = max(max_curvature, kappa)
        max_centrip = max(max_centrip, s_dot**2 * kappa)

        for obs_s, obs_d, obs_r in obstacles:
            ds = s - obs_s
            dd = d - obs_d
            dist = math.sqrt(ds * ds + dd * dd)
            if dist < obs_r * 3.0:
                obstacle_cost += math.exp(-dist / (obs_r + 1e-6))

    inv_n = 1.0 / num_steps
    jerk_int *= inv_n
    lat_accel_int *= inv_n
    lon_accel_int *= inv_n
    ref_dev_int *= inv_n
    obstacle_cost *= inv_n

    s_dot_final = eval_poly4_dot(lon_c, T)
    vel_dev = (s_dot_final - v_target) ** 2

    total = 0.0
    total += weights["w_jerk"] * jerk_int
    total += weights["w_lat_accel"] * lat_accel_int
    total += weights["w_lon_accel"] * lon_accel_int
    total += weights["w_ref_dev"] * ref_dev_int
    total += weights["w_obstacle"] * obstacle_cost
    total += weights["w_vel_target"] * vel_dev
    total += weights["w_time"] * T
    if max_curvature > 0.3:
        total += weights["w_curvature"] * max_curvature
    if max_centrip > 3.0:
        total += weights["w_centripetal"] * max_centrip

    return total


# ─── Candidate generation ────────────────────────────────────────────────────


def generate_candidates(d1_range, T_range, v_range, start_state):
    s0, s0_dot, s0_ddot, d0, d0_dot, d0_ddot = start_state
    t0 = 0.0

    candidates = []
    for T in T_range:
        for v_target in v_range:
            lon_c = solve_quartic_coeffs(t0, T, s0, s0_dot, s0_ddot, v_target, 0.0)
            for d1 in d1_range:
                lat_c = solve_quintic_coeffs(t0, T, d0, d0_dot, d0_ddot, d1, 0.0, 0.0)
                candidates.append((lat_c, lon_c, T, v_target))
    return candidates


# ─── CUDA extension ──────────────────────────────────────────────────────────


def compile_spacetime_extension():
    cuda_source = (Path(__file__).parent / "spacetime_kernel.cu").read_text()
    cpp_source = """
void spacetime_eval_cuda(
    torch::Tensor coeffs,
    torch::Tensor T_values,
    torch::Tensor v_targets,
    torch::Tensor obstacles,
    torch::Tensor total_cost,
    torch::Tensor cost_components,
    torch::Tensor block_best_cost,
    torch::Tensor block_best_idx,
    torch::Tensor global_best_cost,
    torch::Tensor global_best_idx,
    int num_candidates, int num_obstacles, int num_time_steps,
    float w_jerk, float w_lat_accel, float w_lon_accel,
    float w_ref_dev, float w_obstacle, float w_vel_target,
    float w_time, float w_curvature, float w_centripetal
);
"""
    return load_inline(
        name="spacetime_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["spacetime_eval_cuda"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
    )


def spacetime_plan_cuda(candidates, obstacles, num_time_steps, weights, ext):
    DEVICE = "cuda"
    n = len(candidates)
    m = len(obstacles)

    coeffs_np = np.zeros((n, 11), dtype=np.float32)  # a0..a5, b0..b4
    T_np = np.zeros(n, dtype=np.float32)
    v_np = np.zeros(n, dtype=np.float32)

    for i, (lat_c, lon_c, T, v_target) in enumerate(candidates):
        coeffs_np[i, 0:6] = lat_c
        coeffs_np[i, 6:11] = lon_c
        T_np[i] = T
        v_np[i] = v_target

    obstacles_np = np.array(obstacles, dtype=np.float32).reshape(-1, 3)

    threads = 256
    blocks = (n + threads - 1) // threads

    coeffs = torch.from_numpy(coeffs_np).to(DEVICE)
    T_tensor = torch.from_numpy(T_np).to(DEVICE)
    v_tensor = torch.from_numpy(v_np).to(DEVICE)
    obstacles_t = torch.from_numpy(obstacles_np).to(DEVICE)
    total_cost = torch.empty(n, dtype=torch.float32, device=DEVICE)
    cost_components = torch.empty(n * 9, dtype=torch.float32, device=DEVICE)
    block_best_cost = torch.empty(blocks, dtype=torch.float32, device=DEVICE)
    block_best_idx = torch.empty(blocks, dtype=torch.int32, device=DEVICE)
    global_best_cost = torch.empty(1, dtype=torch.float32, device=DEVICE)
    global_best_idx = torch.empty(1, dtype=torch.int32, device=DEVICE)

    torch.cuda.synchronize()

    ext.spacetime_eval_cuda(
        coeffs,
        T_tensor,
        v_tensor,
        obstacles_t,
        total_cost,
        cost_components,
        block_best_cost,
        block_best_idx,
        global_best_cost,
        global_best_idx,
        n,
        m,
        num_time_steps,
        weights["w_jerk"],
        weights["w_lat_accel"],
        weights["w_lon_accel"],
        weights["w_ref_dev"],
        weights["w_obstacle"],
        weights["w_vel_target"],
        weights["w_time"],
        weights["w_curvature"],
        weights["w_centripetal"],
    )

    torch.cuda.synchronize()

    all_costs = total_cost.cpu().numpy()
    best_idx = np.argmin(all_costs)
    components = cost_components.cpu().numpy().reshape(n, 9)

    return candidates[best_idx], all_costs[best_idx], all_costs, components[best_idx]


def spacetime_plan_cpu(candidates, obstacles, num_time_steps, weights):
    best_candidate = None
    best_cost = float("inf")

    for lat_c, lon_c, T, v_target in candidates:
        cost = eval_trajectory_cost_cpu(
            lat_c, lon_c, T, v_target, obstacles, num_time_steps, weights
        )
        if cost < best_cost:
            best_cost = cost
            best_candidate = (lat_c, lon_c, T, v_target)

    return best_candidate, best_cost


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("Spatio-Temporal Joint Trajectory Optimization")
    print("Frenet Frame + Quintic/Quartic Polynomials + Multi-Cost")
    print("Reference: Werling et al., ICRA 2010")
    print("=" * 70)

    start_state = (0.0, 15.0, 0.0, 0.0, 0.0, 0.0)

    obstacles = [
        (30.0, 1.0, 1.5),
        (50.0, -1.5, 1.5),
        (70.0, 2.0, 1.5),
        (90.0, -1.0, 2.0),
        (110.0, 0.0, 1.5),
    ]

    weights = {
        "w_jerk": 1.0,
        "w_lat_accel": 10.0,
        "w_lon_accel": 10.0,
        "w_ref_dev": 5.0,
        "w_obstacle": 50.0,
        "w_vel_target": 5.0,
        "w_time": 1.0,
        "w_curvature": 50.0,
        "w_centripetal": 30.0,
    }

    NUM_TIME_STEPS = 50

    configurations = [
        (
            np.linspace(-3.0, 3.0, 7),
            np.linspace(2.0, 6.0, 5),
            np.linspace(10.0, 20.0, 5),
            "small (175 candidates)",
        ),
        (
            np.linspace(-4.0, 4.0, 25),
            np.linspace(1.0, 8.0, 20),
            np.linspace(5.0, 25.0, 20),
            "large (10000 candidates)",
        ),
    ]

    ext = compile_spacetime_extension()

    for d1_r, T_r, v_r, label in configurations:
        candidates = generate_candidates(d1_r, T_r, v_r, start_state)
        print(f"\n{'─' * 70}")
        print(f"Config: {label}, Obstacles: {len(obstacles)}")
        print(f"{'─' * 70}")

        print("\n[CUDA] Evaluating...")
        start = time.perf_counter()
        best_c, best_cost, all_costs, components = spacetime_plan_cuda(
            candidates, obstacles, NUM_TIME_STEPS, weights, ext
        )
        cuda_time = time.perf_counter() - start

        lat_c_c, lon_c_c, T_c, v_c = best_c
        print(
            f"  Best: d_final={eval_poly5(lat_c_c, T_c):.3f}m T={T_c:.2f}s "
            f"v_target={v_c:.1f}m/s cost={best_cost:.4f}"
        )
        print(f"  Time: {cuda_time:.4f}s")

        print("\n[CPU] Evaluating...")
        start = time.perf_counter()
        best_c, best_cost_cpu = spacetime_plan_cpu(
            candidates, obstacles, NUM_TIME_STEPS, weights
        )
        cpu_time = time.perf_counter() - start

        lat_c_c, lon_c_c, T_c, v_c = best_c
        print(
            f"  Best: d_final={eval_poly5(lat_c_c, T_c):.3f}m T={T_c:.2f}s "
            f"v_target={v_c:.1f}m/s cost={best_cost_cpu:.4f}"
        )
        print(f"  Time: {cpu_time:.4f}s")

        print(f"\n  {'Metric':<20} {'CUDA':>12} {'CPU':>12}")
        print(f"  {'─' * 44}")
        print(f"  {'Total time':<20} {cuda_time:>11.4f}s {cpu_time:>11.4f}s")
        print(f"  {'Speedup':<20} {cpu_time / cuda_time:>11.2f}x {'':>12}")

        # Cost breakdown
        names = [
            "jerk",
            "lat_accel",
            "lon_accel",
            "ref_dev",
            "obstacle",
            "curvature",
            "centripetal",
            "vel_dev",
            "T",
        ]
        print(f"\n  Cost breakdown (CUDA best):")
        for i, name in enumerate(names):
            print(f"    {name:>12}: {components[i]:8.4f}")


if __name__ == "__main__":
    main()
