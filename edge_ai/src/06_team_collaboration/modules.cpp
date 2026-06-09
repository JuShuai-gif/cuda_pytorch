#include "modules.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <queue>
#include <random>
#include <vector>

// ============================================================================
// PerceptionModule: 辅助函数
// ============================================================================

static void generate_image(uint8_t *img, int w, int h, int c, int frame_id) {
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx = (y * w + x) * c;
            int r, g, b;
            if (y < h * 2 / 5) {
                r = 80 + (y * 30 / h) + (frame_id % 20);
                g = 120 + (y * 40 / h);
                b = 200 - (y * 50 / h);
            } else {
                r = 50 + (y - h * 2 / 5) * 30 / h;
                g = 90 + (y - h * 2 / 5) * 20 / h;
                b = 40 + (y - h * 2 / 5) * 10 / h;
            }
            float dx = static_cast<float>(x - (w / 2 + static_cast<int>(std::sin(frame_id * 0.1f) * 60)));
            float dy = static_cast<float>(y - (h / 2 + 40));
            if (dx * dx + dy * dy < 400.0f) {
                r = 200;
                g = 50;
                b = 50;
            }
            int noise = (frame_id * 7 + x * 13 + y * 31) % 15 - 7;
            img[idx + 0] = static_cast<uint8_t>(std::max(0, std::min(255, r + noise)));
            img[idx + 1] = static_cast<uint8_t>(std::max(0, std::min(255, g + noise)));
            img[idx + 2] = static_cast<uint8_t>(std::max(0, std::min(255, b + noise)));
        }
    }
}

static void resize_image(const uint8_t *src, int sw, int sh, int sc,
                         uint8_t *dst, int dw, int dh) {
    float sx_ratio = static_cast<float>(sw) / dw;
    float sy_ratio = static_cast<float>(sh) / dh;
    for (int dy = 0; dy < dh; ++dy) {
        float sy = dy * sy_ratio;
        int sy0 = static_cast<int>(sy);
        int sy1 = std::min(sy0 + 1, sh - 1);
        float fy = sy - sy0;
        for (int dx = 0; dx < dw; ++dx) {
            float sx_f = dx * sx_ratio;
            int sx0 = static_cast<int>(sx_f);
            int sx1 = std::min(sx0 + 1, sw - 1);
            float fx = sx_f - sx0;
            for (int c = 0; c < sc; ++c) {
                float v00 = src[(sy0 * sw + sx0) * sc + c];
                float v10 = src[(sy0 * sw + sx1) * sc + c];
                float v01 = src[(sy1 * sw + sx0) * sc + c];
                float v11 = src[(sy1 * sw + sx1) * sc + c];
                float top = v00 * (1.0f - fx) + v10 * fx;
                float bot = v01 * (1.0f - fx) + v11 * fx;
                float val = top * (1.0f - fy) + bot * fy;
                dst[(dy * dw + dx) * sc + c] = static_cast<uint8_t>(
                    std::max(0.0f, std::min(255.0f, val)));
            }
        }
    }
}

static int sobel_edge_detect(const uint8_t *img, int w, int h, int c,
                             float threshold) {
    const int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    int detections = 0;
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            float sumx = 0.0f, sumy = 0.0f;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int px = (y + ky) * w + (x + kx);
                    float gray = (static_cast<float>(img[px * c + 0])
                                  + img[px * c + 1]
                                  + img[px * c + 2])
                                 / 3.0f;
                    sumx += gray * gx[ky + 1][kx + 1];
                    sumy += gray * gy[ky + 1][kx + 1];
                }
            }
            float mag = std::sqrt(sumx * sumx + sumy * sumy);
            if (mag > threshold) ++detections;
        }
    }
    return detections;
}

// ============================================================================
// PerceptionModule 实现
// ============================================================================

PerceptionModule::PerceptionModule() {
}

PerformanceContract PerceptionModule::get_contract() const {
    PerformanceContract c;
    c.module_name = "PerceptionModule";
    c.team_owner = "感知团队";
    c.latency_p50_us = 50000;
    c.latency_p99_us = 80000;
    c.latency_max_us = 100000;
    c.jitter_max_us = 20000;
    c.min_fps = 15.0;
    c.missed_detections_per_1000 = 50;
    c.planning_timeout_count = 999999;
    c.max_cpu_percent = 60.0;
    c.max_memory_mb = 2048;
    return c;
}

PerceptionOutput PerceptionModule::process(const PerceptionInput &input) {
    auto t0 = std::chrono::high_resolution_clock::now();

    int src_w = input.image_width > 0 ? input.image_width : 640;
    int src_h = input.image_height > 0 ? input.image_height : 480;
    int src_c = 3;
    int dst_w = 320;
    int dst_h = 240;

    std::vector<uint8_t> src(src_w * src_h * src_c);
    generate_image(src.data(), src_w, src_h, src_c, input.frame_id);

    std::vector<uint8_t> resized(dst_w * dst_h * src_c);
    resize_image(src.data(), src_w, src_h, src_c, resized.data(), dst_w, dst_h);

    int detections = sobel_edge_detect(resized.data(), dst_w, dst_h, src_c, 80.0f);

    auto t1 = std::chrono::high_resolution_clock::now();
    int64_t latency = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    PerceptionOutput out;
    out.frame_id = input.frame_id;
    out.num_detections = detections;
    out.latency_us = latency;
    out.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           t0.time_since_epoch())
                           .count();
    return out;
}

// ============================================================================
// PlanningModule: A* 辅助类型和函数
// ============================================================================

struct AStarNode {
    int x, y;
    float g_cost;
    float f_cost;
    int parent_idx;
    bool operator>(const AStarNode &other) const {
        return f_cost > other.f_cost;
    }
};

static float heuristic(int x1, int y1, int x2, int y2) {
    float dx = static_cast<float>(x1 - x2);
    float dy = static_cast<float>(y1 - y2);
    return std::sqrt(dx * dx + dy * dy);
}

// ============================================================================
// PlanningModule 实现
// ============================================================================

PlanningModule::PlanningModule() {
}

PerformanceContract PlanningModule::get_contract() const {
    PerformanceContract c;
    c.module_name = "PlanningModule";
    c.team_owner = "规划与控制团队";
    c.latency_p50_us = 20000;
    c.latency_p99_us = 35000;
    c.latency_max_us = 50000;
    c.jitter_max_us = 10000;
    c.min_fps = 20.0;
    c.missed_detections_per_1000 = 999999;
    c.planning_timeout_count = 10;
    c.max_cpu_percent = 40.0;
    c.max_memory_mb = 512;
    return c;
}

PlanningOutput PlanningModule::process(const PlanningInput &input) {
    auto t0 = std::chrono::high_resolution_clock::now();
    bool timed_out = false;

    const int N = GRID_SIZE;
    std::vector<std::vector<bool>> grid(N, std::vector<bool>(N, false));

    std::mt19937 rng(static_cast<unsigned>(input.frame_id * 7919 + 42));
    int num_obstacles = std::min(input.num_obstacles, N * N / 4);
    std::uniform_int_distribution<int> pos_dist(0, N - 1);

    for (int i = 0; i < num_obstacles; ++i) {
        int ox = pos_dist(rng);
        int oy = pos_dist(rng);
        if ((ox == 0 && oy == 0) || (ox == N - 1 && oy == N - 1)) continue;
        grid[oy][ox] = true;
        if (ox > 0) grid[oy][ox - 1] = true;
        if (ox < N - 1) grid[oy][ox + 1] = true;
        if (oy > 0) grid[oy - 1][ox] = true;
        if (oy < N - 1) grid[oy + 1][ox] = true;
    }

    int start_x = 0, start_y = 0;
    int goal_x = N - 1, goal_y = N - 1;

    std::vector<std::vector<bool>> closed(N, std::vector<bool>(N, false));
    std::vector<std::vector<float>> g_scores(N, std::vector<float>(N, 1e9f));
    std::vector<std::vector<int>> parent_idx(N, std::vector<int>(N, -1));
    std::vector<AStarNode> expanded;

    std::priority_queue<AStarNode, std::vector<AStarNode>, std::greater<AStarNode>> open;
    g_scores[start_y][start_x] = 0.0f;
    open.push({start_x, start_y, 0.0f, heuristic(start_x, start_y, goal_x, goal_y), -1});

    const int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const float move_cost[8] = {1.414f, 1.0f, 1.414f, 1.0f, 1.0f, 1.414f, 1.0f, 1.414f};

    bool found = false;
    int max_expansions = 20000;
    int expansions = 0;

    while (!open.empty() && expansions < max_expansions) {
        AStarNode current = open.top();
        open.pop();
        if (closed[current.y][current.x]) continue;
        closed[current.y][current.x] = true;
        int my_idx = static_cast<int>(expanded.size());
        expanded.push_back(current);
        ++expansions;
        if (current.x == goal_x && current.y == goal_y) {
            found = true;
            break;
        }
        for (int d = 0; d < 8; ++d) {
            int nx = current.x + dx[d];
            int ny = current.y + dy[d];
            if (nx < 0 || nx >= N || ny < 0 || ny >= N) continue;
            if (grid[ny][nx]) continue;
            if (closed[ny][nx]) continue;
            float new_g = current.g_cost + move_cost[d];
            if (new_g < g_scores[ny][nx]) {
                g_scores[ny][nx] = new_g;
                float f = new_g + heuristic(nx, ny, goal_x, goal_y);
                open.push({nx, ny, new_g, f, my_idx});
                parent_idx[ny][nx] = my_idx;
            }
        }
    }

    std::vector<std::pair<int, int>> trajectory;
    if (found) {
        int cx = goal_x, cy = goal_y;
        while (cx != start_x || cy != start_y) {
            trajectory.emplace_back(cx, cy);
            int pid = parent_idx[cy][cx];
            if (pid < 0 || pid >= static_cast<int>(expanded.size())) break;
            cx = expanded[pid].x;
            cy = expanded[pid].y;
        }
        trajectory.emplace_back(start_x, start_y);
        std::reverse(trajectory.begin(), trajectory.end());
    } else {
        timed_out = true;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    int64_t latency = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    PlanningOutput out;
    out.frame_id = input.frame_id;
    out.trajectory = std::move(trajectory);
    out.latency_us = latency;
    out.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           t0.time_since_epoch())
                           .count();
    out.timed_out = timed_out;
    return out;
}

// ============================================================================
// ControlModule 实现
// ============================================================================

ControlModule::ControlModule() {
}

PerformanceContract ControlModule::get_contract() const {
    PerformanceContract c;
    c.module_name = "ControlModule";
    c.team_owner = "控制团队";
    c.latency_p50_us = 5000;
    c.latency_p99_us = 8000;
    c.latency_max_us = 12000;
    c.jitter_max_us = 3000;
    c.min_fps = 50.0;
    c.missed_detections_per_1000 = 999999;
    c.planning_timeout_count = 999999;
    c.max_cpu_percent = 20.0;
    c.max_memory_mb = 128;
    return c;
}

ControlOutput ControlModule::process(const ControlInput &input) {
    auto t0 = std::chrono::high_resolution_clock::now();

    const double Kp = 0.8;
    const double Ki = 0.05;
    const double Kd = 0.15;
    const double dt = 0.01;
    const int steps = 200;

    double pos = 0.0;
    double vel = 0.0;
    double heading = 0.0;

    double integral_speed = 0.0;
    double integral_steer = 0.0;
    double prev_error_speed = 0.0;
    double prev_error_steer = 0.0;

    double throttle = 0.0;
    double steering = 0.0;

    for (int i = 0; i < steps; ++i) {
        double error_speed = input.target_speed - vel;
        integral_speed += error_speed * dt;
        double deriv_speed = (error_speed - prev_error_speed) / dt;
        prev_error_speed = error_speed;
        throttle = Kp * error_speed + Ki * integral_speed + Kd * deriv_speed;
        throttle = std::max(-1.0, std::min(1.0, throttle));

        double error_steer = input.target_angle - heading;
        integral_steer += error_steer * dt;
        double deriv_steer = (error_steer - prev_error_steer) / dt;
        prev_error_steer = error_steer;
        steering = Kp * 0.5 * error_steer + Ki * 0.02 * integral_steer + Kd * 0.1 * deriv_steer;
        steering = std::max(-1.0, std::min(1.0, steering));

        double accel = throttle * 3.0 - vel * 0.2;
        vel += accel * dt;
        heading += steering * vel * dt * 0.5;
        pos += vel * dt;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    int64_t latency = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    ControlOutput out;
    out.frame_id = input.frame_id;
    out.throttle = throttle;
    out.steering = steering;
    out.latency_us = latency;
    out.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           t0.time_since_epoch())
                           .count();
    return out;
}
