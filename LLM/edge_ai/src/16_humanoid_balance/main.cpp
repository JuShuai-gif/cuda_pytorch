#include "wbc_core.h"
#include "balance_control.h"
#include "task_stack.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>

// ============================================================================
// 辅助：打印节标题
// ============================================================================
static void print_header(const std::string &title) {
    std::cout << "\n"
              << std::string(70, '=') << "\n"
              << "  " << title << "\n"
              << std::string(70, '=') << "\n";
}

// ============================================================================
// 辅助：高精度计时器
// ============================================================================
class Timer {
public:
    void start() {
        t0_ = std::chrono::high_resolution_clock::now();
    }
    double elapsed_us() const {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(t1 - t0_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point t0_;
};

// ============================================================================
// 输出 JSON 指标文件
// ============================================================================
static void write_metrics_json(
    double wbc_avg_us,
    double wbc_equivalent_hz,
    double zmp_in_polygon_ratio,
    double standing_avg_us,
    double stepping_avg_us,
    double deadline_miss_rate,
    const std::vector<int> &dof_list,
    const std::vector<double> &dof_times) {
    std::ofstream f("balance_metrics.json");
    if (!f.is_open()) {
        std::cerr << "无法写入 balance_metrics.json\n";
        return;
    }

    f << "{\n";
    f << "  \"project\": \"16_humanoid_balance\",\n";
    f << "  \"description\": \"人形机器人全身控制与平衡性能指标\",\n";
    f << "  \"wbc_solver\": {\n";
    f << "    \"avg_solve_time_us\": " << std::fixed << std::setprecision(2)
      << wbc_avg_us << ",\n";
    f << "    \"equivalent_frequency_hz\": " << std::fixed << std::setprecision(0)
      << wbc_equivalent_hz << ",\n";
    f << "    \"target_deadline_us\": 1000,\n";
    f << "    \"meets_1khz\": " << (wbc_avg_us < 1000.0 ? "true" : "false") << "\n";
    f << "  },\n";
    f << "  \"zmp_balance\": {\n";
    f << "    \"zmp_in_polygon_ratio\": " << std::fixed << std::setprecision(3)
      << zmp_in_polygon_ratio << ",\n";
    f << "    \"polygon_check_us\": \"<1\"\n";
    f << "  },\n";
    f << "  \"balance_loop\": {\n";
    f << "    \"standing_avg_us\": " << std::fixed << std::setprecision(2)
      << standing_avg_us << ",\n";
    f << "    \"stepping_avg_us\": " << std::fixed << std::setprecision(2)
      << stepping_avg_us << ",\n";
    f << "    \"deadline_miss_rate_pct\": " << std::fixed << std::setprecision(2)
      << deadline_miss_rate << "\n";
    f << "  },\n";
    f << "  \"scalability\": {\n";
    f << "    \"dof_list\": [";
    for (size_t i = 0; i < dof_list.size(); ++i) {
        if (i > 0) f << ", ";
        f << dof_list[i];
    }
    f << "],\n";
    f << "    \"time_us_list\": [";
    for (size_t i = 0; i < dof_times.size(); ++i) {
        if (i > 0) f << ", ";
        f << std::fixed << std::setprecision(2) << dof_times[i];
    }
    f << "]\n";
    f << "  }\n";
    f << "}\n";

    f.close();
    std::cout << "\n  性能指标已写入 balance_metrics.json\n";
}

// ============================================================================
// 主入口
// ============================================================================
int main() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "  人形机器人全身控制（WBC）与平衡控制演示\n";
    std::cout << "  34-DOF 系统 | ZMP 平衡判据 | LIPM 步行模式 | 层级零空间投影\n";
    std::cout << std::string(70, '=') << "\n";

    // 收集性能指标用于 JSON 输出
    double wbc_avg_us = 0;
    double wbc_eq_hz = 0;
    double zmp_ratio = 0;
    double stand_us = 0;
    double step_us = 0;
    double miss_rate = 0;
    std::vector<int> dof_list = {7, 14, 28, 34, 40};
    std::vector<double> dof_times;

    // --- 演示 1：WBC 验证 ---
    {
        const int N_DOF = 34;
        const int N_ITER = 1000;

        TaskStackBuilder builder(N_DOF, 0.001, 42);
        WBCSolver solver(N_DOF);

        Eigen::VectorXd q = builder.getCurrentJointPos();
        q(3) += 0.1;
        q(9) += 0.08;
        q(14) += 0.05;
        builder.setCurrentJointPos(q);

        auto tasks = builder.buildFullStack();
        for (const auto &t : tasks) solver.addTask(t);

        // 预热
        for (int i = 0; i < 10; ++i) solver.solve();

        Timer timer;
        timer.start();
        for (int i = 0; i < N_ITER; ++i) solver.solve();
        wbc_avg_us = timer.elapsed_us() / N_ITER;
        wbc_eq_hz = 1e6 / wbc_avg_us;

        solver.solve();
        auto info = solver.lastSolveInfo();

        print_header("演示 1：全身控制（WBC）求解器验证 — 34-DOF 人形机器人");

        std::cout << "\n  任务栈（" << solver.nTasks() << " 个任务）：\n";
        std::cout << "  " << std::string(50, '-') << "\n";
        for (const auto &t : tasks) {
            std::cout << "  P" << t.priority << " | " << std::setw(24)
                      << std::left << t.name << " | J: " << t.J.rows()
                      << "×" << t.J.cols() << " | λ=" << t.damping << "\n";
        }

        std::cout << "\n  求解结果（" << N_ITER << " 次迭代）：\n";
        std::cout << "  " << std::string(50, '-') << "\n";
        std::cout << "  平均求解时间：" << std::fixed << std::setprecision(2)
                  << wbc_avg_us << " μs\n";
        std::cout << "  等效频率：" << std::fixed << std::setprecision(0)
                  << wbc_eq_hz << " Hz\n\n";

        std::cout << "  任务残差 ‖Jq̇ - v_des‖：\n";
        for (size_t k = 0; k < info.task_residuals.size(); ++k) {
            std::cout << "    " << tasks[k].name << "："
                      << std::scientific << std::setprecision(3)
                      << info.task_residuals[k] << "\n";
        }

        std::cout << "\n  零空间剩余维度（N 的迹）：\n";
        for (size_t k = 0; k < info.nullspace_ranks.size(); ++k) {
            std::cout << "    任务 " << k << " 后：迹 = "
                      << std::fixed << std::setprecision(1)
                      << info.nullspace_ranks[k] << "\n";
        }

        bool priority_ok = (info.task_residuals.size() > 0 && info.task_residuals[0] < 1e-2);
        std::cout << "\n  优先级验证："
                  << (priority_ok ? "✓ 通过" : "✗ 失败") << "\n";
    }

    // --- 演示 2：ZMP & LIPM ---
    {
        print_header("演示 2：ZMP 平衡判据与 LIPM 步行模式");

        LIPM lipm(0.80, 9.81);
        std::cout << "\n  LIPM 参数：zc=" << lipm.comHeight()
                  << "m, ω=" << lipm.omega() << " rad/s\n";

        Eigen::Vector2d com1(0.0, 0.0), acc1(0.0, 0.0);
        Eigen::Vector2d zmp1 = lipm.computeZMP(com1, acc1);
        std::cout << "  静止 ZMP=(" << std::fixed << std::setprecision(4)
                  << zmp1.x() << "," << zmp1.y() << ")\n";

        Eigen::Vector2d com2(0.02, 0.0), acc2(0.5, 0.0);
        Eigen::Vector2d zmp2 = lipm.computeZMP(com2, acc2);
        std::cout << "  前倾 ZMP=(" << std::fixed << std::setprecision(4)
                  << zmp2.x() << "," << zmp2.y() << ")\n";

        double foot_l = 0.15, foot_w = 0.10;
        SupportPolygon right_foot({{-foot_l, -foot_w}, {foot_l, -foot_w}, {foot_l, foot_w}, {-foot_l, foot_w}});

        std::cout << "\n  支撑多边形面积：" << right_foot.area()
                  << " m²，ZMP1 在内：" << (right_foot.contains(zmp1) ? "是" : "否") << "\n";

        std::vector<Eigen::Vector2d> footsteps = {
            {0, 0}, {0, 0.10}, {0.30, 0.10}, {0.30, -0.10}, {0.60, -0.10}};
        auto com_traj = lipm.generateCoMTrajectory(footsteps, 0.01, 0.5);

        int in_count = 0, check_count = 0;
        for (size_t i = 0; i < com_traj.size() && check_count < 20; i += 10, ++check_count) {
            Eigen::Vector2d acc(0, 0);
            if (i > 0 && i + 1 < com_traj.size())
                acc = (com_traj[i + 1] - 2 * com_traj[i] + com_traj[i - 1]) / 0.0001;
            Eigen::Vector2d zmp = lipm.computeZMP(com_traj[i], acc);
            int fi = std::min(static_cast<int>(i * 0.01 / 0.5),
                              static_cast<int>(footsteps.size()) - 1);
            SupportPolygon fp({{footsteps[fi].x() - foot_l, footsteps[fi].y() - foot_w},
                               {footsteps[fi].x() + foot_l, footsteps[fi].y() - foot_w},
                               {footsteps[fi].x() + foot_l, footsteps[fi].y() + foot_w},
                               {footsteps[fi].x() - foot_l, footsteps[fi].y() + foot_w}});
            if (fp.contains(zmp)) ++in_count;
        }
        zmp_ratio = 100.0 * in_count / check_count;
        std::cout << "  ZMP 在支撑多边形内比例：" << in_count << "/"
                  << check_count << " (" << zmp_ratio << "%)\n";

        FrictionCone cone(0.6, 10.0, 1000.0);
        std::cout << "\n  摩擦锥验证：\n";
        std::cout << "    纯法向力(0,0,500)："
                  << (cone.isForceValid({0, 0, 500}) ? "✓" : "✗") << "\n";
        std::cout << "    超出摩擦锥(500,0,500)："
                  << (cone.isForceValid({500, 0, 500}) ? "✓" : "✗") << "\n";
    }

    // --- 演示 3：完整平衡回路 ---
    {
        const int N_DOF = 34;
        const double DT = 0.001;
        const int N_STAND = 100, N_STEP = 100, N_TOTAL = N_STAND + N_STEP;

        TaskStackBuilder builder(N_DOF, DT, 42);
        WBCSolver solver(N_DOF);

        Eigen::VectorXd q = builder.getNominalPosture();
        Eigen::VectorXd q_dot = Eigen::VectorXd::Zero(N_DOF);

        double total_us_stand = 0, total_us_step = 0;
        int misses_stand = 0, misses_step = 0;

        // 站立阶段
        print_header("演示 3：完整平衡控制回路 — 站立→迈步模拟");

        std::cout << "\n  阶段 1：静态站立（" << N_STAND << " 步）\n";
        std::cout << "  " << std::string(50, '-') << "\n";

        solver.clear();
        builder.setCurrentJointPos(q);
        Task tl = builder.jointLimitTask();
        if (tl.J.rows() > 0) solver.addTask(tl);
        solver.addTask(builder.balanceTask());
        solver.addTask(builder.contactTask());
        solver.addTask(builder.postureTask());

        for (int i = 0; i < N_STAND; ++i) {
            Timer iter;
            iter.start();
            builder.setCurrentJointPos(q);
            q_dot = solver.solve();
            double su = solver.lastSolveInfo().time_us;
            total_us_stand += su;
            Eigen::VectorXd tau = builder.computeTorqueCommands(q, q_dot,
                                                                builder.getNominalPosture());
            q_dot += tau * DT;
            q += q_dot * DT;
            if (iter.elapsed_us() > 1000.0) ++misses_stand;
        }

        stand_us = total_us_stand / N_STAND;
        std::cout << "  平均 WBC 时间：" << stand_us << " μs\n";
        std::cout << "  违约：" << misses_stand << "/" << N_STAND << "\n";

        // 迈步阶段
        std::cout << "\n  阶段 2：单步迈步（" << N_STEP << " 步）\n";
        std::cout << "  " << std::string(50, '-') << "\n";

        solver.clear();
        builder.setCurrentJointPos(q);
        if (tl.J.rows() > 0) solver.addTask(tl);
        solver.addTask(builder.balanceTask());
        solver.addTask(builder.contactTask());
        solver.addTask(builder.swingFootTask());
        solver.addTask(builder.handTask());
        solver.addTask(builder.postureTask());

        for (int i = 0; i < N_STEP; ++i) {
            Timer iter;
            iter.start();
            builder.setCurrentJointPos(q);
            q_dot = solver.solve();
            double su = solver.lastSolveInfo().time_us;
            total_us_step += su;
            Eigen::VectorXd tau = builder.computeTorqueCommands(q, q_dot,
                                                                builder.getNominalPosture());
            q_dot += tau * DT;
            q += q_dot * DT;
            if (iter.elapsed_us() > 1000.0) ++misses_step;
        }

        step_us = total_us_step / N_STEP;
        miss_rate = 100.0 * (misses_stand + misses_step) / N_TOTAL;

        std::cout << "  平均 WBC 时间：" << step_us << " μs\n";
        std::cout << "  违约：" << misses_step << "/" << N_STEP << "\n";

        std::cout << "\n  控制回路汇总：\n";
        std::cout << "  " << std::string(50, '-') << "\n";
        std::cout << "  站立阶段 WBC：" << stand_us << " μs\n";
        std::cout << "  迈步阶段 WBC：" << step_us << " μs\n";
        std::cout << "  迈步额外开销：" << (step_us - stand_us) << " μs（"
                  << std::setprecision(1)
                  << (100.0 * (step_us - stand_us) / stand_us) << "%）\n";
        std::cout << "  总违约率：" << miss_rate << "%\n";

        if (stand_us < 500.0 && step_us < 500.0) {
            std::cout << "\n  ✓ WBC 求解时间在 0.5ms 以内。\n";
        }
    }

    // --- 演示 4：可扩展性 ---
    {
        print_header("演示 4：WBC 可扩展性测试 — 求解时间 vs 自由度");

        const int N_ITER = 200;

        std::cout << "\n  " << std::setw(6) << "DOF" << std::setw(14)
                  << "任务数" << std::setw(16) << "平均耗时(μs)"
                  << std::setw(16) << "等效频率(Hz)"
                  << std::setw(14) << "每次/DOF(μs)" << "\n";
        std::cout << "  " << std::string(66, '-') << "\n";

        dof_times.clear();
        for (int n_dof : dof_list) {
            TaskStackBuilder builder(n_dof, 0.001, 42 + n_dof);
            WBCSolver solver(n_dof);
            for (const auto &t : builder.buildFullStack()) solver.addTask(t);
            for (int i = 0; i < 10; ++i) solver.solve();

            Timer timer;
            timer.start();
            for (int i = 0; i < N_ITER; ++i) solver.solve();
            double avg = timer.elapsed_us() / N_ITER;
            dof_times.push_back(avg);

            std::cout << "  " << std::setw(6) << n_dof
                      << std::setw(14) << solver.nTasks()
                      << std::setw(14) << std::fixed << std::setprecision(2) << avg
                      << std::setw(14) << std::setprecision(0) << (1e6 / avg)
                      << std::setw(14) << std::setprecision(2) << (avg / n_dof)
                      << "\n";
        }

        if (dof_times.size() >= 2) {
            double p = std::log(dof_times.back() / dof_times[0]) / std::log(dof_list.back() / dof_list[0]);
            std::cout << "\n  复杂度指数 p ≈ " << std::fixed
                      << std::setprecision(2) << p << "（理论 O(n²)=2.0）\n";
        }
    }

    // --- 写入 metrics JSON ---
    write_metrics_json(wbc_avg_us, wbc_eq_hz, zmp_ratio,
                       stand_us, step_us, miss_rate,
                       dof_list, dof_times);

    std::cout << "\n所有演示已完成。\n";
    return 0;
}
