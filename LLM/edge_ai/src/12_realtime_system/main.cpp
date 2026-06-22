#include "task.h"
#include "rms_scheduler.h"
#include "edf_scheduler.h"
#include "priority_inversion.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// ============================================================================
// 打印调度时间线
// ============================================================================
void print_timeline(const std::vector<ScheduleEvent> &events, int64_t duration_us) {
    std::cout << "\n"
              << std::string(70, '=') << "\n";
    std::cout << "  调度时间线（前 5000 us）\n";
    std::cout << std::string(70, '=') << "\n";

    int64_t show_until = std::min(duration_us, (int64_t)5000);
    int64_t bucket_us = 100; // 每字符表示 100us

    // 收集每个时间桶内的活跃任务
    std::map<std::string, std::vector<char>> task_lines;
    std::vector<std::string> task_names;
    std::set<std::string> seen;

    for (const auto &ev : events) {
        if (ev.event_type == "START" || ev.event_type == "PREEMPT" || ev.event_type == "COMPLETE" || ev.event_type == "RELEASE") {
            std::string base_name = ev.task_name;
            // 去除任务后缀以便分组
            auto pos = base_name.find("_J");
            if (pos != std::string::npos) {
                base_name = base_name.substr(0, pos);
            }
            if (seen.insert(base_name).second) {
                task_names.push_back(base_name);
            }
        }
    }

    if (task_names.empty()) return;

    int64_t num_buckets = (show_until + bucket_us - 1) / bucket_us;
    for (const auto &name : task_names) {
        task_lines[name] = std::vector<char>(num_buckets, '.');
    }

    // 标记执行时间段
    std::string current_task;
    int64_t current_start = 0;
    for (const auto &ev : events) {
        if (ev.event_type == "START") {
            current_task = ev.task_name;
            current_start = ev.time_us;
        } else if ((ev.event_type == "COMPLETE" || ev.event_type == "PREEMPT") && !current_task.empty()) {
            std::string base = current_task;
            auto pos = base.find("_J");
            if (pos != std::string::npos) base = base.substr(0, pos);

            int64_t end = ev.time_us;
            for (int64_t t = current_start; t < end && t < show_until; t += bucket_us) {
                int idx = static_cast<int>(t / bucket_us);
                if (idx >= 0 && idx < static_cast<int>(num_buckets)) {
                    task_lines[base][idx] = '#';
                }
            }
            current_task.clear();
        }
    }

    std::cout << "  （每字符 = " << bucket_us << "us，'#' = 执行中，'.' = 空闲）\n\n";
    for (const auto &name : task_names) {
        std::cout << "  " << std::setw(12) << std::left << name << " |";
        for (auto ch : task_lines[name]) {
            std::cout << ch;
        }
        std::cout << "|\n";
    }
    std::cout << "  " << std::string(14 + num_buckets + 2, '-') << "\n";
    std::cout << "  时间 (us) -> 0";
    for (int64_t i = 500; i <= show_until; i += 500) {
        int pad = static_cast<int>(i / bucket_us) - 2;
        if (pad > 0) {
            std::cout << std::string(pad, ' ') << (i / 1000) << "k";
        }
    }
    std::cout << "\n";
}

// ============================================================================
// 命令行界面
// ============================================================================
static void print_usage(const char *prog) {
    std::cout << "用法：" << prog << " [选项]\n\n"
              << "实时调度模拟器\n\n"
              << "选项：\n"
              << "  --mode <rms|edf|all>     调度算法（默认：all）\n"
              << "  --tasks <t1:c:p:d,...>   自定义任务集：名称:C:T:D\n"
              << "                            示例：A:1000:4000:4000,B:2000:6000:6000\n"
              << "  --tick <us>               模拟时钟粒度（默认：1000us）\n"
              << "  --demo-inversion          运行优先级反转演示\n"
              << "  --demo-inheritance        运行优先级继承演示\n"
              << "  --timeline                 打印调度时间线\n"
              << "  --help                     显示此帮助信息\n\n"
              << "默认任务集（经典 RMS 示例）：\n"
              << "  任务 A：C=1ms, T=4ms, D=4ms\n"
              << "  任务 B：C=2ms, T=6ms, D=6ms\n"
              << "  任务 C：C=3ms, T=12ms, D=12ms\n";
}

static std::vector<Task> parse_tasks(const std::string &spec) {
    std::vector<Task> tasks;
    std::stringstream ss(spec);
    std::string token;
    int id = 0;
    while (std::getline(ss, token, ',')) {
        std::stringstream ts(token);
        std::string name, c_str, t_str, d_str;
        std::getline(ts, name, ':');
        std::getline(ts, c_str, ':');
        std::getline(ts, t_str, ':');
        std::getline(ts, d_str, ':');
        int64_t c = std::stoll(c_str);
        int64_t t = std::stoll(t_str);
        int64_t d = d_str.empty() ? t : std::stoll(d_str);
        tasks.emplace_back(name, c, t, d, id++);
    }
    return tasks;
}

int main(int argc, char *argv[]) {
    std::string mode = "all";
    std::string task_spec;
    int64_t tick_us = 1000; // 1ms 时钟粒度
    bool demo_inversion = false;
    bool demo_inheritance = false;
    bool show_timeline = false;

    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (arg == "--tasks" && i + 1 < argc) {
            task_spec = argv[++i];
        } else if (arg == "--tick" && i + 1 < argc) {
            tick_us = std::stoll(argv[++i]);
        } else if (arg == "--demo-inversion") {
            demo_inversion = true;
        } else if (arg == "--demo-inheritance") {
            demo_inheritance = true;
        } else if (arg == "--timeline") {
            show_timeline = true;
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    std::cout << std::string(60, '=') << "\n";
    std::cout << "  实时调度模拟器\n";
    std::cout << "  RMS + EDF + 优先级反转演示\n";
    std::cout << std::string(60, '=') << "\n";

    // 运行优先级反转演示（操作系统级演示，非调度器模拟）
    if (demo_inversion) {
        demo_priority_inversion();
    }
    if (demo_inheritance) {
        demo_priority_inheritance();
    }

    if (!demo_inversion && !demo_inheritance) {
        demo_inversion = true;
        demo_inheritance = true;
    }

    // 解析任务集
    std::vector<Task> tasks;
    if (!task_spec.empty()) {
        tasks = parse_tasks(task_spec);
    } else {
        // 默认任务集
        tasks.emplace_back("A", 1000, 4000, 4000, 0);
        tasks.emplace_back("B", 2000, 6000, 6000, 1);
        tasks.emplace_back("C", 3000, 12000, 12000, 2);
    }

    // 运行调度器
    if (mode == "rms" || mode == "all") {
        RMSScheduler rms(tasks, tick_us);
        rms.run();
        if (show_timeline) {
            print_timeline(rms.events(), rms.events().empty() ? 0 :
                                                                rms.events().back().time_us);
        }
    }

    if (mode == "edf" || mode == "all") {
        EDFScheduler edf(tasks, tick_us);
        edf.run();
    }

    return 0;
}
