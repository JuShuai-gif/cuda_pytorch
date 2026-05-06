// lecture14_part1.cpp — CS149 第14讲：MSI 缓存一致性协议模拟
// ============================================================================
// 【课程核心概念】
// 在多核系统中，每个核心有自己的私有缓存（L1/L2），但所有核心共享主存。
// 这引出了一个根本问题：当多个核心的缓存中都存有同一内存地址的副本时，
// 如何保证所有核心看到的数据是一致的？这就是"缓存一致性"（Cache Coherence）问题。
//
// MSI 协议是最经典的 snoop-based（嗅探型）缓存一致性协议之一：
//   M (Modified)：   脏数据，独占 —— 系统中只有这一份副本，且与主存不一致
//   S (Shared)：     干净数据，共享 —— 一个或多个核心都持有副本，与主存一致
//   I (Invalid)：    无效 —— 该缓存行不可用或被废除了
//
// 【Snoop-based 协议的工作原理】
// 所有缓存控制器都连接到一条共享总线（bus）上。当一个缓存发出事务（transaction）
// 时，所有其他缓存都"嗅探"（snoop）总线上的地址，检查自己是否有该缓存行，
// 并根据协议规则更新自己的状态。总线天然地充当了"串行化点"——
// 所有一致性事务按总线顺序依次发生，这保证了写原子性（write atomicity）。
//
// 【MSI 协议的核心不变量】
//   1. SWMR（Single Writer, Multiple Reader）：任意时刻，要么至多 1 个核心在 M 状态，
//      要么 0 个或多个核心在 S 状态，但绝不会同时存在 M 和 S。
//   2. 数据值不变量：一个核心从 I→S（或 M）转换时，拿到的是最新写入的值。
//      写操作要么通过总线获取独占权（BusRdX），要么已经在 M 状态直接写。
// ============================================================================
// 编译：g++ -std=c++17 -O2 lecture14_part1.cpp -o lecture14_part1
// 运行：./lecture14_part1

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cassert>
#include <iomanip>

// ============================================================================
// MSI 协议中的缓存行状态
//
// 【状态详解】
// M (Modified, 已修改):
//   - 当前缓存行是有效的、脏的（与主存不一致）、独占的
//   - 该核心可以直接读取和写入，无需总线事务
//   - 如果其他核心要读取此行，必须先从本核心获取脏数据并降级为 S
//
// S (Shared, 共享):
//   - 当前缓存行是有效的、干净的（与主存一致）、可与其他核心共享的
//   - 该核心可以直接读取，无需总线事务
//   - 如果要写入，必须先通过 BusRdX 升级为 M（同时废除其他核心的副本）
//
// I (Invalid, 无效):
//   - 该缓存行不存在或内容是过期的
//   - 任何读取或写入都会触发缓存缺失（cache miss），需要总线事务
// ============================================================================
enum class MsiState {
    M,  // Modified:   有效、脏、独占 —— 系统中唯一副本
    S,  // Shared:     有效、干净、共享 —— 一个或多个缓存持有此副本
    I   // Invalid:    不存在或过期
};

// 将状态枚举转为可读字符串
const char* state_str(MsiState s) {
    switch (s) {
        case MsiState::M: return "M";
        case MsiState::S: return "S";
        case MsiState::I: return "I";
    }
    return "?";
}

// ============================================================================
// 总线事务类型
//
// BusRd (Bus Read):  请求获取共享副本（不需独占）。发起者从 I→S。
// BusRdX (Bus Read Exclusive):  请求获取独占副本（要写入）。
//         发起者从 I→M 或 S→M，同时废除其他核心的所有副本。
// BusWB (Bus Write-Back):  将脏数据写回主存。
//         通常由一个缓存的 snoop 响应触发（M→S 或 M→I 时交出数据）。
// ============================================================================
enum class BusTrans {
    NONE,     // 无事务（缓存命中，无需总线通信）
    BusRd,    // 读请求 —— 获取共享副本（无需独占写权限）
    BusRdX,   // 独占读请求 —— 获取独占副本（意味着后续要写入，废除其他副本）
    BusWB     // 写回 —— 将脏数据写回主存（发生在降级或废除时）
};

const char* bus_str(BusTrans t) {
    switch (t) {
        case BusTrans::NONE:  return "---";
        case BusTrans::BusRd: return "BusRd";
        case BusTrans::BusRdX:return "BusRdX";
        case BusTrans::BusWB: return "BusWB";
    }
    return "???";
}

// 处理器操作：PrRd（处理器读）和 PrWr（处理器写）
enum class ProcOp { PrRd, PrWr };

const char* op_str(ProcOp op) {
    return op == ProcOp::PrRd ? "PrRd" : "PrWr";
}

// ============================================================================
// 单个缓存中的单个缓存行
// 在真实硬件中，每一行还包括 tag 用于地址匹配、valid/dirty 位等
// ============================================================================
struct CacheLine {
    MsiState state = MsiState::I;   // 当前 MSI 状态
    int tag = -1;                     // 地址标签（-1 表示无效）
    int data = 0;                    // 数据值
    bool dirty = false;              // 是否为脏数据
};

// ============================================================================
// 实现 MSI 协议的缓存控制器
//
// 每个 MSI 缓存有 4 个直接映射（direct-mapped）的行（简化，地址按取模映射）。
// 支持三种操作：
//   pr_read:  处理器请求读取 → 可能触发 BusRd（I→S）或 BusWB（M→I 的驱逐）
//   pr_write: 处理器请求写入 → 可能触发 BusRdX（I→M 或 S→M）或无需事务（M 命中）
//   snoop:    嗅探总线事务 → 根据协议规则更新本缓存行的状态
// ============================================================================
class MsiCache {
public:
    MsiCache(int id, const std::string& name) : id_(id), name_(name) {
        // 4 条缓存行（简化：按地址直接映射）
        lines_.resize(4);
    }

    // 处理器请求读取数据
    // 返回值：是否需要以及需要哪种总线事务
    BusTrans pr_read(int addr) {
        int idx = addr % lines_.size();   // 直接映射：地址取模确定所在行
        auto& line = lines_[idx];

        // 命中且有效（M 或 S 状态）→ 无需总线事务
        if (line.tag == addr && line.state != MsiState::I) {
            log("PrRd  命中 — 状态=" + std::string(state_str(line.state)));
            return BusTrans::NONE;
        }

        // 缺失：I → S，通过 BusRd 获取数据
        log("PrRd  缺失 — BusRd (I→S)");
        BusTrans prev = BusTrans::NONE;
        if (line.state == MsiState::M) {
            // 如果当前行是脏数据且地址不同，需要先驱逐（write-back）
            log("  (驱逐脏行 " + std::to_string(line.tag) + ": M→I, BusWB)");
            prev = BusTrans::BusWB;
        }
        // 分配新缓存行：状态设为 S（共享读），数据从总线获取
        allocate(line, addr, MsiState::S);
        // 如果有驱逐写回，返回 BusWB（实际协议会先写回再发 BusRd）
        return (prev != BusTrans::NONE) ? prev : BusTrans::BusRd;
    }

    // 处理器请求写入数据
    BusTrans pr_write(int addr, int value) {
        int idx = addr % lines_.size();
        auto& line = lines_[idx];

        if (line.tag == addr) {
            if (line.state == MsiState::M) {
                // 已经是 M 状态 —— 直接写入，无需总线事务
                line.data = value;
                log("PrWr  命中 M — 无需总线事务");
                return BusTrans::NONE;
            }
            if (line.state == MsiState::S) {
                // S 状态命中 —— 需要升级：S → M 通过 BusRdX（废除其他副本）
                log("PrWr  命中 S — BusRdX (S→M 升级)");
                line.state = MsiState::M;
                line.data = value;
                line.dirty = true;
                return BusTrans::BusRdX;
            }
        }

        // 缺失：I → M，通过 BusRdX 获取独占副本
        log("PrWr  缺失 — BusRdX (I→M)");
        BusTrans prev = BusTrans::NONE;
        if (line.state == MsiState::M) {
            // 驱逐脏行
            log("  (驱逐脏行 " + std::to_string(line.tag) + ": M→I, BusWB)");
            prev = BusTrans::BusWB;
        }
        allocate(line, addr, MsiState::M);
        line.data = value;
        line.dirty = true;
        return (prev != BusTrans::NONE) ? prev : BusTrans::BusRdX;
    }

    // 嗅探总线事务（来自其他缓存的一致性消息）
    // 返回值：如果需要提供脏数据，则返回 BusWB；否则返回 NONE
    BusTrans snoop(BusTrans bus_op, int addr) {
        int idx = addr % lines_.size();
        auto& line = lines_[idx];

        // 不命中本缓存的地址 → 无需处理
        if (line.tag != addr) return BusTrans::NONE;

        switch (bus_op) {
            case BusTrans::BusRd:
                // 另一缓存想要共享副本
                if (line.state == MsiState::M) {
                    // 必须提供脏数据并降级为 S（SWMR 不变量：不能同时 M+S）
                    log("snoop BusRd — 提供数据 (M→S)");
                    line.state = MsiState::S;
                    line.dirty = false;
                    return BusTrans::BusWB;  // 脏数据通过总线写回
                }
                // S 状态：不做任何事（共享行保持共享，无需响应）
                log("snoop BusRd — 保持 S");
                return BusTrans::NONE;

            case BusTrans::BusRdX:
                // 另一缓存想要独占访问 —— 必须废除本地的副本
                if (line.state == MsiState::M) {
                    log("snoop BusRdX — 提供数据 (M→I)");
                    line.state = MsiState::I;
                    return BusTrans::BusWB;  // 脏数据写回供新所有者使用
                }
                if (line.state == MsiState::S) {
                    log("snoop BusRdX — 废除 (S→I)");
                    line.state = MsiState::I;
                    return BusTrans::NONE;  // S 状态数据是干净的，无需写回
                }
                return BusTrans::NONE;

            case BusTrans::BusWB:
                // 写回到主存；这里是观察者，无需状态变化
                return BusTrans::NONE;

            default:
                return BusTrans::NONE;
        }
    }

    // 查询某地址的缓存状态（用于显示）
    MsiState get_state(int addr) const {
        int idx = addr % lines_.size();
        return lines_[idx].tag == addr ? lines_[idx].state : MsiState::I;
    }

    int get_data(int addr) const {
        int idx = addr % lines_.size();
        return lines_[idx].tag == addr ? lines_[idx].data : -1;
    }

    const std::string& name() const { return name_; }

private:
    // 分配一个新的缓存行条目
    void allocate(CacheLine& line, int addr, MsiState s) {
        line.state = s;
        line.tag = addr;
        line.data = 0;
        line.dirty = (s == MsiState::M);  // M 状态标记为脏
    }

    // 日志输出（取消注释可启用详细 trace）
    void log(const std::string& msg) const {
        // 取消下面注释可启用详细跟踪输出:
        // std::cout << "  [缓存 " << name_ << "] " << msg << std::endl;
        (void)msg; // 抑制未使用变量警告
    }

    int id_;
    std::string name_;
    std::vector<CacheLine> lines_;
};

// ============================================================================
// 总线：串行化所有事务并传递 snoop 消息
//
// 总线是 MSI 协议中的核心串行化点（serialization point）。
// 所有缓存一致性事务按总线顺序依次发生，这保证了全局写顺序的一致性。
// 每次事务，总线会通知所有其他缓存（snoop），并收集它们的响应。
// ============================================================================
class Bus {
public:
    // 执行一次总线事务：通知所有其他缓存
    // 返回数据来源（"Memory" 表示主存，或 "Px $" 表示某个缓存的脏数据）
    std::string execute(BusTrans trans, int addr,
                        std::vector<MsiCache>& caches,
                        int requester_id,
                        int& memory_data) {
        if (trans == BusTrans::NONE) return "---";

        std::string supplier = "Memory";   // 默认数据来自主存
        bool need_wb = false;

        // 通知所有其他缓存（snoop）
        for (size_t i = 0; i < caches.size(); ++i) {
            if ((int)i == requester_id) continue;  // 不通知自己
            BusTrans resp = caches[i].snoop(trans, addr);
            if (resp == BusTrans::BusWB) {
                // 某个缓存持有脏数据，需要写出
                memory_data = caches[i].get_data(addr);
                supplier = "P" + std::to_string(i) + " $";  // 标记数据来源
                need_wb = true;
            }
        }

        // 记录总线日志
        if (need_wb) {
            log(std::string(bus_str(trans)) + " → 数据来自 " + supplier);
        } else if (trans == BusTrans::BusRd || trans == BusTrans::BusRdX) {
            log(std::string(bus_str(trans)) + " → 数据来自主存");
        }

        return supplier;
    }

private:
    void log(const std::string& msg) const {
        // 取消下面注释可启用详细跟踪输出:
        // std::cout << "  [总线] " << msg << std::endl;
        (void)msg;
    }
};

// ============================================================================
// 模拟驱动：重放课程幻灯片中的精确示例
//
// 经典的 MSI 示例，涉及 3 个处理器（P1, P2, P3）和地址 X：
//   1. P1 读取 X → 冷缺失，I→S via BusRd，数据来自主存
//   2. P3 读取 X → 冷缺失（对 P3），I→S via BusRd，数据来自主存
//   3. P3 写入 X=42 → S→M 升级 via BusRdX，废除 P1 的 S 副本
//   4. P1 读取 X → 缺失（之前被废除了），I→S via BusRd，数据来自 P3（M→S 降级）
//   5. P1 读取 X → 命中（S 状态）
//   6. P2 写入 X=99 → 缺失，I→M via BusRdX，废除 P1 和 P3 的副本
// ============================================================================
void run_lecture_example() {
    std::cout << "=== CS149 第14讲：MSI 协议 —— 课程幻灯片示例 ===" << std::endl;
    std::cout << std::endl;
    std::cout << std::left
              << std::setw(16) << "处理器操作"
              << std::setw(12) << "P1 状态"
              << std::setw(12) << "P2 状态"
              << std::setw(12) << "P3 状态"
              << std::setw(12) << "总线事务"
              << "数据来源" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    std::vector<MsiCache> caches;
    caches.emplace_back(0, "P1");   // 处理器 1 的缓存
    caches.emplace_back(1, "P2");   // 处理器 2 的缓存
    caches.emplace_back(2, "P3");   // 处理器 3 的缓存

    Bus bus;
    int memory = 0;  // 地址 X 在主存中的初始值

    // 辅助 lambda：打印当前状态表的一行
    auto print_state = [&](const std::string& action, const std::string& bus_trans,
                           const std::string& supplier) {
        std::cout << std::left
                  << std::setw(16) << action
                  << std::setw(12) << state_str(caches[0].get_state(0))
                  << std::setw(12) << state_str(caches[1].get_state(0))
                  << std::setw(12) << state_str(caches[2].get_state(0))
                  << std::setw(12) << bus_trans
                  << supplier << std::endl;
    };

    int addr = 0;   // 地址 X = 0（简化：只追踪一个地址）

    // 步骤 1: P1 读取 X — 冷缺失，BusRd，数据来自主存
    // P1 从 I→S，总线上只有 P1 发起 BusRd
    BusTrans t = caches[0].pr_read(addr);
    std::string src = bus.execute(t, addr, caches, 0, memory);
    print_state("P1 读取 X", bus_str(t), t == BusTrans::NONE ? "---" : src);

    // 步骤 2: P3 读取 X — P3 冷缺失，BusRd，数据来自主存
    // P3 从 I→S，P1 已在 S（snoop BusRd 时保持不变）
    t = caches[2].pr_read(addr);
    src = bus.execute(t, addr, caches, 2, memory);
    print_state("P3 读取 X", bus_str(t), t == BusTrans::NONE ? "---" : src);

    // 步骤 3: P3 写入 X=42 — S 状态升级到 M，BusRdX 废除 P1
    // P3 从 S→M，P1 从 S→I（被 snoop BusRdX 废除）
    t = caches[2].pr_write(addr, 42);
    src = bus.execute(t, addr, caches, 2, memory);
    print_state("P3 写入 X (42)", bus_str(t), t == BusTrans::NONE ? "---" : src);

    // 步骤 4: P1 读取 X — 缺失（之前被废除），BusRd，数据来自 P3 缓存
    // P1 从 I→S，P3 从 M→S（snoop BusRd 时降级并通过 BusWB 提供脏数据）
    t = caches[0].pr_read(addr);
    src = bus.execute(t, addr, caches, 0, memory);
    print_state("P1 读取 X", bus_str(t), t == BusTrans::NONE ? "---" : src);

    // 步骤 5: P1 再次读取 X — 命中（S 状态），无总线事务
    t = BusTrans::NONE;
    print_state("P1 读取 X", "--- (命中)", "---");

    // 步骤 6: P2 写入 X=99 — 缺失，BusRdX，废除 P1 和 P3
    // P2 从 I→M，P1 从 S→I（被 snoop BusRdX 废除），P3 从 S→I
    t = caches[1].pr_write(addr, 99);
    src = bus.execute(t, addr, caches, 1, memory);
    print_state("P2 写入 X (99)", bus_str(t), t == BusTrans::NONE ? "---" : src);

    std::cout << std::endl;
    std::cout << "MSI 不变量维护情况:" << std::endl;
    std::cout << "  1. SWMR（单写多读）：任意时刻，至多 1 个缓存处于 M，或 ≥0 个处于 S，绝不同时存在。" << std::endl;
    std::cout << "  2. 数据值一致性：所有写操作通过总线串行化，保证了全局写顺序。" << std::endl;

    // 验证最终状态
    std::cout << std::endl;
    std::cout << "最终数据值:" << std::endl;
    for (size_t i = 0; i < caches.size(); ++i)
        std::cout << "  P" << (i + 1) << " data = " << caches[i].get_data(addr)
                  << " (状态=" << state_str(caches[i].get_state(addr)) << ")" << std::endl;
}

// ============================================================================
// MSI 状态转换规则的分步演示
//
// 完整列出 MSI 协议中所有可能的状态转换，包括：
//   - 处理器发起操作触发的转换（PrRd / PrWr）
//   - 远程缓存嗅探总线触发的转换（snoop-driven transitions）
// ============================================================================
void demo_msi_transitions() {
    std::cout << std::endl;
    std::cout << "=== MSI 状态转换规则 ===" << std::endl;
    std::cout << std::endl;

    struct Rule { MsiState from; ProcOp op; MsiState to; std::string bus; };
    std::vector<Rule> rules = {
        // 处理器发起的转换
        {MsiState::I, ProcOp::PrRd, MsiState::S, "BusRd        (I→S: 获取共享副本)"},
        {MsiState::I, ProcOp::PrWr, MsiState::M, "BusRdX       (I→M: 获取独占副本)"},
        {MsiState::S, ProcOp::PrRd, MsiState::S, "---          (命中, 无事务)"},
        {MsiState::S, ProcOp::PrWr, MsiState::M, "BusRdX       (S→M 升级, 废除其他副本)"},
        {MsiState::M, ProcOp::PrRd, MsiState::M, "---          (命中, 无事务)"},
        {MsiState::M, ProcOp::PrWr, MsiState::M, "---          (命中, 无事务)"},
    };

    std::cout << std::left
              << std::setw(12) << "起始状态"
              << std::setw(8)  << "操作"
              << std::setw(8)  << "目标状态"
              << std::setw(30) << "总线事务" << std::endl;
    std::cout << std::string(58, '-') << std::endl;

    for (const auto& r : rules) {
        std::cout << std::left
                  << std::setw(12) << state_str(r.from)
                  << std::setw(8)  << op_str(r.op)
                  << std::setw(8)  << state_str(r.to)
                  << r.bus << std::endl;
    }

    std::cout << std::endl;
    std::cout << "远程缓存嗅探触发的转换（snoop-driven transitions）:" << std::endl;
    std::cout << "  S + BusRdX → I   (废除: 另一缓存请求独占, 本地 S 副本无效化)" << std::endl;
    std::cout << "  M + BusRd  → S   (降级: 另一缓存请求读取, 本地 M 副本降为 S 并提供数据)" << std::endl;
    std::cout << "  M + BusRdX → I   (废除: 另一缓存请求独占, 本地 M 副本无效化并提供脏数据)" << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    run_lecture_example();
    demo_msi_transitions();

    std::cout << std::endl;
    std::cout << "=== 核心要点 ===" << std::endl;
    std::cout << "1. MSI 协议通过 SWMR 不变量保证一致性：同时只能 1 个 M 或多个 S，绝不同时存在。" << std::endl;
    std::cout << "2. 即使在 S 状态命中，写入仍需 BusRdX 来废除其他核心的共享副本。" << std::endl;
    std::cout << "3. M→S 降级发生在另一缓存读取脏行时：原持有者提供数据并变为共享状态。" << std::endl;
    std::cout << "4. 总线作为串行化点：所有一致性事务按顺序发生，保证写原子性。" << std::endl;

    return 0;
}
