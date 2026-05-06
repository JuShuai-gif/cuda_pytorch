// lecture14_part3.cpp — CS149 第14讲：基于目录的缓存一致性协议模拟
// ============================================================================
// 【课程核心概念】
// 本文件对比两种缓存一致性实现方式：snooping（嗅探型）和 directory（目录型）。
//
// Snooping-based 协议（如 MSI、MESI）的问题：
//   - 所有一致性消息以广播方式发送到所有缓存，每个缓存控制器都必须检查每条消息。
//   - 总线带宽成为扩展瓶颈：N 个核心需要广播 N-1 条消息，总计 O(N²) 消息量。
//   - 需要顺序广播总线（ordered broadcast bus），对物理互连有严格要求。
//
// Directory-based 协议的解决方案：
//   - 维护一个中央（或分布式）目录，记录每个缓存行的状态和持有者信息。
//   - 一致性消息只发送到实际持有该缓存行的核心（点对点消息，而非广播）。
//   - 点对点消息可以通过任意互连网络（mesh、ring、crossbar）传递，扩展性好。
//   - 代价：需要额外的目录存储空间（每个缓存行一个 P 位向量）。
//
// 【现代 CPU 的实践】
// Intel Core i7 使用分布式的 L3 目录来跟踪跨核心的缓存行状态。
// 每个 L3 缓存切片（slice）负责一组缓存行的目录条目。
// 这种混合方案结合了 snooping 的低延迟和 directory 的可扩展性。
// ============================================================================
// 编译：g++ -std=c++17 -O2 lecture14_part3.cpp -o lecture14_part3
// 运行：./lecture14_part3

#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <map>
#include <iomanip>
#include <cassert>

// ============================================================================
// Snooping vs Directory：概念性对比
//
// 【Snooping（嗅探）协议】
// 逻辑上所有缓存共享一条总线。当一个缓存发起事务时（如 BusRd），
// 该事务被广播到所有其他缓存。每个缓存控制器"嗅探"总线上的每个地址，
// 检查自己是否持有该缓存行，并根据协议规则更新状态。
// 核心缺点：每条消息都要发给所有人 → O(N²) 的消息量，总线带宽是瓶颈。
//
// 【Directory（目录）协议】
// 目录（可以是集中式或分布式）跟踪每条缓存行的状态：
//   - 当前状态（Uncached / Shared / Modified）
//   - 哪些核心持有该行（owner 字段 + sharers 集合）
// 当一个核心发起请求时，只向目录发送消息。目录查找后，
// 只向需要被通知的核心发送点对点消息（如废除消息、数据转发消息）。
// 没有广播 → 消息量 O(1)～O(N)（废除所有共享者时），远优于 O(N²)。
// ============================================================================

// ============================================================================
// 目录条目状态（简化，类似于 MSI 在目录上的映射）
//
// U (Uncached-未缓存):      没有任何核心持有该缓存行；数据只在主存中
// S (Shared-共享):          一个或多个核心持有只读副本；主存是最新的
// M (Modified-已修改):      恰好一个核心持有脏副本；主存已过期
// ============================================================================
enum class DirState {
    U,   // Uncached:  没有任何处理器持有该缓存行
    S,   // Shared:    一个或多个处理器持有只读副本
    M    // Modified:  恰好一个处理器持有脏副本
};

// 将目录状态转为可读字符串（带中文注释）
const char* dir_state_str(DirState s) {
    switch (s) {
        case DirState::U: return "U (未缓存)";
        case DirState::S: return "S (共享)";
        case DirState::M: return "M (已修改)";
    }
    return "?";
}

// ============================================================================
// 单个缓存行的目录条目
// 记录该缓存行由哪些核心持有哪些状态
// ============================================================================
struct DirEntry {
    DirState state = DirState::U;    // 当前目录状态
    int owner = -1;                  // 持有 M 副本的核心 ID（-1 表示没有）
    std::set<int> sharers;           // 持有 S 副本的核心 ID 集合
};

// ============================================================================
// 每个核心的缓存行（简化版）
// 本地状态镜像（mirror）目录中的记录
// ============================================================================
struct CoreLine {
    bool valid = false;              // 该缓存行是否有效
    int tag = -1;                    // 地址标签
    DirState state = DirState::U;    // 本地状态（与目录保持一致）
    int data = 0;                    // 数据值
    bool dirty = false;              // 是否为脏数据
};

// ============================================================================
// Core（处理器 + 私有缓存）
// 每个核心有自己的私有缓存，并通过互连网络与目录通信
// ============================================================================
class Core {
public:
    Core(int id) : id_(id) {
        lines_.resize(4);  // 4 条直接映射缓存行（简化）
    }

    int id() const { return id_; }

    // 读请求：返回 true 表示需要向目录发起查找（缓存缺失）
    bool read(int addr) {
        int idx = addr % lines_.size();
        auto& line = lines_[idx];
        if (line.valid && line.tag == addr) {
            log("读取命中, 状态=" + std::string(dir_state_str(line.state)));
            return false;  // 命中 —— 无需目录查找
        }
        log("读取缺失 → 需要目录查找");
        return true;
    }

    // 写请求：返回 true 表示需要向目录发起写入请求
    // 只有本地处于 M 状态时才能直接写入（静默写）
    bool write(int addr, int val) {
        int idx = addr % lines_.size();
        auto& line = lines_[idx];
        if (line.valid && line.tag == addr && line.state == DirState::M) {
            line.data = val;
            log("写入命中 (M 状态), 无需目录请求");
            return false;
        }
        log("写入缺失或需要升级 → 需要目录查找");
        return true;
    }

    // 目录告知核心以指定状态加载缓存行（目录响应消息）
    void load_line(int addr, DirState st, int data) {
        int idx = addr % lines_.size();
        auto& line = lines_[idx];
        line.valid = true;
        line.tag = addr;
        line.state = st;
        line.data = data;
        line.dirty = (st == DirState::M);  // M 状态标记为脏
        log("加载缓存行, 状态=" + std::string(dir_state_str(st)));
    }

    // 目录发来的废除请求（点对点消息，类似 snoop，但只发送给相关核心）
    void invalidate(int addr) {
        int idx = addr % lines_.size();
        auto& line = lines_[idx];
        if (line.valid && line.tag == addr) {
            log("收到废除请求 → 状态变为 I");
            line.valid = false;
            if (line.dirty)
                log("  (脏数据将返回给目录)");
        }
    }

    // 查询某个地址的缓存行状态
    DirState line_state(int addr) const {
        int idx = addr % lines_.size();
        const auto& line = lines_[idx];
        return (line.valid && line.tag == addr) ? line.state : DirState::U;
    }

    int line_data(int addr) const {
        int idx = addr % lines_.size();
        return lines_[idx].tag == addr ? lines_[idx].data : -1;
    }

private:
    void log(const std::string& msg) const {
        // 取消下面注释可启用详细跟踪输出:
        // std::cout << "  [核心 " << id_ << "] " << msg << std::endl;
        (void)msg;
    }
    int id_;
    std::vector<CoreLine> lines_;
};

// ============================================================================
// Directory（目录）：集中式的一致性控制器
//
// 目录是目录协议的核心。它类似于一个"电话簿"，记录每个缓存行的状态：
//   - 谁持有（哪些核心）
//   - 持有什么状态（U/S/M）
//
// 当核心发出读写请求时，目录根据当前状态做出决策：
//   1. 如果状态转移合法，更新目录条目
//   2. 如果需要废除其他核心，发送点对点废除消息
//   3. 如果需要从其他核心获取脏数据，发送转发消息
//   4. 将响应（数据 + 允许的状态）返回给请求核心
//
// 与 snooping 的关键区别：所有通信都是点对点的（无广播！）
// ============================================================================
class Directory {
public:
    Directory(int num_cores, int num_lines)
        : num_cores_(num_cores) {
        entries_.resize(num_lines);  // 每个缓存行一个目录条目
    }

    // 处理核心的读请求
    // 在实际系统中，这是一个消息交换过程：
    //   核心 → 目录（请求消息）
    //   目录 → 主存/持有者（转发消息，如果需要）
    //   目录 → 核心（响应消息，携带数据）
    int handle_read(int core_id, int addr, int memory_val,
                    std::vector<Core>& cores, bool verbose) {
        auto& entry = entries_[addr % entries_.size()];

        if (verbose) {
            std::cout << "  目录[" << addr << "]: 状态=" << dir_state_str(entry.state);
            if (entry.owner >= 0)
                std::cout << ", 持有者=P" << entry.owner;
            if (!entry.sharers.empty()) {
                std::cout << ", 共享者={";
                bool first = true;
                for (int s : entry.sharers) {
                    if (!first) std::cout << ",";
                    std::cout << "P" << s;
                    first = false;
                }
                std::cout << "}";
            }
            std::cout << std::endl;
        }

        switch (entry.state) {
            case DirState::U:
                // 无人持有该行 — 从主存加载，进入 S 状态
                entry.state = DirState::S;
                entry.sharers.insert(core_id);
                cores[core_id].load_line(addr, DirState::S, memory_val);
                if (verbose) std::cout << "    → U→S, 数据来自主存" << std::endl;
                return memory_val;

            case DirState::S:
                // 已有一个或多个核心持有 — 新增共享者，从主存获取数据
                entry.sharers.insert(core_id);
                cores[core_id].load_line(addr, DirState::S, memory_val);
                if (verbose) std::cout << "    → 保持 S, 数据来自主存, 新增 P" << core_id << " 为共享者" << std::endl;
                return memory_val;

            case DirState::M: {
                // 另一个核心持有脏副本！
                // 步骤 1: 告知持有者提供脏数据并降级为 S
                int owner = entry.owner;
                int dirty_data = cores[owner].line_data(addr);
                cores[owner].load_line(addr, DirState::S, dirty_data);  // 降级 M→S
                // 步骤 2: 更新目录：M→S，将原持有者和请求者都加入共享者集合
                entry.state = DirState::S;
                entry.owner = -1;
                entry.sharers.insert(owner);
                entry.sharers.insert(core_id);
                // 步骤 3: 将脏数据提供给请求核心
                cores[core_id].load_line(addr, DirState::S, dirty_data);
                if (verbose) std::cout << "    → M→S (降级 P" << owner << "), 数据来自 P" << owner << " 缓存" << std::endl;
                return dirty_data;
            }
        }
        return memory_val;
    }

    // 处理核心的写请求
    int handle_write(int core_id, int addr, int val, int memory_val,
                     std::vector<Core>& cores, bool verbose) {
        auto& entry = entries_[addr % entries_.size()];

        if (verbose) {
            std::cout << "  目录[" << addr << "]: 状态=" << dir_state_str(entry.state);
            if (entry.owner >= 0)
                std::cout << ", 持有者=P" << entry.owner;
            if (!entry.sharers.empty()) {
                std::cout << ", 共享者={";
                bool first = true;
                for (int s : entry.sharers) {
                    if (!first) std::cout << ",";
                    std::cout << "P" << s;
                    first = false;
                }
                std::cout << "}";
            }
            std::cout << std::endl;
        }

        switch (entry.state) {
            case DirState::U:
                // 无人持有 — 直接分配独占访问
                entry.state = DirState::M;
                entry.owner = core_id;
                cores[core_id].load_line(addr, DirState::M, val);
                if (verbose) std::cout << "    → U→M, P" << core_id << " 成为新持有者" << std::endl;
                return val;

            case DirState::S: {
                // 废除所有共享者（点对点消息，不是广播！）
                // 每个共享者都会收到一条单独的废除消息
                for (int s : entry.sharers) {
                    if (s != core_id) {
                        cores[s].invalidate(addr);
                        if (verbose) std::cout << "    → 废除 P" << s << " (点对点消息)" << std::endl;
                    }
                }
                entry.sharers.clear();
                entry.state = DirState::M;
                entry.owner = core_id;
                cores[core_id].load_line(addr, DirState::M, val);
                if (verbose) std::cout << "    → S→M, 已废除所有共享者" << std::endl;
                return val;
            }

            case DirState::M: {
                // 另一个核心持有脏副本 — 废除原持有者，转移所有权
                int old_owner = entry.owner;
                if (old_owner != core_id) {
                    int dirty_data = cores[old_owner].line_data(addr);
                    cores[old_owner].invalidate(addr);
                    entry.owner = core_id;
                    cores[core_id].load_line(addr, DirState::M, val);
                    if (verbose) std::cout << "    → M→M (废除 P" << old_owner << "), 新持有者 P" << core_id << std::endl;
                    return val;
                }
                // 同一个核心已经是持有者 — 直接写入
                cores[core_id].load_line(addr, DirState::M, val);
                return val;
            }
        }
        return val;
    }

    // 打印当前目录和所有核心的缓存状态
    void print_state(int addr, std::vector<Core>& cores) const {
        const auto& entry = entries_[addr % entries_.size()];
        std::cout << "  目录[" << addr << "]: " << dir_state_str(entry.state);
        if (entry.owner >= 0)
            std::cout << " 持有者=P" << entry.owner;
        if (!entry.sharers.empty()) {
            std::cout << " 共享者={";
            for (auto it = entry.sharers.begin(); it != entry.sharers.end(); ++it) {
                if (it != entry.sharers.begin()) std::cout << ",";
                std::cout << "P" << *it;
            }
            std::cout << "}";
        }
        std::cout << std::endl;
        for (size_t i = 0; i < cores.size(); ++i) {
            auto st = cores[i].line_state(addr);
            std::cout << "  P" << i << " 缓存: " << dir_state_str(st);
            if (st != DirState::U)
                std::cout << " data=" << cores[i].line_data(addr);
            std::cout << std::endl;
        }
    }

private:
    int num_cores_;
    std::vector<DirEntry> entries_;   // 每个缓存行一个目录条目
};

// ============================================================================
// 演示：使用与 MSI 讲座相同示例的目录协议
// ============================================================================
void run_directory_example() {
    std::cout << "=== CS149 第14讲：基于目录的缓存一致性 ===" << std::endl;
    std::cout << std::endl;
    std::cout << "与 snooping 不同，目录协议仅向实际持有该缓存行的核心" << std::endl;
    std::cout << "发送一致性消息（点对点通信）。不使用广播 —— 因此可扩展到多核心系统。" << std::endl;
    std::cout << std::endl;

    const int NUM_CORES = 3;   // 3 个处理器核心
    const int NUM_LINES = 4;   // 目录跟踪 4 个缓存行

    // 创建核心
    std::vector<Core> cores;
    for (int i = 0; i < NUM_CORES; ++i)
        cores.emplace_back(i);

    Directory dir(NUM_CORES, NUM_LINES);
    int memory = 0;    // 主存中的数据
    int addr = 0;      // 地址 X = 0

    std::cout << "初始状态:" << std::endl;
    dir.print_state(addr, cores);
    std::cout << std::endl;

    // 辅助函数：执行读操作
    auto do_read = [&](int core_id, bool verbose) {
        std::cout << "P" << core_id << " 读取 X" << std::endl;
        bool need_dir = cores[core_id].read(addr);
        if (need_dir) {
            dir.handle_read(core_id, addr, memory, cores, verbose);
        } else {
            std::cout << "  (缓存命中)" << std::endl;
        }
        dir.print_state(addr, cores);
        std::cout << std::endl;
    };

    // 辅助函数：执行写操作
    auto do_write = [&](int core_id, int val, bool verbose) {
        std::cout << "P" << core_id << " 写入 X = " << val << std::endl;
        bool need_dir = cores[core_id].write(addr, val);
        if (need_dir) {
            dir.handle_write(core_id, addr, val, memory, cores, verbose);
        } else {
            std::cout << "  (缓存命中, M 状态)" << std::endl;
        }
        dir.print_state(addr, cores);
        std::cout << std::endl;
    };

    // 与 MSI 讲座相同的操作序列
    do_read(0, true);                     // P1 读 X → U→S, 数据来自主存
    do_read(2, true);                     // P3 读 X → 加入 S 共享者集合
    do_write(2, 42, true);                // P3 写 42 → S→M, 废除 P1 副本（点对点！）
    do_read(0, true);                     // P1 读 X → M→S, 降级 P3 并转发数据
    do_read(0, true);                     // P1 读 X → S 状态命中
    do_write(1, 99, true);                // P2 写 99 → 废除 P1 和 P3, M 状态

    std::cout << "================================================================" << std::endl;
    std::cout << "对比表格: Snooping vs Directory" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::left
              << std::setw(20) << "特性"
              << std::setw(30) << "Snooping（嗅探）"
              << "Directory（目录）" << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    std::cout << std::left
              << std::setw(20) << "消息投递方式"
              << std::setw(30) << "广播给所有核心"
              << "点对点（精准投递）" << std::endl;
    std::cout << std::left
              << std::setw(20) << "可扩展性"
              << std::setw(30) << "受限（总线带宽瓶颈）"
              << "扩展性好（O(1)~O(N)消息）" << std::endl;
    std::cout << std::left
              << std::setw(20) << "总线要求"
              << std::setw(30) << "需要顺序广播总线"
              << "任意互连网络均可" << std::endl;
    std::cout << std::left
              << std::setw(20) << "存储开销"
              << std::setw(30) << "无（目录内建在总线逻辑中）"
              << "每行一个 P 位向量" << std::endl;
    std::cout << std::left
              << std::setw(20) << "串行化点"
              << std::setw(30) << "总线"
              << "目录" << std::endl;
    std::cout << std::left
              << std::setw(20) << "典型应用"
              << std::setw(30) << "早期 SMP 系统"
              << "Intel Core i7 (L3 目录)" << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    run_directory_example();
    return 0;
}
