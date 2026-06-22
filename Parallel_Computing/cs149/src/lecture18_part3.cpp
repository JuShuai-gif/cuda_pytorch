/*
 * lecture18_part3.cpp - 硬件事务内存（HTM: Hardware Transactional Memory）模拟
 * Stanford CS149, Fall 2025 - 第18讲
 *
 * 模拟基于缓存的硬件事务内存系统：
 *
 * 讲座中涉及的 HTM 核心概念：
 *   1. 缓存中的数据版本管理（data versioning）：
 *      在缓存行中维护写缓冲区（write buffer）或撤销日志（undo log）
 *   2. 缓存行上的 R/W 位（Read/Write bits）：
 *      用于追踪每个缓存行是否属于当前事务的读集合（read-set）
 *      或写集合（write-set）
 *   3. 通过缓存一致性协议（cache coherence protocol）进行冲突检测：
 *      - BusRd 命中 W 位缓存行 → 读-写冲突（R-W conflict）
 *        含义：另一个核的事务写了该数据，我试图读取它
 *      - BusRdX 命中 R 位缓存行 → 写-读冲突（W-R conflict）
 *        含义：另一个核的事务读了该数据，我试图写入它
 *      - BusRdX 命中 W 位缓存行 → 写-写冲突（W-W conflict）
 *        含义：两个事务同时试图修改同一数据
 *   4. 两阶段提交（two-phase commit）：
 *      第一阶段：验证（validate）- 检查写集合中所有行的冲突
 *      第二阶段：提交（commit）- 批量清除 R/W 位，将写集合数据写入内存
 *   5. 快速中止（fast abort）：
 *      使写集合失效 → 批量清除 R/W 位 → 恢复寄存器检查点
 *
 * 还演示了类似Intel Haswell RTM（Restricted Transactional Memory）的语义：
 *   - 硬件事务可因多种原因中止（缓存行驱逐、中断、缺页等）
 *   - 必须提供基于锁的回退路径（fallback path）
 *   - 需要设置最大重试次数，超过后转用锁机制
 *
 * 编译命令: g++ -std=c++17 -pthread lecture18_part3.cpp -o lecture18_part3
 * 运行命令: ./lecture18_part3
 */

#include <iostream>
#include <thread>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <atomic>
#include <mutex>
#include <cassert>
#include <random>
#include <chrono>

// ============================================================
// 带 R/W 事务位的模拟缓存行结构
// ============================================================
// 表示一个带有 HTM 元数据的缓存行。
// 在真实硬件中，缓存行包含：MESI 状态 + R 位 + W 位 + 标签（Tag）+ 数据（Data）。
// 此处按"地址"维度模拟这一结构。
//
// MESI 协议状态说明：
//   INVALID  - 缓存行无效，数据不可用
//   SHARED   - 缓存行有效，可能被多个核共享（只读）
//   MODIFIED - 缓存行有效且被修改过，只有当前核持有，需要写回内存

enum class CacheState { INVALID, SHARED, MODIFIED };

struct CacheLine {
    int addr;                // 该缓存行对应的内存地址
    int data;                // 缓存的数据内容
    CacheState mesi;         // MESI 一致性协议状态
    bool r_bit;              // 读集合成员位（事务中的加载操作设置此位）
    bool w_bit;              // 写集合成员位（事务中的存储操作设置此位）
    bool dirty;              // 标记数据是否与内存中的值不同（脏数据标记）

    CacheLine() : addr(-1), data(0), mesi(CacheState::INVALID),
                  r_bit(false), w_bit(false), dirty(false) {}
};

// ============================================================
// 硬件事务内存（HTM）模拟器
// ============================================================
// 模拟一个支持 HTM 的缓存系统。
// 事务通过缓存行上的 R/W 位来追踪读取和写入操作。
// 冲突检测通过模拟缓存一致性协议的查找（BusRd/BusRdX）来触发。
//
// 设计要点：
// - 使用 lazy versioning（惰性版本管理）：写入操作仅更新缓存，不立即写回内存
// - 使用 eager conflict detection（急切冲突检测）：每次访问都检查是否与
//   其他核的事务发生冲突
// - 两阶段提交确保原子性：先验证再批量清除元数据

class HardwareTransactionalMemory {
public:
    static constexpr int CACHE_SIZE = 16; // 为模拟方便使用较小的缓存大小

    HardwareTransactionalMemory() {
        // 初始化全局内存（模拟主存）
        for (int i = 0; i < 256; ++i) {
            main_memory_[i] = 0;
        }
    }

    // ============================================================
    // HTM 核心操作（每个核心视角）
    // ============================================================

    // XBEGIN：开始一个硬件事务
    // 在真实硬件中（如 Intel RTM）：
    //   - 保存寄存器检查点（checkpoint）
    //   - 清除所有缓存行的 R/W 位
    //   - 设置回退地址：当事务失败时跳转到的代码位置
    //
    // 参数 fallback_addr：
    //   回退代码的地址（-1 表示无回退地址）
    //   当事务中止时，程序计数器跳转到此地址继续执行
    bool xbegin(int core_id, int fallback_addr = -1) {
        if (txn_active_[core_id]) {
            std::cerr << "错误: 核心 " << core_id
                      << " 已处于事务中！" << std::endl;
            return false;
        }

        // 保存寄存器状态检查点（模拟真实硬件行为）
        // 初始化事务性缓存状态
        txn_active_[core_id] = true;
        txn_fallback_[core_id] = fallback_addr;

        // 保存寄存器检查点值，用于中止时恢复
        checkpoint_[core_id] = reg_state_[core_id];

        // 批量清除所有缓存行的 R/W 位（事务开始时重置事务状态）
        for (auto& line : caches_[core_id]) {
            line.r_bit = false;
            line.w_bit = false;
        }

        return true;
    }

    // XLOAD：在事务中执行加载操作
    // 操作流程：
    //   1. 在缓存中查找或加载目标地址的缓存行
    //   2. 该行的 R 位被设置为 1（标记为读集合成员）
    //   3. 进行冲突检测：如果另一个核对此行发出了 BusRdX（独占请求），
    //      意味着另一个核试图写入此行，这是一个写-读冲突，会导致当前事务中止
    //
    // 在真实 Intel RTM 中：
    //   如果事务的读集合或写集合中的任何缓存行被驱逐出缓存，
    //   事务会立即中止（容量限制）
    int xload(int core_id, int addr) {
        assert(txn_active_[core_id] && "未处于事务中");

        // 在缓存中查找或加载该地址对应的缓存行
        int cache_idx = find_or_load_line(core_id, addr);

        if (cache_idx < 0) {
            // 缓存未命中 → 在真实 HTM 中，缓存行驱逐可能导致事务中止
            // Intel RTM：读/写集合中任何缓存行的驱逐都会导致中止
            std::cout << "  [核心 " << core_id << "] XLOAD 地址[" << addr
                      << "] → 缓存未命中，中止事务！" << std::endl;
            xabort(core_id);
            return -1;
        }

        auto& line = caches_[core_id][cache_idx];

        // 冲突检查：是否有其他核心在写集合中包含此行？
        // 参数 true 表示这是一个读操作
        if (check_conflict(core_id, addr, true)) {
            std::cout << "  [核心 " << core_id << "] XLOAD 地址[" << addr
                      << "] → 读-写冲突（R-W conflict），中止事务！" << std::endl;
            xabort(core_id);
            return -1;
        }

        // 标记该缓存行为读集合的一部分
        line.r_bit = true;

        // 从缓存中加载数据并返回
        return line.data;
    }

    // XSTORE：在事务中执行存储操作
    // 操作流程：
    //   1. 在缓存中查找或加载目标缓存行
    //   2. 该行的 W 位被设置为 1（标记为写集合成员）
    //   3. 对于 lazy versioning：数据缓存在缓存中，不立即写入主存
    //   4. 对于 eager versioning：写入主存，在单独的缓存行中维护撤销日志
    //
    // 注意：在 lazy versioning 模式下，事务进行期间主存不会看到未提交的修改，
    // 这保证了事务的隔离性（isolation）。只有当事务成功提交后，
    // 修改才会被刷新到主存中。
    bool xstore(int core_id, int addr, int value) {
        assert(txn_active_[core_id] && "未处于事务中");

        // 在缓存中查找或加载该地址对应的缓存行
        int cache_idx = find_or_load_line(core_id, addr);

        if (cache_idx < 0) {
            std::cout << "  [核心 " << core_id << "] XSTORE 地址[" << addr
                      << "] → 缓存未命中，中止事务！" << std::endl;
            xabort(core_id);
            return false;
        }

        auto& line = caches_[core_id][cache_idx];

        // 冲突检查：参数 false 表示这是一个写操作
        if (check_conflict(core_id, addr, false)) {
            std::cout << "  [核心 " << core_id << "] XSTORE 地址[" << addr
                      << "] → 写冲突（W-W 或 W-R conflict），中止事务！" << std::endl;
            xabort(core_id);
            return false;
        }

        // Lazy versioning（惰性版本管理）：
        // 将写入缓冲在缓存中（标记为 dirty，设置 W 位）
        // 主存此时不会被更新 —— 修改仅在事务提交后才对外可见
        line.w_bit = true;
        line.dirty = true;
        line.data = value;
        line.mesi = CacheState::MODIFIED;  // 进入修改状态（独占+脏）

        return true;
    }

    // XCOMMIT：提交事务（两阶段提交）
    // 第一阶段（验证阶段）：请求写集合中所有缓存行的独占访问权限
    //   在真实硬件中：对所有 W 位缓存行发出 RdX（read-exclusive）请求
    //   验证没有其他核的读/写集合与当前事务的写集合有交叠
    //
    // 第二阶段（提交阶段）：批量清除 R/W 位
    //   将写集合中所有脏行刷新到主存
    //   清除所有 R/W 位，写集合行变为普通的有效脏缓存行
    //   这使得事务的修改对系统中所有核可见
    bool xcommit(int core_id) {
        assert(txn_active_[core_id] && "未处于事务中");

        std::cout << "  [核心 " << core_id << "] XCOMMIT 开始..." << std::endl;

        // 第一阶段：验证（validate）- 检查写集合的冲突情况
        // 在真实硬件中：对每个 W 位缓存行发出 BusRdX 以获取独占访问权
        for (auto& line : caches_[core_id]) {
            if (line.w_bit) {
                // 检查是否有其他核正在读取或写入此行
                for (int other = 0; other < MAX_CORES; ++other) {
                    if (other == core_id || !txn_active_[other]) continue;
                    for (auto& other_line : caches_[other]) {
                        if (other_line.addr == line.addr &&
                            (other_line.r_bit || other_line.w_bit)) {
                            std::cout << "  [核心 " << core_id << "] XCOMMIT 失败: "
                                      << "地址[" << line.addr << "] 与核心"
                                      << other << " 冲突" << std::endl;
                            xabort(core_id);
                            return false;
                        }
                    }
                }
            }
        }

        // 第二阶段：提交（commit）- 将脏行刷新到主存
        // 将写集合中的所有脏缓存行写回主存，使修改对外可见
        for (auto& line : caches_[core_id]) {
            if (line.w_bit && line.dirty) {
                main_memory_[line.addr] = line.data;
                std::cout << "  [核心 " << core_id << "] 提交: 将地址["
                          << line.addr << "] = " << line.data << " 刷新到主存" << std::endl;
            }
        }

        // 批量清除所有缓存行的 R/W 位（gang-clear）
        // 写集合行变为普通的有效脏缓存行（非事务性）
        for (auto& line : caches_[core_id]) {
            line.r_bit = false;
            line.w_bit = false;
        }

        txn_active_[core_id] = false;
        std::cout << "  [核心 " << core_id << "] XCOMMIT 成功！" << std::endl;
        return true;
    }

    // XABORT：中止事务（快速中止 fast abort）
    // 中止步骤（三步走）：
    //   1. 使写集合失效：丢弃 W 位缓存行中的所有脏数据
    //      这样未提交的修改就完全消失了
    //   2. 批量清除 R/W 位：重置所有缓存行的事务元数据
    //   3. 恢复寄存器检查点：将寄存器状态恢复到事务开始前的值
    //
    // 之后检查是否需要回退到基于锁的代码路径
    // （Intel RTM 的关键特性：硬件无法保证事务一定成功）
    void xabort(int core_id) {
        assert(txn_active_[core_id] && "未处于事务中");

        std::cout << "  [核心 " << core_id << "] XABORT! 正在回滚..." << std::endl;

        // 第一步：使写集合的缓存行失效（丢弃未提交的写入）
        for (auto& line : caches_[core_id]) {
            if (line.w_bit) {
                line.mesi = CacheState::INVALID;  // 标记为无效
                line.dirty = false;                // 清除脏标记
                line.data = 0;                     // 丢弃数据
            }
        }

        // 第二步：批量清除所有缓存行的 R/W 位
        for (auto& line : caches_[core_id]) {
            line.r_bit = false;
            line.w_bit = false;
        }

        // 第三步：恢复寄存器检查点（恢复到事务开始前的状态）
        reg_state_[core_id] = checkpoint_[core_id];

        txn_active_[core_id] = false;

        // 检查是否应该回退到基于锁的代码路径
        if (txn_fallback_[core_id] >= 0) {
            std::cout << "  [核心 " << core_id << "] 回退到基于锁的代码路径。"
                      << std::endl;
        }
    }

    // ============================================================
    // 缓存一致性协议模拟
    // ============================================================
    // 模拟一致性协议的监听（snooping）机制：
    //
    // BusRd（共享请求）命中 W 位行 → 读-写冲突
    //   一个核试图读取已被另一个核写入的数据，会导致不一致
    //   （读取者可能看到部分更新的状态）
    //
    // BusRdX（独占请求）命中 R 位行 → 写-读冲突
    //   一个核试图写入被另一个核读取的数据，读集合验证将失败
    //   （已读取的值可能不再有效）
    //
    // BusRdX（独占请求）命中 W 位行 → 写-写冲突
    //   两个核同时试图修改同一数据，只有一个能成功
    //   （需要序列化这两个写操作）
    bool check_conflict(int core_id, int addr, bool is_read) {
        // 模拟一致性总线的监听（snoop）操作

        for (int other = 0; other < MAX_CORES; ++other) {
            if (other == core_id || !txn_active_[other]) continue;

            for (auto& other_line : caches_[other]) {
                if (other_line.addr != addr) continue;

                if (is_read) {
                    // 当前操作为读取（READ）：如果另一个核设置了 W 位，则冲突
                    // 对应：BusRd 命中 W 位行 → 读-写冲突
                    if (other_line.w_bit) {
                        return true;
                    }
                } else {
                    // 当前操作为写入（WRITE，对应 BusRdX）：
                    // 如果另一个核设置了 R 位或 W 位，则冲突
                    // 对应：BusRdX 命中 R/W 位行 → 写-读 或 写-写冲突
                    if (other_line.r_bit || other_line.w_bit) {
                        return true;
                    }
                }
            }
        }
        return false; // 无冲突
    }

    // ============================================================
    // 回退路径（fallback path）：基于锁的执行方式
    // ============================================================
    // Intel RTM 要求必须提供基于锁的回退路径，因为硬件事务可能
    // 因以下原因反复中止：
    // - 缓存行驱逐（事务工作集超过 L1 缓存容量）
    // - 中断、异常、缺页错误
    // - 上下文切换
    // - 系统管理中断（SMI）
    //
    // 回退路径保证程序的向前推进（forward progress guarantee）：
    // 即使用互斥锁（mutex）保护的串行执行版本
    void fallback_transfer(int core_id, int from_addr, int to_addr, int amount) {
        std::lock_guard<std::mutex> guard(fallback_lock_);

        // 安全的基于锁的执行：使用互斥锁保证原子性
        main_memory_[from_addr] -= amount;
        main_memory_[to_addr] += amount;

        std::cout << "  [核心 " << core_id << "] 回退路径: 已将 " << amount
                  << " 从地址[" << from_addr << "] 转移到地址[" << to_addr << "]" << std::endl;
    }

    // ============================================================
    // 类似 Intel RTM 的乐观事务与回退机制
    // ============================================================
    // 模拟 Intel RTM 的使用模式：
    //   1. 乐观尝试（optimistic）：先用硬件事务执行
    //   2. 重试（retry）：如果事务中止，重试若干次
    //   3. 回退（fallback）：重试耗尽后，使用锁保护的代码路径
    //
    // 这种混合策略（HTM + Lock）在最佳情况下获得 HTM 的高并发性能，
    // 在最坏情况下保证向前推进（通过锁机制）
    void rtm_transfer(int core_id, int from_addr, int to_addr, int amount) {
        int max_retries = 3;  // 最大重试次数（Intel 建议 3-5 次）
        for (int attempt = 0; attempt < max_retries; ++attempt) {
            if (xbegin(core_id)) {
                // 乐观路径：尝试使用硬件事务执行

                int from_val = xload(core_id, from_addr);
                if (from_val < 0) continue; // 事务已中止，重试

                int to_val = xload(core_id, to_addr);
                if (to_val < 0) continue; // 事务已中止，重试

                bool ok1 = xstore(core_id, from_addr, from_val - amount);
                if (!ok1) continue;

                bool ok2 = xstore(core_id, to_addr, to_val + amount);
                if (!ok2) continue;

                if (xcommit(core_id)) {
                    return; // 硬件事务成功提交！
                }
                // 提交失败，重试或回退到锁路径
            }
        }

        // 回退路径：所有 HTM 尝试失败后，使用锁保护的执行方式
        fallback_transfer(core_id, from_addr, to_addr, amount);
    }

    // ============================================================
    // 工具函数
    // ============================================================
    int read_memory(int addr) const { return main_memory_[addr]; }
    void write_memory(int addr, int value) { main_memory_[addr] = value; }

    static constexpr int MAX_CORES = 4;  // 模拟的处理器核心数

private:
    // 在缓存中查找或加载指定地址的缓存行
    // 返回缓存行索引，如果所有行都在事务中且无空位则返回 -1
    int find_or_load_line(int core_id, int addr) {
        // 检查地址是否已在缓存中
        for (int i = 0; i < CACHE_SIZE; ++i) {
            if (caches_[core_id][i].addr == addr &&
                caches_[core_id][i].mesi != CacheState::INVALID) {
                return i;
            }
        }

        // 查找空缓存行（或逐出一个非事务行）
        for (int i = 0; i < CACHE_SIZE; ++i) {
            if (caches_[core_id][i].mesi == CacheState::INVALID) {
                caches_[core_id][i].addr = addr;
                caches_[core_id][i].data = main_memory_[addr];
                caches_[core_id][i].mesi = CacheState::SHARED;
                caches_[core_id][i].dirty = false;
                caches_[core_id][i].r_bit = false;
                caches_[core_id][i].w_bit = false;
                return i;
            }
        }

        // 类似于 LRU 驱逐策略：驱逐第一个不在事务中的缓存行
        for (int i = 0; i < CACHE_SIZE; ++i) {
            if (!caches_[core_id][i].r_bit && !caches_[core_id][i].w_bit) {
                // 如果被驱逐的行是脏行，需要写回主存
                if (caches_[core_id][i].dirty) {
                    main_memory_[caches_[core_id][i].addr] = caches_[core_id][i].data;
                }
                caches_[core_id][i].addr = addr;
                caches_[core_id][i].data = main_memory_[addr];
                caches_[core_id][i].mesi = CacheState::SHARED;
                caches_[core_id][i].dirty = false;
                return i;
            }
        }

        // 所有缓存行都在事务读/写集合中 → 驱逐会导致事务中止！
        // 这是 HTM 的关键限制：事务工作集不能超过缓存容量
        return -1;
    }

    // ---- 模拟组件 ----
    int main_memory_[256];                          // 主存（256 个地址）
    CacheLine caches_[MAX_CORES][CACHE_SIZE];       // 每个核的缓存（MAX_CORES × CACHE_SIZE）
    bool txn_active_[MAX_CORES] = {false};           // 各核事务是否活跃
    int txn_fallback_[MAX_CORES] = {-1, -1, -1, -1}; // 各核的回退地址
    int reg_state_[MAX_CORES] = {0};                // 寄存器状态（模拟）
    int checkpoint_[MAX_CORES] = {0};               // 寄存器检查点（用于中止恢复）
    std::mutex fallback_lock_;                      // 回退路径的互斥锁
};

// ============================================================
// 演示部分
// ============================================================

// 演示1：单核事务执行流程
// 展示 HTM 的基本指令序列：XBEGIN → XLOAD → XSTORE → XCOMMIT
// 重点展示 lazy versioning 的行为：事务进行期间，主存不会看到未提交的修改
void demo_htm_instructions() {
    std::cout << "=== HTM: 单核事务执行流程 ===" << std::endl;
    std::cout << std::endl;

    HardwareTransactionalMemory htm;
    htm.write_memory(100, 10); // 初始化 addr[100] = 10
    htm.write_memory(200, 20); // 初始化 addr[200] = 20

    std::cout << "初始内存状态: addr[100]=" << htm.read_memory(100)
              << ", addr[200]=" << htm.read_memory(200) << std::endl;
    std::cout << std::endl;

    // 执行事务：XBEGIN → XLOAD → XSTORE → XCOMMIT
    std::cout << "[核心0上的事务]" << std::endl;

    // XBEGIN: 开始事务，保存寄存器检查点，清除 R/W 位
    htm.xbegin(0);

    // XLOAD: 在事务中读取数据，设置 R 位
    int a = htm.xload(0, 100);
    std::cout << "  XLOAD addr[100] = " << a << "（R 位在缓存行 100 上被设置）" << std::endl;

    int b = htm.xload(0, 200);
    std::cout << "  XLOAD addr[200] = " << b << "（R 位在缓存行 200 上被设置）" << std::endl;

    // XSTORE: 在事务中写入数据，设置 W 位
    // 数据被缓冲在缓存中，尚未写入主存（lazy versioning）
    htm.xstore(0, 100, a * 10); // 10 → 100
    std::cout << "  XSTORE addr[100] = " << a * 10
              << "（W 位已设置，数据缓冲在缓存中，尚未写入主存）" << std::endl;

    // 验证 lazy versioning：事务进行期间，主存中的值仍然是旧值
    std::cout << "  事务期间内存状态: addr[100]=" << htm.read_memory(100)
              << "（仍然是旧值 - lazy versioning！）" << std::endl;

    // XCOMMIT: 提交事务，将脏行刷新到主存
    htm.xcommit(0);
    std::cout << "  事务提交后内存状态: addr[100]=" << htm.read_memory(100)
              << "（新值现在对外可见）" << std::endl;
}

// 演示2：跨核心冲突检测
// 展示 HTM 如何通过缓存一致性协议检测事务间的冲突
// 场景：核心0读取地址50，核心1写入地址50 → 写-读冲突（W-R conflict）
void demo_htm_conflict() {
    std::cout << std::endl;
    std::cout << "=== HTM: 跨核心冲突检测 ===" << std::endl;
    std::cout << std::endl;

    HardwareTransactionalMemory htm;
    htm.write_memory(50, 100); // 共享数据 addr[50] = 100

    // 核心0开始事务，读取 addr[50]（设置 R 位）
    htm.xbegin(0);
    int val0 = htm.xload(0, 50);
    std::cout << "核心0 读取 addr[50] = " << val0 << "（R 位已设置）" << std::endl;

    // 核心1开始事务，尝试写入 addr[50]
    // 这会在一致性总线上生成 BusRdX（独占请求）
    // 核心0的缓存检测到 BusRdX 命中了自己带 R 位的缓存行 → 写-读冲突（W-R conflict）
    htm.xbegin(1);
    std::cout << "核心1 尝试写入 addr[50] = 200..." << std::endl;
    htm.xstore(1, 50, 200); // 核心1的写入操作应该成功

    // 核心0的事务现在处于冲突状态
    // 当核心0再次尝试 XLOAD 时，将检测到冲突并中止
    std::cout << "核心0 再次尝试读取 addr[50]..." << std::endl;
    int val0b = htm.xload(0, 50);
    if (val0b < 0) {
        std::cout << "核心0的事务已被中止（检测到写-读冲突）！" << std::endl;
    }

    // 核心1可以成功提交其事务
    htm.xcommit(1);
    std::cout << "核心1 已提交。addr[50] = " << htm.read_memory(50) << std::endl;
}

// 演示3：Intel RTM 的回退路径
// 展示类似 Intel RTM 的混合策略：
// 乐观尝试硬件事务 + 重试 + 基于锁的回退路径
void demo_rtm_fallback() {
    std::cout << std::endl;
    std::cout << "=== Intel RTM: 回退路径 ===" << std::endl;
    std::cout << std::endl;

    std::cout << "Intel Haswell RTM 提供的指令：" << std::endl;
    std::cout << "  xbegin(fallback_addr) - 开始硬件事务" << std::endl;
    std::cout << "  xend                  - 提交硬件事务" << std::endl;
    std::cout << "  xabort                - 显式中止事务" << std::endl;
    std::cout << std::endl;
    std::cout << "关键限制：不保证向前推进（no forward progress guarantee）。" << std::endl;
    std::cout << "  - 读/写集合中任何缓存行的驱逐 → 事务中止" << std::endl;
    std::cout << "  - 中断、缺页错误、上下文切换 → 事务中止" << std::endl;
    std::cout << "  - 必须提供基于锁的回退路径来保证向前推进" << std::endl;
    std::cout << std::endl;

    HardwareTransactionalMemory htm;
    htm.write_memory(10, 1000);
    htm.write_memory(20, 500);

    // 模拟 RTM 风格：先尝试 HTM，失败后回退到锁机制
    std::cout << "从地址[10]向地址[20]转移 $100：" << std::endl;
    htm.rtm_transfer(0, 10, 20, 100);

    std::cout << "结果: addr[10]=" << htm.read_memory(10)
              << ", addr[20]=" << htm.read_memory(20) << std::endl;
}

int main() {
    std::cout << "=== CS149 第18讲: 硬件事务内存（HTM） ===" << std::endl;
    std::cout << std::endl;

    std::cout << "HTM 架构核心概念：" << std::endl;
    std::cout << "  1. 缓存中的数据版本管理（lazy versioning：通过写缓冲实现）" << std::endl;
    std::cout << "  2. 每个缓存行上的 R/W 位追踪事务读/写集合" << std::endl;
    std::cout << "  3. 通过缓存一致性协议进行冲突检测：" << std::endl;
    std::cout << "     - BusRd 命中 W 位行 → 读-写冲突（R-W conflict）" << std::endl;
    std::cout << "     - BusRdX 命中 R 位行 → 写-读冲突（W-R conflict）" << std::endl;
    std::cout << "     - BusRdX 命中 W 位行 → 写-写冲突（W-W conflict）" << std::endl;
    std::cout << "  4. 两阶段提交：验证（validate）+ 批量清除 R/W 位（gang-clear）" << std::endl;
    std::cout << "  5. 快速中止：使写集合失效 + 恢复寄存器检查点" << std::endl;
    std::cout << std::endl;

    demo_htm_instructions();
    demo_htm_conflict();
    demo_rtm_fallback();

    std::cout << std::endl;
    std::cout << "HTM 性能指标（来自讲座数据）：" << std::endl;
    std::cout << "  - 相比 STM 性能提升 2-7 倍" << std::endl;
    std::cout << "  - 单线程性能相比串行执行在 10% 以内" << std::endl;
    std::cout << "  - 在 Vacation 等基准测试中达到接近理想的加速比" << std::endl;
    std::cout << std::endl;
    std::cout << "HTM 局限性：" << std::endl;
    std::cout << "  - L1 缓存大小限制了事务工作集（working set）的上限" << std::endl;
    std::cout << "  - 因中断、缺页等原因导致的虚假中止（spurious abort）" << std::endl;
    std::cout << "  - 必须提供基于锁的回退路径以保障向前推进" << std::endl;
    std::cout << "  - Intel 优化指南（第12章）：提供了提高事务" << std::endl;
    std::cout << "    成功概率的指导原则" << std::endl;

    return 0;
}
