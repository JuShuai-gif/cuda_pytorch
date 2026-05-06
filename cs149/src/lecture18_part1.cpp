/*
 * lecture18_part1.cpp - 软件事务内存(STM)实现
 * Stanford CS149, Fall 2025 - 讲座 18
 *
 * 基于讲座中介绍的 Intel McRT STM 算法实现软件事务内存系统。
 * 该 STM 系统具有以下特点：
 *   - 急切版本管理（基于撤销日志 Undo-Log）：写入直接更新内存
 *   - 乐观读取 (Optimistic Reads)：读取后验证数据一致性
 *   - 悲观写入 (Pessimistic Writes)：写入前获取锁
 *   - 基于时间戳的每对象版本跟踪 (Timestamp-Based Version Tracking)
 *
 * 核心数据结构：
 *   - 事务描述符 (Transaction Descriptor, 每线程)：读集合、写集合、撤销日志
 *     读集合(Read-Set): 记录事务读取了哪些对象，用于提交时验证数据未过期
 *     写集合(Write-Set): 记录事务修改了哪些对象，用于提交时释放锁
 *     撤销日志(Undo-Log): 记录被覆盖的旧值，用于事务回滚时恢复数据
 *
 *   - 事务记录 (Transaction Record, 每对象)：写入者锁 + 版本号
 *     每个共享对象维护一个 64 位记录，同时编码锁状态和版本号
 *
 *   - 全局时间戳 (Global Timestamp)：在每次写入提交时增加 2
 *     (LSb = 写锁位, MS bits = 版本号)
 *     增加 2 的设计是因为: bit 0 用于表示锁状态(0=锁定, 1=未锁定),
 *     因此版本号需要使用 bit 1 及以上的位，每次递增 2 即版本号增加 1
 *
 * STM 操作：
 *   STM Read  (读取): 直接读取 → 验证(已解锁, 版本号 ≤ 本地时间戳) → 插入读集合
 *   STM Write (写入): 验证 → 获取锁 → 创建撤销日志条目 → 原地写入(eager)
 *   STM Commit(提交): 原子递增全局时间戳(加2) → 验证读集合 → 用新版本号释放锁
 *
 * 编译命令：g++ -std=c++17 -pthread lecture18_part1.cpp -o lecture18_part1
 * 运行命令：./lecture18_part1
 */

#include <iostream>
#include <thread>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>

// ============================================================
// 事务记录（每个对象一个）(Transaction Record, Per-Object)
// ============================================================
// 64 位记录编码方案：
//   最低位 (bit 0): 0 = 已写锁锁定, 1 = 未锁定
//   高位 (bits 63:1):
//     - 未锁定时: 时间戳（上一次提交的版本号）
//     - 锁定时:  类似指针的拥有者事务 ID
//
// 这种编码方案的巧妙之处在于：将锁状态和版本号打包到一个机器字中，
// 允许使用原子操作同时读取/修改锁状态和版本号，避免分离字段的竞态条件。
//
// 这与 Intel McRT STM 将锁状态和版本号打包到单个字的做法类似。
// 使用单个机器字的好处：
//   1. 原子性：64 位读取/写入在大多数平台上已经是原子的
//   2. 高效性：单个 CAS(Compare-And-Swap) 操作即可同时更新锁和版本
//   3. 空间效率：每个对象仅需 8 字节的元数据

using TxVersion = uint64_t;

// 编码/解码事务记录的辅助函数
// 函数命名说明：
//   is_locked:    判断记录是否被锁定（bit 0 == 0 表示锁定）
//   get_version:  提取版本号（右移 1 位去掉锁位）
//   make_locked:  创建锁定状态记录（bit 0 = 0，高位存储拥有者ID）
//   make_unlocked: 创建未锁状态记录（bit 0 = 1，高位存储版本号）

inline bool is_locked(TxVersion rec) {
    return (rec & 1ULL) == 0; // 最低位 = 0 表示锁定
}

inline uint64_t get_version(TxVersion rec) {
    return rec >> 1; // 右移一位去掉最低位（锁位），获得版本号
}

inline TxVersion make_locked(int owner_id) {
    // 锁定状态: 最低位=0, 其余位 = (owner_id + 1) 左移一位
    // +1 是为了避免 owner_id=0 时所有位都为 0（与未初始化状态混淆）
    return (static_cast<uint64_t>(owner_id + 1) << 1);
}

inline TxVersion make_unlocked(uint64_t version) {
    // 未锁定状态: 最低位=1, 其余位 = version 左移一位
    // 最低位为 1 表示"已解锁"，高位存储版本号
    return (version << 1) | 1ULL;
}

// ============================================================
// 软件事务内存系统 (Software Transactional Memory System)
// ============================================================
// 本实现的核心设计决策：
//
// 版本管理策略：急切版本管理 (Eager Versioning)
//   写入直接更新共享内存（不使用写缓冲区），同时用撤销日志保存旧值。
//   优势：读取总是看到最新数据（不需要读缓冲区查找）
//   劣势：回滚时需要从撤销日志恢复旧值
//   对比懒版本管理(Lazy Versioning)：写入暂存缓冲区，提交时才刷入内存
//
// 冲突检测策略：混合策略 (Hybrid)
//   读取：乐观检测（读取时记录，提交时验证）
//   写入：悲观检测（写入前获取锁，检测写-写冲突）
//   这种混合设计在性能和简单性之间取得了平衡

class SoftwareTransactionalMemory {
public:
    SoftwareTransactionalMemory() : global_timestamp_(0) {}

    // ---- STM 读取操作（乐观策略）(STM Read, Optimistic) ----
    //
    // STM 读取的工作流程（对应讲座中的 StmRead 屏障）：
    //   1. 直接读取内存位置（急切版本管理 —— 不需要读缓冲区）
    //   2. 验证读取的数据：检查对象未锁定且版本号 ≤ 本地时间戳
    //   3. 如果版本号过新（版本号 > 本地时间戳），则验证整个读集合的一致性
    //      这意味着自上次快照以来有事务提交，需要确认我们的读数据仍有效
    //   4. 将对象 ID 插入读集合，返回读取值
    //
    // 为什么是乐观读取？
    //   - 读取操作是 STM 中最频繁的操作
    //   - 每次读取都检查锁会带来显著的性能开销
    //   - 乐观策略将验证推迟到对象版本变更时或提交时
    //   - 在低冲突场景下，大多数读取无需额外验证即可完成
    int stm_read(int txn_id, int obj_id) {
        int value = memory_[obj_id]; // 直接读取（急切版本管理）

        // 检查对象是否被其他事务锁定
        TxVersion rec = records_[obj_id];
        if (is_locked(rec) && get_version(rec) != static_cast<uint64_t>(txn_id + 1)) {
            // 对象被其他事务锁定 → 中止当前事务
            // 读取被锁定对象可能导致读到未提交的数据（脏读）
            std::cout << "  [STM 读] 事务 " << txn_id << " 读取对象[" << obj_id
                      << "] 失败: 被事务 " << (get_version(rec) - 1) << " 锁定" << std::endl;
            return -1; // 返回 -1 表示冲突信号
        }

        // 版本号检查：数据不应比我们的本地时间戳更新
        // 如果数据比本地时间戳更新，说明有其他事务在我们不知道的情况下提交了
        // 此时需要验证整个读集合，因为我们之前读取的数据可能已经过时
        uint64_t obj_version = get_version(rec);
        uint64_t local_ts = local_timestamps_[txn_id];
        if (!is_locked(rec) && obj_version > local_ts) {
            // 数据比我们的快照更新 → 需要验证整个读集合
            // 这是乐观策略的代价：有时需要在读取时做额外的验证工作
            if (!validate_read_set(txn_id)) {
                std::cout << "  [STM 读] 事务 " << txn_id << " 读取对象[" << obj_id
                          << "] 失败: 版本号 " << obj_version << " > 本地时间戳 "
                          << local_ts << std::endl;
                return -1; // 验证失败 → 中止事务
            }
            // 验证成功后更新本地时间戳
            local_timestamps_[txn_id] = global_timestamp_.load(std::memory_order_acquire);
        }

        // 插入读集合
        read_sets_[txn_id].insert(obj_id);
        return value;
    }

    // ---- STM 写入操作（悲观策略）(STM Write, Pessimistic) ----
    //
    // STM 写入的工作流程（对应讲座中的 StmWrite 屏障）：
    //   1. 验证数据（检查未锁定，版本号 ≤ 本地时间戳）
    //   2. 获取对象的写锁
    //   3. 创建撤销日志条目（保存旧值 —— 用于回滚时恢复）
    //   4. 原地写入数据（急切版本管理）
    //
    // 为什么是悲观写入？
    //   - 写-写冲突总是需要立即解决（两个事务不能同时写同一对象）
    //   - 提前获取锁可以确保写入者的排他访问权
    //   - 撤销日志的存在使得回滚成为可能，弥补了急切版本管理"不可逆"的缺点
    bool stm_write(int txn_id, int obj_id, int new_value) {
        TxVersion rec = records_[obj_id];

        // 冲突检查1：写-写冲突
        // 对象是否被其他事务锁定？
        // 如果是我们自己锁定的（版本编码中的ID匹配），则允许继续
        if (is_locked(rec) && get_version(rec) != static_cast<uint64_t>(txn_id + 1)) {
            std::cout << "  [STM 写] 事务 " << txn_id << " 写入对象[" << obj_id
                      << "] 失败: 被事务 " << (get_version(rec) - 1) << " 锁定" << std::endl;
            return false; // 写-写冲突
        }

        // 冲突检查2：版本号检查（针对未锁定对象）
        // 如果对象未被锁定但其版本号比我们的本地时间戳新，
        // 说明有事务在我们读写之间提交了对该对象的修改
        if (!is_locked(rec)) {
            uint64_t obj_version = get_version(rec);
            uint64_t local_ts = local_timestamps_[txn_id];
            if (obj_version > local_ts) {
                if (!validate_read_set(txn_id)) {
                    std::cout << "  [STM 写] 事务 " << txn_id << " 写入对象[" << obj_id
                              << "] 失败: 验证失败" << std::endl;
                    return false;
                }
                local_timestamps_[txn_id] = global_timestamp_.load(std::memory_order_acquire);
            }
        }

        // 获取写锁
        uint64_t old_locked_rec = make_locked(txn_id);
        records_[obj_id] = old_locked_rec;

        // 将旧值保存到撤销日志（用于急切版本管理的回滚）
        // 撤销日志按写入顺序记录，回滚时反向执行
        int old_value = memory_[obj_id];
        undo_logs_[txn_id].push_back({obj_id, old_value});

        // 原地写入（急切版本管理）
        // 注意：写入立即对全局可见！这是急切版本管理的特征
        // 其他事务可能看到未提交的数据，但 STM 的锁机制+版本检测保证了隔离性
        memory_[obj_id] = new_value;

        // 插入写集合
        write_sets_[txn_id].insert(obj_id);

        return true;
    }

    // ---- STM 提交操作 (STM Commit) ----
    //
    // STM 提交的工作流程（对应讲座中的提交协议）：
    //   1. 原子递增全局时间戳，步长为 2
    //      - 递增 2 而非 1 是因为 bit 0 用作写锁位
    //      - 实际版本号 = 递增后的值右移一位
    //   2. 如果增量前 (old) 时间戳 > 本地时间戳，表示其他事务已提交，
    //      需要验证读集合（检查自上次验证以来是否有新事务提交）
    //   3. 对于写集合中的每个对象：释放锁，将版本号设为新的全局时间戳
    //
    // 为什么递增 2？
    //   锁位 (bit 0) 占用了一个位，版本号位于 bits 63:1。
    //   如果递增 1，版本号可能变化但锁位也会翻转，导致锁状态混乱。
    //   递增 2 确保：锁位保持为 1（已解锁），版本号增加 1。
    bool stm_commit(int txn_id) {
        // 步骤 1: 递增全局时间戳，步长为 2
        // (LSb 用于写锁，高位是版本号)
        // fetch_add 返回旧值，新值 = 旧值 + 2
        uint64_t old_global_ts = global_timestamp_.fetch_add(2, std::memory_order_acq_rel);
        uint64_t new_global_ts = old_global_ts + 2;

        // 步骤 2: 检查自上次验证以来是否有事务提交
        // 如果 old_global_ts > local_ts，说明在我们的事务开始后
        // 有事务通过递增全局时间戳完成了提交。
        // 此时必须验证读集合，确保我们读到的数据还没过期。
        if (old_global_ts > local_timestamps_[txn_id]) {
            if (!validate_read_set(txn_id)) {
                std::cout << "  [STM 提交] 事务 " << txn_id
                          << " 失败: 提交时读集合验证未通过。" << std::endl;
                // 回滚：从撤销日志中恢复旧值
                rollback(txn_id);
                return false;
            }
        }

        // 步骤 3: 释放写锁并用新版本号标记
        // 将写集合中每个对象的记录更新为"未锁定 + 新版本号"
        // 这使我们的写入对其他事务正式可见
        for (int obj_id : write_sets_[txn_id]) {
            records_[obj_id] = make_unlocked(new_global_ts);
        }

        // 清理事务状态
        read_sets_[txn_id].clear();
        write_sets_[txn_id].clear();
        undo_logs_[txn_id].clear();
        local_timestamps_[txn_id] = new_global_ts;

        std::cout << "  [STM 提交] 事务 " << txn_id << " 已提交。全局时间戳 = "
                  << new_global_ts << std::endl;
        return true;
    }

    // ---- STM 中止 / 回滚 (STM Abort / Rollback) ----
    //
    // 回滚操作的工作流程：
    //   1. 按写入的逆序重放撤销日志（恢复旧值）
    //      - 必须逆序是因为后写入的值可能依赖于先写入的值
    //      - 例如：先写 A，再写 B（其中 B 依赖 A 的新值），
    //        逆序回滚应先恢复 B 再恢复 A
    //   2. 释放所有写锁
    //   3. 清理所有事务状态
    //
    // 回滚发生场景：
    //   - 提交时读集合验证失败
    //   - 读/写操作检测到冲突
    //   - 这是急切版本管理的必要代价 —— 每个写操作都立即生效，
    //     中止时必须用撤销日志恢复
    void rollback(int txn_id) {
        std::cout << "  [STM 回滚] 事务 " << txn_id << " 正在回滚..." << std::endl;

        // 逆向重放撤销日志（急切版本管理）
        // 因为后写入的数据可能依赖于先写入的新值，
        // 所以必须按后进先出(LIFO)顺序恢复
        auto& log = undo_logs_[txn_id];
        for (auto it = log.rbegin(); it != log.rend(); ++it) {
            memory_[it->obj_id] = it->old_value;
        }

        // 释放写锁
        for (int obj_id : write_sets_[txn_id]) {
            uint64_t old_version = original_versions_[txn_id][obj_id];
            records_[obj_id] = make_unlocked(old_version);
        }

        // 清理事务状态
        read_sets_[txn_id].clear();
        write_sets_[txn_id].clear();
        undo_logs_[txn_id].clear();
        original_versions_[txn_id].clear();
    }

    // ---- 读集合验证 (Read-Set Validation) ----
    //
    // 验证读集合中的所有对象是否仍在与本地时间戳一致的版本。
    // 这是乐观读取策略的核心机制。
    //
    // 验证逻辑：
    //   - 如果对象被锁定：检查是否被我们自己锁定（重复读取是允许的）
    //   - 如果对象未锁定：检查其版本号是否 ≤ 本地时间戳
    //     （如果版本号更大，说明数据在我们读取后被修改了）
    //
    // 为什么需要验证？
    //   在乐观策略中，读取时不做冲突检查。但是，其他事务可能在我们读取后
    //   提交了对同一对象的修改。验证确保我们从读集合中读取的所有数据
    //   仍然是"最新"的（相对于我们的事务开始时间）。
    bool validate_read_set(int txn_id) {
        for (int obj_id : read_sets_[txn_id]) {
            TxVersion rec = records_[obj_id];
            if (is_locked(rec)) {
                // 如果被锁定，检查是否被我们自己锁定（重复读取是允许的）
                if (get_version(rec) != static_cast<uint64_t>(txn_id + 1)) {
                    return false; // 被其他事务锁定
                }
            } else {
                uint64_t obj_version = get_version(rec);
                if (obj_version > local_timestamps_[txn_id]) {
                    return false; // 数据自读取后被更新
                }
            }
        }
        return true;
    }

    // ---- 初始化辅助函数 ----
    void init_object(int obj_id, int value) {
        memory_[obj_id] = value;
        records_[obj_id] = make_unlocked(0); // 初始版本号 = 0
    }

    void init_txn(int txn_id) {
        local_timestamps_[txn_id] = global_timestamp_.load(std::memory_order_acquire);
        read_sets_[txn_id].clear();
        write_sets_[txn_id].clear();
        undo_logs_[txn_id].clear();
        original_versions_[txn_id].clear();
    }

    int get_value(int obj_id) const {
        auto it = memory_.find(obj_id);
        return (it != memory_.end()) ? it->second : 0;
    }

    uint64_t get_global_ts() const {
        return global_timestamp_.load();
    }

private:
    struct UndoEntry {
        int obj_id;
        int old_value;
    };

    std::unordered_map<int, int> memory_;                          // 共享内存
    std::unordered_map<int, TxVersion> records_;                   // 每个对象的事务记录
    std::atomic<uint64_t> global_timestamp_;                       // 全局时间戳

    // 每个事务的私有状态
    std::unordered_map<int, uint64_t> local_timestamps_;           // 本地时间戳（事务快照）
    std::unordered_map<int, std::unordered_set<int>> read_sets_;   // 每个事务的读集合
    std::unordered_map<int, std::unordered_set<int>> write_sets_;  // 每个事务的写集合
    std::unordered_map<int, std::vector<UndoEntry>> undo_logs_;    // 每个事务的撤销日志
    std::unordered_map<int, std::unordered_map<int, uint64_t>> original_versions_;
};

// ============================================================
// 演示1: 将对象从 foo 复制到 bar（讲座示例）
// ============================================================
//
// 场景说明：
//   事务 X1: 将对象 foo 的内容复制到对象 bar
//   事务 X2: 读取 bar 的字段
//
// 初始状态: foo=(x=9, y=7), bar=(x=0, y=0)
// 预期结果: X2 看到的 bar 要么是 [0,0]（X1 提交前），要么是 [9,7]（X1 提交后）
//          绝对不会出现 [9,0] 或 [0,7] 这样的混合状态
//
// 为什么不会出现混合状态？
//   STM 保证了事务的原子性：
//   - 如果 X2 在 X1 提交前读取 bar，它读到的是旧值 [0,0]
//   - 如果 X2 在 X1 提交后读取 bar，它读到的是新值 [9,7]
//   - X1 的写锁阻止了 X2 在 X1 写入中途的观察
//   - 读集合验证确保了 X2 在看到部分更新的情况下会中止

void demo_stm_copy_example() {
    std::cout << "=== STM 复制示例（讲座18示例）===" << std::endl;
    std::cout << "  X1 将 foo(x=9,y=7) 复制到 bar(x=0,y=0)" << std::endl;
    std::cout << "  X2 读取 bar 的字段" << std::endl;
    std::cout << "  预期结果: bar = [0,0]（X1 提交前）" << std::endl;
    std::cout << "         或 bar = [9,7]（X1 提交后）" << std::endl;
    std::cout << "         绝不会出现: bar = [9,0] 或 [0,7]" << std::endl;
    std::cout << std::endl;

    SoftwareTransactionalMemory stm;

    // 对象布局: foo = 对象 1 (x), 对象 2 (y); bar = 对象 3 (x), 对象 4 (y)
    stm.init_object(1, 9);  // foo.x
    stm.init_object(2, 7);  // foo.y
    stm.init_object(3, 0);  // bar.x
    stm.init_object(4, 0);  // bar.y

    std::cout << "初始状态: foo=(x=" << stm.get_value(1) << ", y=" << stm.get_value(2) << ")" << std::endl;
    std::cout << "          bar=(x=" << stm.get_value(3) << ", y=" << stm.get_value(4) << ")" << std::endl;
    std::cout << std::endl;

    // X1: 复制 foo → bar
    std::cout << "[事务 X1: 复制 foo → bar]" << std::endl;
    stm.init_txn(1);

    int foo_x = stm.stm_read(1, 1); // 读取 foo.x
    int foo_y = stm.stm_read(1, 2); // 读取 foo.y
    std::cout << "  X1 读取 foo: x=" << foo_x << ", y=" << foo_y << std::endl;

    assert(foo_x == 9 && foo_y == 7);

    bool w1 = stm.stm_write(1, 3, foo_x); // 写入 bar.x = foo.x
    assert(w1);
    bool w2 = stm.stm_write(1, 4, foo_y); // 写入 bar.y = foo.y
    assert(w2);
    std::cout << "  X1 写入 bar: x=" << foo_x << ", y=" << foo_y << std::endl;

    // 提交前，另一个线程读取 bar（应该看到旧值）
    // 注意：由于使用了急切版本管理(eager versioning)，写入已立即对内存可见
    // 但 STM 的锁机制保证了：其他事务在读取时会被锁检查阻挡
    std::cout << std::endl;
    std::cout << "[X1 提交前: 检查 bar 的值]" << std::endl;
    std::cout << "  bar.x=" << stm.get_value(3) << " (急切写入 - 已更新！)" << std::endl;
    std::cout << "  bar.y=" << stm.get_value(4) << " (急切写入 - 已更新！)" << std::endl;

    bool committed = stm.stm_commit(1);
    assert(committed);
    std::cout << std::endl;

    std::cout << "提交后: bar=(x=" << stm.get_value(3) << ", y=" << stm.get_value(4) << ")" << std::endl;
}

// ============================================================
// 演示2: 冲突和回滚
// ============================================================
//
// 场景说明：
//   事务 1 写入对象[10] = 42
//   事务 2 尝试写入对象[10] = 99（冲突！）
//
// 演示的 STM 行为：
//   1. 事务 1 获取对象 10 的写锁，执行急切写入（直接更新内存）
//   2. 事务 2 尝试写入对象 10，检测到写-写冲突 → 失败
//   3. 事务 1 提交，释放写锁，更新版本号
//   4. 事务 2 重试，这次成功获取锁并写入
//
// 关键观察：
//   - 急切版本管理使得写入立即可见，即使是未提交的写入
//   - 悲观写入策略在写入时检测写-写冲突
//   - 锁机制保证了在同一个对象上不会有两个写入者同时活跃

void demo_stm_conflict() {
    std::cout << std::endl;
    std::cout << "=== STM 冲突和回滚 ===" << std::endl;
    std::cout << "  事务 1 写入对象[10] = 42" << std::endl;
    std::cout << "  事务 2 尝试写入对象[10] = 99（冲突！）" << std::endl;
    std::cout << std::endl;

    SoftwareTransactionalMemory stm;
    stm.init_object(10, 0);

    // 事务 1 获取对象 10 的写锁
    std::cout << "[事务 1: 写入对象[10] = 42]" << std::endl;
    stm.init_txn(1);
    bool w1_ok = stm.stm_write(1, 10, 42);
    std::cout << "  事务 1 写入对象[10]=42: " << (w1_ok ? "成功" : "失败") << std::endl;
    std::cout << "  内存中对象[10] = " << stm.get_value(10) << " (急切版本管理: 立即可见)" << std::endl;

    // 事务 2 尝试写入对象 10 → 应检测到锁冲突
    std::cout << std::endl;
    std::cout << "[事务 2: 尝试写入对象[10] = 99]" << std::endl;
    stm.init_txn(2);
    bool w2_ok = stm.stm_write(2, 10, 99);
    std::cout << "  事务 2 写入对象[10]=99: " << (w2_ok ? "成功" : "失败（预期行为）")
              << std::endl;

    // 事务 1 提交
    std::cout << std::endl;
    stm.stm_commit(1);

    // 现在事务 2 可以重试
    std::cout << std::endl;
    std::cout << "[事务 2 在中止后重试]" << std::endl;
    stm.init_txn(2);
    w2_ok = stm.stm_write(2, 10, 99);
    std::cout << "  事务 2 写入对象[10]=99: " << (w2_ok ? "成功" : "失败") << std::endl;
    stm.stm_commit(2);

    std::cout << "最终结果: 对象[10] = " << stm.get_value(10) << std::endl;
}

// ============================================================
// 演示3: 基于时间戳的版本跟踪
// ============================================================
//
// 演示全局时间戳如何用于版本跟踪和读一致性验证。
//
// 时间戳版本跟踪的工作原理：
//   1. 每个提交的事务将全局时间戳递增 2
//   2. 被修改的对象被打上新的全局时间戳作为版本号
//   3. 每个事务在开始时记录当前全局时间戳作为其本地时间戳
//   4. 读取操作验证：对象版本号必须 ≤ 本地时间戳
//   5. 如果对象版本号 > 本地时间戳，说明对象在事务开始后被修改
//
// 这种机制的核心优势：
//   - 无需维护每个对象的完整版本历史
//   - 通过比较版本号即可快速判断数据是否过时
//   - 全局时间戳提供了全序关系，简化了"happens-before"关系的判断

void demo_version_tracking() {
    std::cout << std::endl;
    std::cout << "=== 基于时间戳的版本跟踪 ===" << std::endl;
    std::cout << std::endl;

    SoftwareTransactionalMemory stm;
    stm.init_object(1, 100);

    std::cout << "初始全局时间戳: " << stm.get_global_ts() << std::endl;

    // 事务 1: 读取对象 1, 写入对象 1, 提交
    stm.init_txn(1);
    int v = stm.stm_read(1, 1);
    std::cout << "事务 1 读取对象[1] = " << v << " (本地时间戳=" << stm.get_global_ts() << ")" << std::endl;

    stm.stm_write(1, 1, 200);
    stm.stm_commit(1);
    std::cout << "事务 1 提交后, 全局时间戳 = " << stm.get_global_ts() << std::endl;

    // 事务 2: 读取对象 1 - 版本号检查应该通过（本地时间戳在 init 时已更新）
    stm.init_txn(2);
    v = stm.stm_read(2, 1);
    std::cout << "事务 2 读取对象[1] = " << v << " (应该是 200)" << std::endl;
    assert(v == 200);

    std::cout << "版本跟踪和读验证正常工作。" << std::endl;
}

int main() {
    std::cout << "=== CS149 讲座 18: 软件事务内存 ===" << std::endl;
    std::cout << std::endl;

    std::cout << "基于 Intel McRT STM 的实现:" << std::endl;
    std::cout << "  - 急切版本管理（撤销日志）" << std::endl;
    std::cout << "  - 乐观读取（读取后验证）" << std::endl;
    std::cout << "  - 悲观写入（写入前获取锁）" << std::endl;
    std::cout << "  - 基于时间戳的版本跟踪" << std::endl;
    std::cout << "  - 全局时间戳步长递增 2（最低位 = 写锁位）" << std::endl;
    std::cout << std::endl;

    demo_stm_copy_example();
    demo_stm_conflict();
    demo_version_tracking();

    std::cout << std::endl;
    std::cout << "总结:" << std::endl;
    std::cout << "  - STM 屏障（StmRead/StmWrite）由编译器自动插入" << std::endl;
    std::cout << "  - 事务描述符跟踪读集合、写集合和撤销日志" << std::endl;
    std::cout << "  - 每个对象的事务记录将锁状态和版本号打包到一个机器字中" << std::endl;
    std::cout << "  - STM 开销: 每线程 2-8 倍性能损失（这也是 HTM 被青睐的原因）" << std::endl;

    return 0;
}
