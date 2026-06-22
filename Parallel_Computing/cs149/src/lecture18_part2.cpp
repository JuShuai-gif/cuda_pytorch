/*
 * lecture18_part2.cpp - STM 编译器优化：屏障分解（Barrier Decomposition）
 * Stanford CS149, Fall 2025 - 第18讲
 *
 * 演示讲座中介绍的 STM 编译器优化技术：
 *
 * 问题背景：单体式（Monolithic）STM 屏障（如 tmTxnBegin/tmTxnCommit）将冗余的
 *   日志记录和锁操作隐藏在编译器无法观察到的黑盒中，导致编译器无法进行优化。
 *
 * 优化方案：将单体式屏障分解为细粒度操作：
 *   - txnOpenForWrite(obj)：对对象获取写锁（仅一次）
 *   - txnLogObjectInt(&field, obj)：保存撤销日志（undo-log）条目
 *   - txnOpenForRead(obj)：将对象注册到读集合中（仅一次）
 *
 * 通过分解屏障，编译器可以实现以下优化：
 *   1. 消除冗余的 OpenForWrite 调用（打开一次，多次写入）
 *   2. 消除冗余的 OpenForRead 调用（打开一次，多次读取）
 *   3. 将屏障调用提升（hoist）到循环外部
 *   4. 合并同一对象上的连续撤销日志条目
 *
 * 优化结果：相比串行执行的开销降至 <40%，相比基于锁的方案开销降至 <30%。
 *
 * 编译命令: g++ -std=c++17 -pthread lecture18_part2.cpp -o lecture18_part2
 * 运行命令: ./lecture18_part2
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
// 优化后的 STM 系统 - 使用分解屏障（Decomposed Barriers）
// ============================================================
// 核心思想：不再使用单体式的屏障调用，而是将其分解为独立的细粒度操作：
//   openForWrite → logField → writeField → commit
//
// 这样编译器可以：
//   - 识别出对同一对象的重复 openForWrite/openForRead 调用并消除之
//   - 将屏障调用提升到循环外
//   - 合并连续的撤销日志条目

class OptimizedSTM {
public:
    // ---- 分解后的 STM 屏障操作 ----

    // 对对象获取写锁（打开对象用于写入）
    // 关键优化点：同一个事务中，对同一对象只需调用一次 openForWrite
    // 编译器可以消除后续对同一对象的重复 openForWrite 调用
    bool openForWrite(int txn_id, int obj_id) {
        // 如果已经对该对象打开了写权限，则跳过（冗余调用消除）
        if (write_opened_[txn_id].count(obj_id)) {
            return true; // 已经打开过了，直接返回成功
        }

        uint64_t& rec = records_[obj_id];
        // 检查该对象是否已被其他事务锁定
        if (is_locked_by_other(rec, txn_id)) {
            std::cout << "  [OpenForWrite] 事务 " << txn_id << " 对象[" << obj_id
                      << "] 已被其他事务锁定。冲突！" << std::endl;
            return false;
        }

        // 保存原始版本号，用于回滚时恢复
        if (!write_opened_[txn_id].count(obj_id)) {
            original_versions_[txn_id][obj_id] = get_obj_version(rec);
        }

        // 获取写锁：将记录标记为已锁定状态
        rec = make_locked(txn_id);
        write_opened_[txn_id].insert(obj_id);
        return true;
    }

    // 记录字段的撤销日志（在修改字段之前保存其旧值）
    // 这是实现原子性回滚的关键：如果事务失败，可以通过撤销日志恢复原值
    void logField(int txn_id, int obj_id, const std::string& field_name, int old_value) {
        undo_logs_[txn_id].push_back({obj_id, field_name, old_value});
    }

    // 写入字段值（此处模拟直接写入内存的 eager 模式）
    void writeField(int obj_id, int value) {
        memory_[obj_id] = value;
    }

    // 对对象获取读锁（打开对象用于读取）
    // 关键优化点：同一事务中，对同一对象只需调用一次 openForRead
    // 编译器可以消除后续对同一对象的重复 openForRead 调用
    bool openForRead(int txn_id, int obj_id) {
        // 如果已经对该对象打开了读权限，则跳过（冗余调用消除）
        if (read_opened_[txn_id].count(obj_id)) {
            return true;
        }

        uint64_t rec = records_[obj_id];
        // 检查该对象是否被其他事务以写模式锁定
        if (is_locked_by_other(rec, txn_id)) {
            return false; // 发生冲突
        }

        read_opened_[txn_id].insert(obj_id);
        return true;
    }

    // 读取字段的当前值
    int readField(int obj_id) {
        auto it = memory_.find(obj_id);
        return (it != memory_.end()) ? it->second : 0;
    }

    // 提交事务：验证读集合，释放所有锁
    // 验证读集合是保证可串行化（serializability）的关键步骤
    bool commit(int txn_id) {
        // 验证读集合：确保事务执行期间读取的所有对象都没有被其他事务修改
        for (int obj_id : read_opened_[txn_id]) {
            uint64_t rec = records_[obj_id];
            if (is_locked_by_other(rec, txn_id)) {
                std::cout << "  [提交] 事务 " << txn_id << " 失败：读集合中对象["
                          << obj_id << "] 发生冲突" << std::endl;
                rollback(txn_id);
                return false;
            }
        }

        // 释放写锁，并分配新版本号
        // 使用原子递增保证版本号的全局唯一性
        uint64_t new_version = global_version_.fetch_add(1) + 1;
        for (int obj_id : write_opened_[txn_id]) {
            records_[obj_id] = make_unlocked(new_version);
        }

        std::cout << "  [提交] 事务 " << txn_id << " 已提交。新版本号 = "
                  << new_version << std::endl;

        // 清理事务状态：清空读集合、写集合、撤销日志和原始版本记录
        read_opened_[txn_id].clear();
        write_opened_[txn_id].clear();
        undo_logs_[txn_id].clear();
        original_versions_[txn_id].clear();
        return true;
    }

    // 回滚事务：反向重放撤销日志，恢复所有被修改字段的原始值
    // 通过撤销日志（undo log）的逆序遍历，将内存状态恢复到事务开始前的状态
    void rollback(int txn_id) {
        auto& log = undo_logs_[txn_id];
        // 反向遍历撤销日志（从最新到最早），逐条恢复旧值
        for (auto it = log.rbegin(); it != log.rend(); ++it) {
            memory_[it->obj_id] = it->old_value;
        }
        // 释放写锁，恢复原始版本号
        for (int obj_id : write_opened_[txn_id]) {
            records_[obj_id] = make_unlocked(original_versions_[txn_id][obj_id]);
        }
        // 清理事务状态
        read_opened_[txn_id].clear();
        write_opened_[txn_id].clear();
        undo_logs_[txn_id].clear();
        original_versions_[txn_id].clear();
    }

    // 初始化对象（设置初始值和版本记录）
    void init_object(int obj_id, int value) {
        memory_[obj_id] = value;
        records_[obj_id] = make_unlocked(0);
    }

    // 获取对象的当前值（只读访问）
    int get_value(int obj_id) const {
        auto it = memory_.find(obj_id);
        return (it != memory_.end()) ? it->second : 0;
    }

private:
    // 撤销日志条目结构：记录被修改的对象 ID、字段名和旧值
    struct UndoEntry {
        int obj_id;
        std::string field_name;
        int old_value;
    };

    // 检查记录是否被非当前事务的其他事务锁定
    // 记录格式：最低位为锁定位（0=已锁定，1=未锁定），其余位为拥有者 ID 或版本号
    bool is_locked_by_other(uint64_t rec, int txn_id) {
        if ((rec & 1ULL) == 0) { // 最低位为0表示已锁定
            uint64_t owner = rec >> 1;  // 提取拥有者 ID
            return owner != static_cast<uint64_t>(txn_id + 1);
        }
        return false;  // 未锁定
    }

    // 从记录中提取对象版本号
    uint64_t get_obj_version(uint64_t rec) {
        return rec >> 1;  // 版本号存储在除最低位外的所有位中
    }

    // 构造锁定状态的记录：最低位=0，其余位=拥有者ID
    static uint64_t make_locked(int owner_id) {
        return (static_cast<uint64_t>(owner_id + 1) << 1);
    }

    // 构造未锁定状态的记录：最低位=1，其余位=版本号
    static uint64_t make_unlocked(uint64_t version) {
        return (version << 1) | 1ULL;
    }

    // ---- 模拟内存和元数据 ----
    std::unordered_map<int, int> memory_;                    // 对象数据存储
    std::unordered_map<int, uint64_t> records_;              // 对象锁/版本记录
    std::atomic<uint64_t> global_version_{0};                // 全局版本号计数器

    // ---- 每个事务的元数据 ----
    std::unordered_map<int, std::unordered_set<int>> read_opened_;     // 读集合（已打开读的对象）
    std::unordered_map<int, std::unordered_set<int>> write_opened_;    // 写集合（已打开写的对象）
    std::unordered_map<int, std::vector<UndoEntry>> undo_logs_;        // 撤销日志
    std::unordered_map<int, std::unordered_map<int, uint64_t>> original_versions_; // 原始版本号（用于回滚）
};

// ============================================================
// 优化前版本：单体式屏障（Monolithic Barrier）STM
// ============================================================
// 模拟未优化的 STM 插桩方式：每次内存访问都触发完整的屏障调用，
// 编译器无法观察屏障内部逻辑，因此无法消除冗余操作。

class MonolithicSTM {
public:
    bool txnBegin(int txn_id) {
        active_txns_[txn_id] = true;
        return true;
    }

    void txnWrite(int txn_id, int obj_id, int& field, int value) {
        // 单体式调用：将 openForWrite + logField + write 封装在一个调用中
        // 编译器无法看到内部结构，因此无法消除冗余
        barrier_count_++;
        field = value;
    }

    int txnRead(int txn_id, int obj_id, int field) {
        barrier_count_++;
        return field;
    }

    bool txnCommit(int txn_id) {
        barrier_count_++;
        active_txns_[txn_id] = false;
        return true;
    }

    int get_barrier_count() const { return barrier_count_; }

private:
    std::unordered_map<int, bool> active_txns_;
    int barrier_count_ = 0;
};

// ============================================================
// 演示：屏障分解优化（Barrier Decomposition）
// ============================================================
// 本节通过对比展示单体式屏障和分解后屏障的差异：
// - 单体式：每次字段访问都产生一次屏障调用，共 6 次
// - 分解后：编译器可以消除冗余的 open 调用，约减少至 5 次

void demo_barrier_decomposition() {
    std::cout << "=== STM 屏障分解优化 ===" << std::endl;
    std::cout << std::endl;

    // 以下是需要添加事务保护的示例代码：
    // atomic {
    //     a.x = t1;
    //     a.y = t2;
    //     if (a.z == 0) {
    //         a.x = 0;
    //         a.z = t3;
    //     }
    // }
    //
    // 对象布局：obj_a.x=1, obj_a.y=2, obj_a.z=3（对象ID）
    const int OBJ_A_X = 1;
    const int OBJ_A_Y = 2;
    const int OBJ_A_Z = 3;

    // ======== 优化前：单体式屏障（未优化插桩） ========
    std::cout << "--- 优化前：单体式屏障（未优化插桩） ---" << std::endl;
    {
        MonolithicSTM stm;
        int t1 = 10, t2 = 20, t3 = 30;
        int a_x = 0, a_y = 0, a_z = 0;

        stm.txnBegin(0);

        // 每次内存访问都被单体式屏障调用包裹，编译器无法优化
        stm.txnWrite(0, OBJ_A_X, a_x, t1);  // 屏障调用 #1
        stm.txnWrite(0, OBJ_A_Y, a_y, t2);  // 屏障调用 #2
        if (stm.txnRead(0, OBJ_A_Z, a_z) == 0) {  // 屏障调用 #3
            stm.txnWrite(0, OBJ_A_X, a_x, 0);  // 屏障调用 #4
            stm.txnWrite(0, OBJ_A_Z, a_z, t3);  // 屏障调用 #5
        }

        stm.txnCommit(0);  // 屏障调用 #6
        std::cout << "屏障调用总数: " << stm.get_barrier_count()
                  << "（单体式：每次访问 = 一次屏障调用）" << std::endl;
    }

    // ======== 优化后：分解屏障（编译器优化后） ========
    std::cout << std::endl;
    std::cout << "--- 优化后：分解屏障（编译器优化后） ---" << std::endl;
    {
        OptimizedSTM stm;
        stm.init_object(OBJ_A_X, 0);
        stm.init_object(OBJ_A_Y, 0);
        stm.init_object(OBJ_A_Z, 0);

        int txn_id = 0;
        int t1 = 10, t2 = 20, t3 = 30;

        // 编译器优化：将对同一对象的 openForWrite 提升到多次写入之前
        // obj_a 只需打开一次，而不是每个字段都打开一次
        bool ok = stm.openForWrite(txn_id, 1); // obj_a 基础对象 - 只打开一次！
        assert(ok);
        std::cout << "OpenForWrite(obj_a)：对整个对象只调用一次" << std::endl;

        // 记录并在缓冲区中写入 a.x = t1
        stm.logField(txn_id, OBJ_A_X, "x", stm.get_value(OBJ_A_X));
        stm.writeField(OBJ_A_X, t1);

        // 写入 a.y = t2（无需重新打开 - 编译器消除了冗余的 open 调用）
        stm.logField(txn_id, OBJ_A_Y, "y", stm.get_value(OBJ_A_Y));
        stm.writeField(OBJ_A_Y, t2);

        // 读取 a.z
        bool rd_ok = stm.openForRead(txn_id, OBJ_A_Z);
        assert(rd_ok);
        int a_z = stm.readField(OBJ_A_Z);

        std::cout << "读取 a.z = " << a_z << "（OpenForRead 只调用一次）" << std::endl;

        if (a_z == 0) {
            // 写入 a.x = 0（已打开写权限 - 无需重复 open！）
            stm.logField(txn_id, OBJ_A_X, "x", t1);
            stm.writeField(OBJ_A_X, 0);

            // 写入 a.z = t3（需要先打开写权限，但编译器可以与之前的写打开合并）
            stm.openForWrite(txn_id, OBJ_A_Z);
            stm.logField(txn_id, OBJ_A_Z, "z", a_z);
            stm.writeField(OBJ_A_Z, t3);
        }

        stm.commit(txn_id);

        std::cout << std::endl;
        std::cout << "结果: a.x=" << stm.get_value(OBJ_A_X)
                  << " a.y=" << stm.get_value(OBJ_A_Y)
                  << " a.z=" << stm.get_value(OBJ_A_Z) << std::endl;

        int barrier_calls = 1 + 2 + 1 + (a_z == 0 ? 1 : 0) + 1;
        std::cout << "分解屏障调用总数: ~" << barrier_calls
                  << "（对比单体式的 6 次调用）" << std::endl;
    }
}

// ============================================================
// 优化演示：消除冗余的 OpenForRead/OpenForWrite 调用
// ============================================================
// 本部分展示编译器如何利用分解屏障消除冗余操作：
// - 对同一对象的多次写入共享一次 openForWrite
// - 对同一对象的多次读取共享一次 openForRead
// 这种优化使每次事务的额外开销从 2-8 倍降至 <40%
void demo_redundant_elimination() {
    std::cout << std::endl;
    std::cout << "=== 冗余 Open 调用的消除 ===" << std::endl;
    std::cout << std::endl;

    OptimizedSTM stm;
    stm.init_object(1, 0);
    stm.init_object(2, 0);

    std::cout << "编译器检测到对同一对象的多次字段访问" << std::endl;
    std::cout << "不需要重复调用 OpenForWrite/OpenForRead：" << std::endl;
    std::cout << std::endl;

    int txn_id = 0;

    // 第一次对对象1调用 OpenForWrite - 实际执行
    bool ok1 = stm.openForWrite(txn_id, 1);
    std::cout << "OpenForWrite(txn, obj1): " << (ok1 ? "已调用（首次）" : "失败") << std::endl;

    // 第二次对对象1调用 OpenForWrite - 编译器可以消除此调用
    bool ok2 = stm.openForWrite(txn_id, 1);
    std::cout << "OpenForWrite(txn, obj1): " << (ok2 ? "空操作（已打开）" : "失败")
              << " ← 编译器消除此调用！" << std::endl;

    stm.logField(txn_id, 1, "field", 0);
    stm.writeField(1, 42);
    stm.logField(txn_id, 1, "field2", 42);
    stm.writeField(1, 99);

    std::cout << "将 obj1 写入为 99（两次写入，仅一次 OpenForWrite）" << std::endl;

    // 第一次对对象2调用 OpenForRead - 实际执行
    bool rd1 = stm.openForRead(txn_id, 2);
    std::cout << "OpenForRead(txn, obj2): " << (rd1 ? "已调用（首次）" : "失败") << std::endl;

    // 第二次对对象2调用 OpenForRead - 编译器可以消除此调用
    bool rd2 = stm.openForRead(txn_id, 2);
    std::cout << "OpenForRead(txn, obj2): " << (rd2 ? "空操作（已打开）" : "失败")
              << " ← 编译器消除此调用！" << std::endl;

    stm.commit(txn_id);

    std::cout << std::endl;
    std::cout << "这就是编译器如何将每个线程的额外开销" << std::endl;
    std::cout << "从 2-8 倍降至相比串行执行不到 40% 的水平。" << std::endl;
}

int main() {
    std::cout << "=== CS149 第18讲: STM 编译器优化 ===" << std::endl;
    std::cout << std::endl;

    demo_barrier_decomposition();
    demo_redundant_elimination();

    std::cout << std::endl;
    std::cout << "STM 优化技术总结：" << std::endl;
    std::cout << "  1. 将单体式屏障（tmWr/tmRd）分解为细粒度原语" << std::endl;
    std::cout << "     （OpenForWrite, LogField, OpenForRead）" << std::endl;
    std::cout << "  2. 消除冗余的 OpenForWrite 调用：每个对象打开一次，" << std::endl;
    std::cout << "     而非每个字段写入打开一次" << std::endl;
    std::cout << "  3. 消除冗余的 OpenForRead 调用：每个对象打开一次，" << std::endl;
    std::cout << "     而非每个字段读取打开一次" << std::endl;
    std::cout << "  4. 将屏障调用提升（hoist）到循环外部" << std::endl;
    std::cout << "  5. 合并同一对象上的连续撤销日志条目" << std::endl;
    std::cout << std::endl;
    std::cout << "优化结果: 相比串行执行的开销 <40%，相比基于锁的方案开销 <30%。" << std::endl;

    return 0;
}
