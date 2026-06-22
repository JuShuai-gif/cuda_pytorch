/*
 * lecture17_part3.cpp - 事务内存(TM)冲突检测：悲观检测 vs 乐观检测
 * Stanford CS149, Fall 2025 - 讲座 17
 *
 * 本程序演示了事务内存(TM, Transactional Memory)的两种冲突检测策略：
 *
 *   1. 悲观（急切）检测 (Pessimistic / Eager Detection)：
 *      - 在每次加载(load)和存储(store)操作时都检查冲突
 *      - 检测到冲突时：立即暂停(stall)或中止(abort)当前事务
 *      - 优点：尽早检测冲突，避免浪费计算资源
 *      - 缺点：每次操作都有额外开销，不保证前向进度(forward progress)
 *      - 关键思想：对冲突采取"防御性"策略 —— 假设冲突会发生，提前检测
 *
 *   2. 乐观（懒惰）检测 (Optimistic / Lazy Detection)：
 *      - 仅在事务提交(commit)时才检查冲突
 *      - 检测到冲突时：中止非提交方的事务
 *      - 优点：每操作无额外开销，保证前向进度
 *      - 缺点：可能在注定失败的事务上浪费计算资源
 *      - 关键思想：对冲突采取"信任性"策略 —— 假设不会冲突，最后再验证
 *
 * 两种策略的核心权衡 (Trade-off)：
 *   - 悲观检测：检测延迟低（即时发现冲突），但吞吐量可能受限于锁竞争
 *   - 乐观检测：无每操作开销，吞吐量高，但冲突场景下可能浪费大量工作
 *   - 选择哪种策略取决于应用特性：高冲突场景适合悲观检测，低冲突场景适合乐观检测
 *
 * 编译命令：g++ -std=c++17 -pthread lecture17_part3.cpp -o lecture17_part3
 * 运行命令：./lecture17_part3
 */

#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <cassert>
#include <chrono>
#include <unordered_map>
#include <unordered_set>

// ============================================================
// 共享事务内存系统，带冲突跟踪功能
// ============================================================
// 使用每个地址的元数据(metadata)来跟踪哪个事务"拥有"每个地址的读或写权限。
// 这种设计模拟了硬件事务内存(HTM)中缓存一致性协议的简化版本。
//
// 核心数据结构：
//   TxRecord - 每个地址的事务记录，记录当前写入者和所有读取者
//   TxnDesc  - 每个事务的描述符，记录读集合、写集合和活动状态
//
// 关键概念：
//   读集合(read_set)  - 事务已读取的地址集合，用于检测读后写(read-after-write)冲突
//   写集合(write_set) - 事务已写入的地址集合，用于检测写后读/写后写冲突

struct TxRecord {
    int writer_id = -1;          // 当前正在写入的事务ID（-1 表示无写入者）
    std::unordered_set<int> reader_ids; // 当前正在读取的事务ID集合
    std::mutex mtx;              // 保护此记录的互斥锁
};

class TransactionalMemory {
public:
    TransactionalMemory() : global_version_(0) {}

    // 事务描述符 (Transaction Descriptor)
    // 每个活跃事务维护以下信息：
    //   id:        事务唯一标识符
    //   read_set:  已读取的内存地址集合 —— 用于乐观检测时验证数据一致性
    //   write_set: 已写入的内存地址集合 —— 用于提交时释放锁和更新内存
    //   active:    事务是否仍处于活跃状态
    struct TxnDesc {
        int id;
        std::unordered_set<int> read_set;  // 已读取的地址集合
        std::unordered_set<int> write_set; // 已写入的地址集合
        bool active = true;
    };

    // ---- 悲观（急切）冲突检测 (Pessimistic / Eager Detection) ----

    // 悲观读操作 (Pessimistic Read)
    // 在读取时检查写-写冲突(Write-Write)或写-读冲突(Write-Read)
    // 工作流程：
    //   1. 获取目标地址的元数据锁
    //   2. 检查是否有其他事务正在写入此地址（写-读冲突）
    //   3. 如无冲突，将当前事务注册为读取者
    //   4. 将地址添加到读集合，返回读取值
    // 返回 false 表示检测到冲突，事务应中止
    bool read_pessimistic(TxnDesc& txn, int addr, int& out_value) {
        auto& record = records_[addr];
        std::lock_guard<std::mutex> guard(record.mtx);

        // 冲突检测：另一个事务正在写入此地址
        // 这是写-读 (Write-Read) 冲突 —— 一个事务读，另一个事务写
        // 在悲观策略中，读操作会立即检测到这种冲突
        if (record.writer_id != -1 && record.writer_id != txn.id) {
            std::cout << "  [悲观检测] 读操作冲突: 事务 " << txn.id
                      << " 读取地址[" << addr << "]，但事务 " << record.writer_id
                      << " 正在写入该地址。中止！" << std::endl;
            return false; // 检测到冲突 → 中止事务
        }

        // 无冲突：注册为读取者
        record.reader_ids.insert(txn.id);
        txn.read_set.insert(addr);
        out_value = memory_[addr];
        return true;
    }

    // 悲观写操作 (Pessimistic Write)
    // 在写入时检查所有类型的冲突：读-写(R-W)、写-读(W-R)、写-写(W-W)
    // 工作流程：
    //   1. 获取目标地址的元数据锁
    //   2. 检查写-写冲突：是否有其他写入者持有此地址
    //   3. 检查读-写冲突：是否有其他读取者在读取此地址
    //   4. 如无冲突，获取此地址的写所有权（排他访问）
    // 返回 false 表示检测到冲突，事务应中止
    bool write_pessimistic(TxnDesc& txn, int addr, int value) {
        auto& record = records_[addr];
        std::lock_guard<std::mutex> guard(record.mtx);

        // 冲突检测1：写-写冲突 (Write-Write Conflict)
        // 另一个事务已经获取了此地址的写所有权
        if (record.writer_id != -1 && record.writer_id != txn.id) {
            std::cout << "  [悲观检测] 写操作冲突: 事务 " << txn.id
                      << " 写入地址[" << addr << "]，但事务 " << record.writer_id
                      << " 已拥有该地址。中止！" << std::endl;
            return false;
        }

        // 冲突检测2：读-写冲突 (Read-Write Conflict)
        // 有其他事务正在读取此地址，写入会破坏读取事务的快照一致性
        if (!record.reader_ids.empty()) {
            bool has_other_readers = false;
            for (int rid : record.reader_ids) {
                if (rid != txn.id) {
                    has_other_readers = true;
                    break;
                }
            }
            if (has_other_readers) {
                std::cout << "  [悲观检测] 写操作冲突: 事务 " << txn.id
                          << " 写入地址[" << addr << "]，但其他事务正在读取该地址。"
                          << " 中止！" << std::endl;
                return false;
            }
        }

        // 获取写所有权 (Acquire Write Ownership)
        // 设置当前事务为写入者，并清除所有读取者（获取排他访问）
        record.writer_id = txn.id;
        record.reader_ids.clear(); // 写入者获取排他访问权
        txn.write_set.insert(addr);
        return true; // 在检测时未发现冲突
    }

    // 悲观提交操作 (Pessimistic Commit)
    // 在悲观策略中，由于每次操作都已检查冲突，提交时只需：
    //   1. 做最终冲突确认（通常应该是干净的）
    //   2. 释放所有写锁
    //   3. 清除读取者记录
    // 注意：在悲观检测中，冲突通常在操作时就被检测到，所以提交相对简单
    bool commit_pessimistic(TxnDesc& txn) {
        // 最终冲突检查（在悲观检测下通常已经是干净的）
        // 这是最后一道防线，确保在提交过程中没有新冲突出现
        for (int addr : txn.write_set) {
            auto& record = records_[addr];
            std::lock_guard<std::mutex> guard(record.mtx);
            if (record.writer_id != txn.id) {
                std::cout << "  [悲观检测] 提交时在地址[" << addr << "]检测到冲突！"
                          << std::endl;
                return false;
            }
        }
        // 释放锁并更新内存
        for (int addr : txn.write_set) {
            // （在急切版本管理(eager versioning)的实际写入路径中更新内存）
            auto& record = records_[addr];
            std::lock_guard<std::mutex> guard(record.mtx);
            record.writer_id = -1;
            record.reader_ids.clear();
        }
        return true;
    }

    // ---- 乐观（懒惰）冲突检测 (Optimistic / Lazy Detection) ----

    // 乐观读操作 (Optimistic Read)
    // 仅记录读取，不检查冲突 —— 这是"乐观"的核心体现
    // 工作流程：
    //   1. 将地址添加到读集合
    //   2. 直接返回内存中的值
    //   3. 在元数据中记录此读取（供提交时检测冲突使用）
    // 关键点：乐观读假设没有冲突，冲突检测推迟到提交时
    bool read_optimistic(TxnDesc& txn, int addr, int& out_value) {
        txn.read_set.insert(addr);
        out_value = memory_[addr];

        // 在事务记录中记录此读取（供提交时的冲突检测使用）
        auto& record = records_[addr];
        std::lock_guard<std::mutex> guard(record.mtx);
        record.reader_ids.insert(txn.id);
        return true; // 乐观策略：读取时总是成功
    }

    // 乐观写操作 (Optimistic Write)
    // 仅记录写入，但会检查活跃的写-写冲突
    // 注意：即使是乐观策略，写-写冲突也会在写入时检测
    // 原因：两个事务同时写入同一地址必然导致不一致，必须立即处理
    bool write_optimistic(TxnDesc& txn, int addr, int value) {
        txn.write_set.insert(addr);

        auto& record = records_[addr];
        std::lock_guard<std::mutex> guard(record.mtx);
        // 冲突检测：是否有另一个活跃的写入者？
        // 写-写冲突即使是乐观策略也需要在写入时检测
        if (record.writer_id != -1 && record.writer_id != txn.id) {
            return false; // 与活跃写入者的写-写冲突（在写入时检测）
        }
        record.writer_id = txn.id;
        return true;
    }

    // 乐观提交操作 (Optimistic Commit)
    // 在提交时检查所有冲突 —— 这是乐观策略的核心
    // 工作流程：
    //   Check 1: 对于写集合中的每个地址，确保没有其他事务读取或写入它
    //   Check 2: 对于读集合中的每个地址，确保没有其他事务写入它
    //   如果都通过：使写入对全局可见，释放所有锁
    //   如果失败：清理读记录，返回失败
    //
    // 这是乐观策略的关键权衡所在 —— 在提交时进行批量冲突检测，
    // 可能发现事务执行了无用功（所有操作被回滚），但在无冲突场景下效率很高
    bool commit_optimistic(TxnDesc& txn) {
        // 检查1 (Check 1)：写集合验证
        // 对于每个写入的地址，确保没有其他事务已读取或写入它
        // 这是写后读(Write-After-Read)和写后写(Write-After-Write)冲突检测
        for (int addr : txn.write_set) {
            auto& record = records_[addr];
            std::lock_guard<std::mutex> guard(record.mtx);

            // 已提交的事务可能已经修改了此地址
            // 在实际的STM系统中，这通过版本号/时间戳来检测
            if (record.writer_id != txn.id) {
                std::cout << "  [乐观检测] 提交失败: 地址[" << addr
                          << "] 已被其他事务修改。" << std::endl;
                // 清理乐观读取记录
                for (int r_addr : txn.read_set) {
                    auto& r_rec = records_[r_addr];
                    std::lock_guard<std::mutex> r_guard(r_rec.mtx);
                    r_rec.reader_ids.erase(txn.id);
                }
                return false;
            }
        }

        // 检查2 (Check 2)：读集合验证
        // 对于每个读取的地址，确保没有其他事务写入它
        // 这是检测"读了过时数据"(stale read)的关键步骤
        for (int addr : txn.read_set) {
            auto& record = records_[addr];
            std::lock_guard<std::mutex> guard(record.mtx);
            // 如果数据被读取后，现在被另一个事务写锁持有...
            // 说明我们读到的数据可能已经过时
            if (record.writer_id != -1 && record.writer_id != txn.id) {
                std::cout << "  [乐观检测] 提交失败: 地址[" << addr
                          << "] 被读取后又被其他事务写入。" << std::endl;
                // 清理读记录
                for (int r_addr : txn.read_set) {
                    auto& r_rec = records_[r_addr];
                    std::lock_guard<std::mutex> r_guard(r_rec.mtx);
                    r_rec.reader_ids.erase(txn.id);
                }
                return false;
            }
        }

        // 提交成功：使写入可见，释放锁
        // 在乐观策略中，写入之前被缓冲，现在才真正生效
        for (int addr : txn.write_set) {
            auto& record = records_[addr];
            std::lock_guard<std::mutex> guard(record.mtx);
            memory_[addr] = write_buffer_[addr];
            record.writer_id = -1;
            record.reader_ids.clear();
        }
        for (int addr : txn.read_set) {
            auto& record = records_[addr];
            std::lock_guard<std::mutex> guard(record.mtx);
            record.reader_ids.erase(txn.id);
        }

        std::cout << "  [乐观检测] 提交成功！" << std::endl;
        return true;
    }

    // 为乐观写入缓冲存储值
    // 在乐观策略中，写操作不会立即更新共享内存，而是写入缓冲区
    // 只有在提交成功后才将缓冲区内容刷新到共享内存
    void buffer_write(int addr, int value) {
        write_buffer_[addr] = value;
    }

    // 非事务性或急切写入的直接内存访问
    void direct_write(int addr, int value) {
        memory_[addr] = value;
    }

    int direct_read(int addr) const {
        auto it = memory_.find(addr);
        return (it != memory_.end()) ? it->second : 0;
    }

    void reset() {
        memory_.clear();
        write_buffer_.clear();
        records_.clear();
    }

private:
    std::unordered_map<int, int> memory_;          // 实际共享内存
    std::unordered_map<int, int> write_buffer_;    // 乐观策略的写缓冲区
    std::unordered_map<int, TxRecord> records_;    // 每个地址的元数据（事务记录）
    std::atomic<uint64_t> global_version_;
};

// ============================================================
// 演示程序 (Demonstration)
// ============================================================
// 通过三个场景展示悲观检测和乐观检测的行为差异。
// 场景1: 无真实冲突 —— 两个策略都应该成功提交
// 场景2: 写-写冲突  —— 展示悲观检测即时发现冲突 vs 乐观检测在提交时发现冲突
// 场景3: 读-写冲突  —— 展示两种策略的不同处理方式

int main() {
    std::cout << "=== CS149 讲座 17: 事务内存冲突检测 ===" << std::endl;
    std::cout << std::endl;

    // ---- 场景1: 无真实冲突 ----
    // 事务0读取A并写入A=42，事务1写入B=99
    // 两个事务操作的是不同的地址，因此没有实际冲突
    // 预期：两种策略下两个事务都应该成功提交
    std::cout << "--- 场景1: 无真实冲突 ---" << std::endl;
    std::cout << "  事务 0: 读取A, 写入A=42" << std::endl;
    std::cout << "  事务 1: 写入B=99" << std::endl;
    std::cout << "  预期结果: 两个事务都提交成功 (A和B是不同的地址)" << std::endl;
    std::cout << std::endl;

    {
        TransactionalMemory tm;
        tm.direct_write(0, 10); // 初始化 A=10 在地址 0
        tm.direct_write(1, 20); // 初始化 B=20 在地址 1

        // 悲观策略方法
        {
            std::cout << "[悲观检测]" << std::endl;
            TransactionalMemory::TxnDesc txn0{0};
            TransactionalMemory::TxnDesc txn1{1};

            int val_a;
            bool r_ok = tm.read_pessimistic(txn0, 0, val_a);
            assert(r_ok);
            std::cout << "  事务 0 读取 A=" << val_a << std::endl;

            bool w_ok0 = tm.write_pessimistic(txn0, 0, 42);
            assert(w_ok0);
            tm.direct_write(0, 42);
            std::cout << "  事务 0 写入 A=42" << std::endl;

            bool w_ok1 = tm.write_pessimistic(txn1, 1, 99);
            assert(w_ok1);
            tm.direct_write(1, 99);
            std::cout << "  事务 1 写入 B=99" << std::endl;

            tm.commit_pessimistic(txn0);
            tm.commit_pessimistic(txn1);
            std::cout << "  结果: A=" << tm.direct_read(0) << " B=" << tm.direct_read(1) << std::endl;
        }

        tm.reset();
        tm.direct_write(0, 10);
        tm.direct_write(1, 20);

        // 乐观策略方法
        // 与悲观策略不同，写操作使用缓冲写入(buffer_write)，在提交时才真正生效
        {
            std::cout << "[乐观检测]" << std::endl;
            TransactionalMemory::TxnDesc txn0{0};
            TransactionalMemory::TxnDesc txn1{1};

            int val_a;
            tm.read_optimistic(txn0, 0, val_a);
            std::cout << "  事务 0 读取 A=" << val_a << std::endl;

            tm.write_optimistic(txn0, 0, 42);
            tm.buffer_write(0, 42);
            std::cout << "  事务 0 缓冲写入 A=42" << std::endl;

            tm.write_optimistic(txn1, 1, 99);
            tm.buffer_write(1, 99);
            std::cout << "  事务 1 缓冲写入 B=99" << std::endl;

            tm.commit_optimistic(txn0);
            tm.commit_optimistic(txn1);
            std::cout << "  结果: A=" << tm.direct_read(0) << " B=" << tm.direct_read(1) << std::endl;
        }
    }

    // ---- 场景2: 写-写冲突 ----
    // 两个事务都试图写入同一个地址A
    // 悲观检测：第二个写操作立即检测到冲突
    // 乐观检测：写操作时检测到活跃写入者冲突
    // 关键区别：悲观检测在每次操作时都检查，乐观检测仅在写入时检查活跃写入者
    std::cout << std::endl;
    std::cout << "--- 场景2: 地址A上的写-写冲突 ---" << std::endl;
    std::cout << "  事务 0: 写入 A=42" << std::endl;
    std::cout << "  事务 1: 写入 A=99 (与事务0冲突！)" << std::endl;
    std::cout << std::endl;

    {
        TransactionalMemory tm;
        tm.direct_write(0, 10);

        // 悲观策略：在第二次写操作时立即检测到冲突
        // 这是悲观策略的优势 —— 不会在注定失败的操作上浪费时间
        {
            std::cout << "[悲观检测]" << std::endl;
            TransactionalMemory::TxnDesc txn0{0};
            TransactionalMemory::TxnDesc txn1{1};

            bool w0 = tm.write_pessimistic(txn0, 0, 42);
            std::cout << "  事务 0 写入 A=42: " << (w0 ? "成功" : "冲突") << std::endl;

            bool w1 = tm.write_pessimistic(txn1, 0, 99);
            std::cout << "  事务 1 写入 A=99: " << (w1 ? "成功" : "冲突（立即检测到！）")
                      << std::endl;
            std::cout << "  → 悲观检测在写入时立刻发现了冲突！" << std::endl;
        }
    }

    {
        TransactionalMemory tm;
        tm.direct_write(0, 10);

        // 乐观策略：写-写冲突也会在写入时检测
        // 这是因为写-写冲突本质上是资源独占问题，即使是乐观策略也需要及时处理
        {
            std::cout << "[乐观检测]" << std::endl;
            TransactionalMemory::TxnDesc txn0{0};
            TransactionalMemory::TxnDesc txn1{1};

            bool w0 = tm.write_optimistic(txn0, 0, 42);
            tm.buffer_write(0, 42);
            std::cout << "  事务 0 写入 A=42: " << (w0 ? "成功（乐观策略）" : "冲突") << std::endl;

            // 事务1尝试写入A：即使在乐观策略下也应失败（存在活跃写入者冲突）
            bool w1 = tm.write_optimistic(txn1, 0, 99);
            std::cout << "  事务 1 写入 A=99: " << (w1 ? "成功" : "冲突（已存在活跃写入者）")
                      << std::endl;
        }
    }

    // ---- 场景3: 读-写冲突 ----
    // 事务0读取A，事务1写入A=99
    // 悲观检测：事务1的写操作检测到有读取者 → 中止
    // 乐观检测：事务0读取，事务1写入；
    //   如果事务0先提交 → 成功；
    //   如果事务1先提交，事务0的读集合数据已过时 → 中止事务0
    // 这个场景清晰地展示了两种策略在"检测时机"上的根本差异
    std::cout << std::endl;
    std::cout << "--- 场景3: 读-写冲突 ---" << std::endl;
    std::cout << "  事务 0: 读取 A" << std::endl;
    std::cout << "  事务 1: 写入 A=99" << std::endl;
    std::cout << "  悲观检测: 事务1的写操作检测到有读取者 → 中止" << std::endl;
    std::cout << "  乐观检测: 事务0读取，事务1写入；在提交时，"
              << std::endl;
    std::cout << "    如果事务0先提交 → 成功；如果事务1先提交，"
              << std::endl;
    std::cout << "    事务0的读集合已过时 → 中止事务0" << std::endl;
    std::cout << std::endl;

    // ---- 总结对比 ----
    std::cout << "=== 对比总结 ===" << std::endl;
    std::cout << "+---------------------+--------------------------+--------------------------+" << std::endl;
    std::cout << "|      对比维度        |  悲观检测（急切式）      |  乐观检测（懒惰式）       |" << std::endl;
    std::cout << "+---------------------+--------------------------+--------------------------+" << std::endl;
    std::cout << "| 检测时机             | 每次加载/存储操作时      | 仅在提交时               |" << std::endl;
    std::cout << "| 是否早中止           | 是（减少浪费的工作）    | 否（可能执行无用功）     |" << std::endl;
    std::cout << "| 前向进度保证         | 无法保证（可能活锁）    | 有保证                  |" << std::endl;
    std::cout << "| 通信粒度             | 细粒度（每次操作）      | 批量（提交时）           |" << std::endl;
    std::cout << "| 额外开销             | 每次操作都有开销        | 提交前开销低             |" << std::endl;
    std::cout << "| 能否暂缓执行         | 是（某些情况下可选择    | 否（冲突时总是中止）     |" << std::endl;
    std::cout << "|                      |  暂缓而非中止）          |                          |" << std::endl;
    std::cout << "+---------------------+--------------------------+--------------------------+" << std::endl;

    return 0;
}
