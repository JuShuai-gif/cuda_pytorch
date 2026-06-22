/*
 * lecture17_part2.cpp - 事务内存数据版本管理：急切版本 vs 惰性版本
 * Stanford CS149, Fall 2025 - Lecture 17
 *
 * 本程序模拟事务内存（Transactional Memory, TM）中两种数据版本管理策略，
 * 通过一个简单的键值存储系统来对比两者的差异。
 *
 * === 讲座核心概念 ===
 *
 * 事务内存需要解决一个关键设计问题：在事务执行期间，写操作应该如何处理？
 * 是立即写入共享内存，还是先缓存起来，等到提交时再写入？
 * 这个问题引出了两种截然不同的版本管理策略：
 *
 * 策略 1: 急切版本管理 (Eager Versioning / Undo-Log Based)
 *   核心思想：「立即写入内存，假设事务不会中止。如果真中止了，再撤销回来。」
 *   - 写操作：直接修改共享内存中的值，同时在 undo-log 中记录旧值
 *   - 提交操作：很快——数据已经在内存中，只需清空 undo-log
 *   - 中止操作：很慢——需要按逆序回放 undo-log，把旧值写回内存
 *   - 容错性差：如果系统在事务执行期间崩溃，内存中留有部分更新的脏数据
 *   - 隔离性弱：未提交的写操作对其他线程是立即可见的
 *
 * 策略 2: 惰性版本管理 (Lazy Versioning / Write-Buffer Based)
 *   核心思想：「只在提交时才写入内存。在此之前，所有写入都缓存起来。」
 *   - 写操作：将新值缓存在事务私有的 write-buffer 中，不触碰共享内存
 *   - 提交操作：很慢——需要将 write-buffer 中的所有更新刷入共享内存
 *   - 中止操作：很快——只需丢弃 write-buffer，不产生任何副作用
 *   - 容错性好：崩溃时不会在内存中留下部分更新（全有或全无语义）
 *   - 隔离性强：未提交的写操作对其他线程完全不可见
 *
 * === 权衡分析（Trade-off Analysis）===
 *
 * 急切版本管理在「大部分事务都提交」的场景下表现更好：
 *   - 提交快（常见情况优化），中止慢（罕见情况可以接受）
 *   - 类比：乐观地假设事务会成功
 *
 * 惰性版本管理在「冲突频繁」的场景下表现更好：
 *   - 中止快（冲突时快速回滚），提交慢（但提交本身就是序列化的）
 *   - 类比：保守地保护共享数据
 *
 * 编译: g++ -std=c++17 -pthread lecture17_part2.cpp -o lecture17_part2
 * 运行: ./lecture17_part2
 */

#include <iostream>
#include <thread>
#include <vector>
#include <unordered_map>
#include <cassert>
#include <chrono>
#include <atomic>
#include <mutex>
#include <shared_mutex>

// ============================================================
// 共享内存：一个简单的键值存储 (Key-Value Store)
// ============================================================
// 使用读写锁 (shared_mutex) 保护，允许多个读者并发访问，
// 同时保证写者的独占访问。
// 这个共享内存模拟了事务内存系统中的「全局内存状态」。
//
// 读写锁的工作原理：
//   - shared_lock (读锁): 多个线程可以同时持有读锁
//   - unique_lock (写锁): 只有一个线程可以持有写锁，且排斥所有读锁
// 这种设计在「读多写少」的场景下能显著提升并发性能。

class SharedMemory {
public:
    // 写入操作：获取独占写锁
    void write(int addr, int value) {
        std::unique_lock lock(mtx_);
        memory_[addr] = value;
    }

    // 读取操作：获取共享读锁（允许多个读者并发）
    int read(int addr) {
        std::shared_lock lock(mtx_);
        auto it = memory_.find(addr);
        return (it != memory_.end()) ? it->second : 0;
    }

    // 获取完整快照：用于演示目的
    std::unordered_map<int, int> snapshot() const {
        std::shared_lock lock(mtx_);
        return memory_;
    }

private:
    std::unordered_map<int, int> memory_;
    mutable std::shared_mutex mtx_; // mutable 允许在 const 方法中加锁
};

// ============================================================
// 第 1 部分：急切版本管理事务 (Eager Versioning / Undo-Log Based)
// ============================================================
//
// === 设计哲学 ===
// 「立即写入内存，相信事务不会中止。如果真的需要中止，再来处理回滚。」
//
// === 执行流程 ===
//
// 写入操作 (write):
//   1. 读取当前内存中的旧值
//   2. 将 {地址, 旧值} 记录到 undo-log 中
//   3. 立即将新值写入共享内存（这是「急切」的本质）
//   4. 记录写集合（用于冲突检测）
//
// 读取操作 (read):
//   直接从共享内存读取（因为没有写缓冲，读自己的写就是读共享内存）
//   记录读集合用于冲突检测
//
// 提交操作 (commit):
//   - 很快：数据已经在共享内存中！
//   - 只需将事务状态标记为 COMMITTED
//   - 清空 undo-log（不再需要回滚信息）
//
// 中止操作 (abort):
//   - 很慢：需要逐一撤销已经写入的修改！
//   - 按逆序回放 undo-log：最后写入的值最先撤销
//   - 将每个被修改的地址恢复为事务开始前的旧值
//
// === 容错性分析 ===
// 如果系统在事务执行期间崩溃：
//   - 内存中可能留有部分更新的数据（部分提交的脏数据）
//   - 恢复时需要依靠崩溃恢复机制（如 WAL 日志）
//   - 这是急切版本管理的主要缺点

class EagerTransaction {
public:
    enum State { ACTIVE, COMMITTED, ABORTED };

    EagerTransaction(SharedMemory& mem) : mem_(mem), state_(ACTIVE) {}

    // 写入操作：急切模式 - 直接写入共享内存，同时保存撤销信息
    // 参数:
    //   addr  - 要写入的内存地址
    //   value - 要写入的新值
    void write(int addr, int value) {
        assert(state_ == ACTIVE && "事务已经终止，不能再执行操作");

        // 保存旧值到 undo-log，用于可能的中止回滚
        int old_value = mem_.read(addr);
        undo_log_.push_back({addr, old_value});

        // 立即写入共享内存（急切版本管理的核心特征）
        mem_.write(addr, value);

        write_set_.push_back(addr);
    }

    // 读取操作：急切模式直接从共享内存读取
    // （因为没有写缓冲，自己的写入已经反映在共享内存中）
    int read(int addr) {
        assert(state_ == ACTIVE && "事务已经终止，不能再执行操作");
        read_set_.push_back(addr);
        return mem_.read(addr);
    }

    // 提交操作：很快！数据已经在共享内存中
    // 只需清空 undo-log 即可（不再需要回滚能力）
    void commit() {
        assert(state_ == ACTIVE && "事务已经终止，不能重复提交");
        state_ = COMMITTED;
        undo_log_.clear(); // 丢弃 undo log，提交后不再需要回滚
        std::cout << "  [急切版本] 已提交。Undo log 已清空。（提交很快，因为数据早已在内存中）" << std::endl;
    }

    // 中止操作：很慢！需要按逆序回放 undo-log
    // 为什么是逆序？因为最后写入的值可能依赖前面的值，
    // 逆序恢复可以保证数据一致性
    void abort_txn() {
        assert(state_ == ACTIVE && "事务已经终止，不能重复中止");
        std::cout << "  [急切版本] 正在中止！回放 Undo log 中..." << std::endl;

        // 按逆序回放 undo log（最后写入的最先撤销）
        // rbegin()/rend() 提供反向迭代器
        for (auto it = undo_log_.rbegin(); it != undo_log_.rend(); ++it) {
            std::cout << "    撤销: addr[" << it->addr << "] ← " << it->old_value
                      << " (恢复为事务开始前的值)" << std::endl;
            mem_.write(it->addr, it->old_value);
        }
        state_ = ABORTED;
    }

    State get_state() const { return state_; }
    const std::vector<int>& get_read_set() const { return read_set_; }
    const std::vector<int>& get_write_set() const { return write_set_; }

private:
    // Undo 条目：记录 {地址, 旧值} 对
    struct UndoEntry {
        int addr;
        int old_value;
    };

    SharedMemory& mem_;
    State state_;
    std::vector<int> read_set_;   // 读集合：用于冲突检测
    std::vector<int> write_set_;  // 写集合：用于冲突检测
    std::vector<UndoEntry> undo_log_; // Undo 日志：用于中止时回滚
};

// ============================================================
// 第 2 部分：惰性版本管理事务 (Lazy Versioning / Write-Buffer Based)
// ============================================================
//
// === 设计哲学 ===
// 「只在提交时才写入内存。在此之前，将所有修改缓存起来，不污染共享状态。」
//
// === 执行流程 ===
//
// 写入操作 (write):
//   1. 将新值存入事务私有的 write-buffer (write_buffer_)
//   2. 不触碰共享内存！（这是「惰性」的本质）
//   3. 记录写集合用于冲突检测
//
// 读取操作 (read):
//   1. 首先检查 write-buffer：如果该地址已被本事务写入，返回缓冲值
//      （这就是「读取自己的写入」/ read-your-own-writes 语义）
//   2. 如果地址不在 write-buffer 中，从共享内存读取
//   3. 记录读集合用于冲突检测
//
// 提交操作 (commit):
//   - 很慢：需要遍历 write-buffer，将每个修改刷入共享内存
//   - 将所有缓冲的 {地址, 值} 对依次写入共享内存
//   - 提交完成后再清空 write-buffer
//
// 中止操作 (abort):
//   - 很快：只需丢弃 write-buffer！
//   - 因为从未修改过共享内存，不需要任何回滚操作
//
// === 隔离性分析 ===
// 惰性版本管理提供了更强的隔离性：
//   - 未提交的写入对其他事务完全不可见
//   - 实现了真正的「全有或全无」(all-or-nothing) 语义
//   - 崩溃时内存中不会有部分更新的数据

class LazyTransaction {
public:
    enum State { ACTIVE, COMMITTED, ABORTED };

    LazyTransaction(SharedMemory& mem) : mem_(mem), state_(ACTIVE) {}

    // 写入操作：惰性模式 - 将写入缓存到 write-buffer 中，不触碰共享内存
    // 参数:
    //   addr  - 要写入的内存地址
    //   value - 要写入的新值（先缓存，提交时才真正写入）
    void write(int addr, int value) {
        assert(state_ == ACTIVE && "事务已经终止，不能再执行操作");
        write_buffer_[addr] = value; // 仅缓存，不写入共享内存
        write_set_.push_back(addr);
    }

    // 读取操作：先检查 write-buffer（读取自己的写入），
    // 如果没找到再从共享内存读取。
    // 这实现了「read-your-own-writes」语义：事务可以看到自己之前的写入，
    // 即使这些写入还没有提交到共享内存。
    int read(int addr) {
        assert(state_ == ACTIVE && "事务已经终止，不能再执行操作");
        read_set_.push_back(addr);

        // 检查该地址是否被本事务写入过
        // 如果是，返回 write-buffer 中的值（读取自己的未提交写入）
        auto it = write_buffer_.find(addr);
        if (it != write_buffer_.end()) {
            return it->second; // 读取本事务自己的写入（尚未对外提交）
        }
        return mem_.read(addr); // 未写入过，从共享内存读取
    }

    // 提交操作：将 write-buffer 中的所有更新刷入共享内存
    // 这是惰性版本管理最慢的操作——所有延迟的写入都在这一刻集中执行
    void commit() {
        assert(state_ == ACTIVE && "事务已经终止，不能重复提交");
        std::cout << "  [惰性版本] 正在提交。将写缓冲刷入共享内存..." << std::endl;

        // 遍历 write-buffer，将所有缓冲的修改写入共享内存
        for (const auto& [addr, value] : write_buffer_) {
            std::cout << "    刷入: addr[" << addr << "] ← " << value << std::endl;
            mem_.write(addr, value);
        }
        state_ = COMMITTED;
        write_buffer_.clear(); // 提交完成，清空缓冲区
    }

    // 中止操作：极快！只需丢弃 write-buffer
    // 因为从未修改过共享内存，不需要任何回滚
    void abort_txn() {
        assert(state_ == ACTIVE && "事务已经终止，不能重复中止");
        std::cout << "  [惰性版本] 正在中止！直接丢弃写缓冲（无需回滚，因为共享内存从未被修改）。" << std::endl;
        write_buffer_.clear(); // 丢弃所有缓冲的写入
        state_ = ABORTED;
    }

    State get_state() const { return state_; }
    const std::vector<int>& get_read_set() const { return read_set_; }
    const std::vector<int>& get_write_set() const { return write_set_; }

private:
    SharedMemory& mem_;
    State state_;
    std::vector<int> read_set_;   // 读集合：用于冲突检测
    std::vector<int> write_set_;  // 写集合：用于冲突检测
    std::unordered_map<int, int> write_buffer_; // 事务私有的写缓冲
};

// ============================================================
// 演示部分
// ============================================================

// 演示急切版本管理（Undo-Log 方式）的完整生命周期
void demo_eager_versioning() {
    std::cout << "=== 急切版本管理（基于 Undo Log） ===" << std::endl;

    SharedMemory mem;
    mem.write(100, 10); // 初始化: addr[100] = 10
    mem.write(200, 20); // 初始化: addr[200] = 20

    std::cout << "初始状态: addr[100]=" << mem.read(100)
              << ", addr[200]=" << mem.read(200) << std::endl;

    // 场景 1：中止 (Abort)
    // 展示急切版本管理中中止的开销——需要回放 undo log
    {
        EagerTransaction txn(mem);
        txn.write(100, 42); // 急切写入：立即将 42 写入内存，同时保存 undo(100, 旧值=10)
        txn.write(200, 99); // 急切写入：立即将 99 写入内存，同时保存 undo(200, 旧值=20)
        std::cout << "急切写入后: addr[100]=" << mem.read(100)
                  << " (立即可见！), addr[200]=" << mem.read(200) << std::endl;

        // 模拟：决定中止事务
        txn.abort_txn();
    }

    std::cout << "中止后: addr[100]=" << mem.read(100)
              << " (已从 undo log 恢复), addr[200]=" << mem.read(200)
              << " (已从 undo log 恢复)" << std::endl;
    std::cout << std::endl;

    // 场景 2：提交 (Commit)
    // 展示急切版本管理中提交的速度——几乎不需要做什么
    {
        EagerTransaction txn(mem);
        txn.write(100, 500); // 急切写入：立即生效
        txn.commit();        // 提交很快：只需清空 undo log
    }

    std::cout << "提交后: addr[100]=" << mem.read(100) << " (值保持不变，提交只是确认)" << std::endl;
    std::cout << std::endl;
}

// 演示惰性版本管理（Write-Buffer 方式）的完整生命周期
void demo_lazy_versioning() {
    std::cout << "=== 惰性版本管理（基于写缓冲） ===" << std::endl;

    SharedMemory mem;
    mem.write(100, 10); // 初始化: addr[100] = 10
    mem.write(200, 20); // 初始化: addr[200] = 20

    std::cout << "初始状态: addr[100]=" << mem.read(100)
              << ", addr[200]=" << mem.read(200) << std::endl;

    // 场景 1：中止 (Abort)
    // 展示惰性版本管理中中止的速度——只需丢弃缓冲区
    {
        LazyTransaction txn(mem);
        txn.write(100, 42); // 惰性写入：仅缓存到 write-buffer，共享内存不变
        txn.write(200, 99); // 惰性写入：仅缓存到 write-buffer，共享内存不变
        std::cout << "惰性写入后（尚未写入共享内存）: addr[100]=" << mem.read(100)
                  << " (仍然是旧值！), addr[200]=" << mem.read(200) << " (仍然是旧值！)" << std::endl;

        // 读取自己的写入 (Read-Your-Own-Writes)
        // 事务内部可以看到自己缓冲的写入
        std::cout << "事务内部读取: addr[100]=" << txn.read(100)
                  << " (看到了自己的写入), addr[200]=" << txn.read(200)
                  << " (看到了自己的写入)" << std::endl;

        // 模拟中止
        txn.abort_txn();
    }

    std::cout << "中止后: addr[100]=" << mem.read(100)
              << " (未改变 - 缓冲只是被丢弃，共享内存从未被修改)" << std::endl;
    std::cout << "         addr[200]=" << mem.read(200)
              << " (未改变 - 同上)" << std::endl;
    std::cout << std::endl;

    // 场景 2：提交 (Commit)
    // 展示惰性版本管理中提交的开销——需要刷入所有缓冲的写入
    {
        LazyTransaction txn(mem);
        txn.write(100, 500); // 缓冲: addr[100] ← 500
        txn.write(200, 600); // 缓冲: addr[200] ← 600
        txn.commit();        // 提交慢：需要将所有缓冲刷入共享内存
    }

    std::cout << "提交后: addr[100]=" << mem.read(100)
              << " (现在对外可见), addr[200]=" << mem.read(200)
              << " (现在对外可见)" << std::endl;
    std::cout << std::endl;
}

// 两种版本管理策略的全面对比
void demo_comparison() {
    std::cout << "=== 全面对比总结 ===" << std::endl;
    std::cout << "+----------------------+--------------------------------+--------------------------------+" << std::endl;
    std::cout << "|      对比维度          |  急切版本 (Undo-Log)            |  惰性版本 (Write-Buffer)        |" << std::endl;
    std::cout << "+----------------------+--------------------------------+--------------------------------+" << std::endl;
    std::cout << "| 提交速度 (Commit)     | 快（数据已在内存中）            | 慢（需刷入写缓冲）              |" << std::endl;
    std::cout << "| 中止速度 (Abort)      | 慢（需回放 Undo Log）           | 快（直接丢弃缓冲）              |" << std::endl;
    std::cout << "| 容错性 (Fault Tol.)   | 差（崩溃留部分更新）            | 好（全有或全无语义）            |" << std::endl;
    std::cout << "| 写入可见性            | 立即（对所有线程可见）          | 延迟（提交时才可见）            |" << std::endl;
    std::cout << "| 隔离性 (Isolation)    | 较弱（未提交写操作可能泄漏）    | 较强（所有写入都缓冲）          |" << std::endl;
    std::cout << "| 每次写入开销          | 记录旧值到 Undo Log            | 缓存新值到 Write Buffer         |" << std::endl;
    std::cout << "| 适用场景              | 冲突少、提交多的工作负载        | 冲突多、中频繁的工作负载        |" << std::endl;
    std::cout << "+----------------------+--------------------------------+--------------------------------+" << std::endl;

    // 补充说明
    std::cout << std::endl;
    std::cout << "关键洞察：" << std::endl;
    std::cout << "  - 急切版本 = 乐观假设事务会成功，优化常见路径（提交）" << std::endl;
    std::cout << "  - 惰性版本 = 保守保护共享数据，优化异常路径（中止）" << std::endl;
    std::cout << "  - 实际系统中，选择取决于工作负载的「中止率」和「容错性需求」" << std::endl;
    std::cout << "  - 现代 HTM (Hardware Transactional Memory) 通常使用急切版本管理，" << std::endl;
    std::cout << "    因为硬件缓存一致性协议天然支持这种模式" << std::endl;
    std::cout << "  - STM (Software Transactional Memory) 可以灵活选择两种策略" << std::endl;
}

int main() {
    std::cout << "=== CS149 第17讲：事务内存数据版本管理 ===" << std::endl;
    std::cout << std::endl;

    demo_eager_versioning();
    demo_lazy_versioning();
    demo_comparison();

    return 0;
}
