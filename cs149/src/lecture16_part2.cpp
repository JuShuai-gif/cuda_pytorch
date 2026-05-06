/*
 * lecture16_part2.cpp - 细粒度锁：手递手（Hand-over-Hand）链表锁定
 * Stanford CS149, Fall 2025 - Lecture 16（第16讲：细粒度同步）
 *
 * 本文件详细演示了排序链表上的细粒度同步技术，包含两种策略的对比：
 *
 *   1. 粗粒度锁（Coarse-Grained Locking）：单个全局锁
 *      - 最简单的线程安全实现：所有操作（插入、删除、查找）
 *        都受同一个互斥锁保护
 *      - 优点：实现简单，不易出错，死锁风险低
 *      - 缺点：完全串行化所有操作，无并行性可言
 *        （即使两个线程操作链表的不同部分也必须等待）
 *
 *   2. 手递手锁定（Hand-over-Hand Locking）：细粒度锁
 *      - 每个链表节点都有自己的互斥锁
 *      - 遍历过程中采用"手递手"策略：
 *        锁定当前节点 → 锁定下一个节点 → 解锁当前节点
 *        （像攀岩时一只手先抓住下一个支点，再松开前一个支点）
 *      - 关键优势：允许对不同链表段进行并发操作
 *        （例如，一个线程在头部插入，另一个线程在尾部删除）
 *
 * 手递手锁定的核心设计原理：
 *   - 锁的获取顺序始终沿链表遍历方向（从头到尾），
 *     这保证了无死锁（deadlock-free）特性
 *   - 不会出现循环等待（circular wait）条件：
 *     线程 A 等线程 B 释放节点 N 的锁，
 *     但线程 B 只可能在获取节点 N 之后的锁时等待，
 *     不可能回头等待线程 A 已持有的锁
 *   - 插入操作的特殊处理：持有 prev 节点的锁即可安全地
 *     修改 prev->next 指针（因为其他线程要访问该指针
 *     必须先获取 prev 的锁）
 *
 * 手递手锁定的实现细节（以插入为例）：
 *   Step 1: 锁定头节点（哨兵节点）
 *   Step 2: 锁定第一个真实节点
 *   Step 3: 循环遍历：
 *           a. 锁定下一个节点（cur->next）
 *           b. 解锁之前的节点（old_prev）
 *           c. 前进指针（prev = cur, cur = cur->next）
 *           此时始终持有两个锁：prev 和 cur
 *   Step 4: 找到插入位置后：
 *           - 创建新节点并插入到 prev 和 cur 之间
 *           - 因为持有 prev 的锁，其他线程无法修改 prev->next
 *   Step 5: 释放剩余的锁（prev 和 cur）
 *
 * 编译命令：g++ -std=c++17 -pthread lecture16_part2.cpp -o lecture16_part2
 * 运行命令：./lecture16_part2
 */

#include <iostream>
#include <thread>
#include <climits>
#include <mutex>
#include <vector>
#include <cassert>
#include <memory>
#include <functional>

// ============================================================
// 链表节点结构：每个节点带有自己的互斥锁（用于细粒度锁定）
// ============================================================
// FGLNode（Fine-Grained Locking Node）的设计：
//   - value：节点存储的整数值
//   - next：指向下一个节点的 unique_ptr（独占所有权，自动内存管理）
//   - lock：每个节点自己的互斥锁，用于手递手锁定
//
// 使用 unique_ptr 管理节点所有权的优势：
//   - 自动释放内存：当节点从链表中移除时自动 delete
//   - 独占所有权语义：一个节点只能被一个 unique_ptr 拥有，
//     防止了"双重释放"和"悬空指针"问题
//   - 与手递手锁定配合良好：持有父节点的锁时可以安全地
//     转移子节点的所有权（通过 release/reset）

struct FGLNode {
    int value;                      // 节点存储的值
    std::unique_ptr<FGLNode> next;  // 指向下一个节点（独占所有权）
    std::mutex lock;                // 每个节点自己的互斥锁
                                    // 用于手递手锁定策略

    FGLNode(int v) : value(v), next(nullptr) {}
};

// ============================================================
// 细粒度链表类（手递手锁定实现）
// ============================================================
// 这是本 lecture 的核心实现。FineGrainedList 是一个线程安全的
// 排序链表，使用手递手锁定策略实现对不同链表段的并发访问。
//
// 数据结构设计：
//   链表使用哨兵节点（sentinel node）模式：
//     - 头部哨兵：value = -1，永远存在，简化边界条件处理
//     - 尾部哨兵：value = INT_MAX，永远存在，避免空指针检查
//   这种设计使得插入和删除操作不需要特殊处理"空链表"或
//   "头部/尾部"边界情况
//
// 死锁安全性分析：
//   手递手锁定是无死锁的，原因如下：
//   1. 所有线程按相同的全局顺序获取锁（从头节点到尾节点）
//   2. 不存在循环等待：如果线程 T1 持有节点 A 的锁并在等待
//      节点 B 的锁，那么 B 一定在 A 之后；线程 T2 不可能在
//      持有 B 的锁时等待 A 的锁（因为遍历方向固定）
//   3. 每个线程在任意时刻最多持有两个相邻节点的锁
//
// 并发性分析：
//   临界区的粒度是"两个相邻节点"，而非整个链表：
//   - 线程 T1 在节点 [A, B] 区间操作
//   - 线程 T2 在节点 [C, D] 区间操作
//   - 如果这两个区间不重叠，T1 和 T2 可以完全并发执行

class FineGrainedList {
public:
    FineGrainedList() {
        // 创建哨兵头节点，value = -1（比任何正常值都小）
        // 哨兵节点永不删除，简化了"删除第一个节点"等边界情况
        head_ = std::make_unique<FGLNode>(-1);

        // 创建哨兵尾节点，value = INT_MAX（比任何正常值都大）
        // 这样所有正常值都在 [-1, INT_MAX) 范围内，
        // 遍历时总会在找到插入位置之前遇到尾哨兵
        head_->next = std::make_unique<FGLNode>(INT_MAX);
    }

    // ============================================================
    // 插入操作：在保持排序顺序的同时插入一个值
    // 使用手递手锁定策略进行遍历
    //
    // 返回值：true 表示插入成功，false 表示值已存在（重复）
    // ============================================================
    bool insert(int value) {
        // ==== 步骤 1：锁定头节点 ====
        // 所有遍历都从头节点开始，所以必须首先锁住它
        head_->lock.lock();

        FGLNode* prev = head_.get();      // 前驱节点指针（原始指针，不拥有所有权）
        FGLNode* cur = prev->next.get();  // 当前节点指针

        // ==== 步骤 2：锁定第一个真实节点（尾哨兵或普通节点） ====
        cur->lock.lock();
        // 此时持有两个锁：head_ 和 head_->next
        // 这是手递手锁定的初始状态

        // ==== 步骤 3：手递手遍历 ====
        // 遍历直到找到插入位置：cur->value >= value
        // （链表是升序排列的，所以要找到第一个 >= value 的位置）
        while (cur->value < value) {
            // 保存旧的前驱节点（稍后需要解锁它）
            FGLNode* old_prev = prev;

            // 前进指针
            prev = cur;
            cur = cur->next.get();

            // "手递手"的核心操作：
            // 1. 锁定新节点（cur）  —— "一只手抓住下一个支点"
            // 2. 解锁旧节点（old_prev）—— "松开前一个支点"
            cur->lock.lock();          // 先获取新锁
            old_prev->lock.unlock();   // 再释放旧锁
            // 关键：必须先锁新节点再解旧节点，否则会出现
            //       "无锁保护"的窗口期，其他线程可能在此期间修改链表
        }
        // 循环结束时：
        //   - 持有 prev 和 cur 两个节点的锁
        //   - cur->value >= value（找到了插入位置）

        // ==== 步骤 4：检查重复值 ====
        // 如果 cur->value == value，说明该值已存在
        if (cur->value == value) {
            // 释放所有持有的锁后返回
            prev->lock.unlock();
            cur->lock.unlock();
            return false; // 重复值，不插入
        }

        // ==== 步骤 5：创建新节点并插入 ====
        // 创建包含新值的节点
        auto new_node = std::make_unique<FGLNode>(value);

        // 将新节点插入到 prev 和 cur 之间：
        //   之前：prev -> cur -> ...
        //   之后：prev -> new_node -> cur -> ...
        //
        // 步骤分解：
        // 1. prev->next.release()：释放 prev 对 cur 的所有权，
        //    返回指向 cur 的原始指针
        // 2. new_node->next.reset(...)：让 new_node 接管 cur 的所有权
        // 3. prev->next.reset(...)：让 prev 接管 new_node 的所有权
        //
        // 因为持有 prev->lock，其他线程无法访问 prev->next，
        // 这些指针操作是线程安全的

        // new_node->next 接管 cur（cur 的所有权从 prev 转移到 new_node）
        new_node->next.reset(prev->next.release());

        // prev->next 接管 new_node（完成插入）
        // 此时：prev -> new_node -> cur
        prev->next.reset(new_node.release());

        // ==== 步骤 6：释放剩余的锁 ====
        prev->lock.unlock();
        cur->lock.unlock();
        // 注意：new_node 不需要解锁，因为还没有其他线程能访问到它
        // （只有通过 prev 才能到达 new_node，而 prev 的锁刚被释放，
        //  但此时插入已经完成，其他线程可以安全地看到新节点）

        return true; // 插入成功
    }

    // ============================================================
    // 删除操作：从排序链表中删除一个值
    // 使用手递手锁定策略进行遍历
    //
    // 返回值：true 表示删除成功，false 表示值不存在
    // ============================================================
    bool remove(int value) {
        // 锁定头节点（遍历的起点）
        head_->lock.lock();

        FGLNode* prev = head_.get();
        FGLNode* cur = prev->next.get();

        // 检查链表是否为空（实际上不会，因为有哨兵尾节点）
        if (cur == nullptr) {
            head_->lock.unlock();
            return false;
        }

        // 锁定第一个真实节点
        cur->lock.lock();

        // ==== 手递手遍历：寻找要删除的节点 ====
        while (cur->value < value) {
            FGLNode* old_prev = prev;
            prev = cur;
            cur = cur->next.get();

            // 到达链表末尾（正常情况下不会发生，因为尾哨兵的值是 INT_MAX）
            if (cur == nullptr) {
                old_prev->lock.unlock();
                prev->lock.unlock();
                return false;
            }

            // 手递手：锁新节点，解旧节点
            cur->lock.lock();
            old_prev->lock.unlock();
        }

        // ==== 找到要删除的节点 ====
        if (cur->value == value) {
            // 从链表中移除 cur 节点：
            //   之前：prev -> cur -> cur->next
            //   之后：prev -> cur->next
            //
            // cur->next.release()：释放 cur 对下一个节点的所有权
            // prev->next.reset(...)：让 prev 接管下一个节点的所有权
            // 这绕过了 cur，使其从链表中脱离
            prev->next.reset(cur->next.release());

            // 释放锁
            prev->lock.unlock();
            cur->lock.unlock();

            // cur 节点现在没有任何 unique_ptr 拥有它
            // （prev 已经不再指向它，cur 是原始指针不拥有所有权）
            // 但由于我们持有 cur->lock，其他线程也无法访问它
            // 当此函数返回后，cur 指向的内存会...
            // 等等，这里有问题：
            // cur 原来是 prev->next 的一部分（由 unique_ptr 管理）
            // reset 操作后 cur 不再被任何 unique_ptr 管理，
            // 会导致内存泄漏或悬空指针？
            //
            // 实际上这里依赖一个事实：
            // cur 原始指针指向的 FGLNode 对象，在被 prev->next.reset()
            // 接管前是由 prev->next 拥有的。当 prev->next.reset() 接管
            // cur->next.release() 返回的指针后，原来指向 cur 的 unique_ptr
            // 被覆盖了，cur 节点会被自动 delete。
            // 此时 cur 原始指针变成悬空指针，但我们在 unlock 后不再使用它。
            return true;
        }

        // ==== 值不存在于链表中 ====
        prev->lock.unlock();
        cur->lock.unlock();
        return false;
    }

    // ============================================================
    // 查找操作：检查值是否存在于链表中
    // 使用手递手锁定策略（虽然是只读操作，但仍需加锁以保证安全）
    //
    // 为什么只读操作也需要加锁？
    //   因为其他线程可能同时在修改链表结构（插入/删除节点）。
    //   如果不加锁，查找操作可能读取到不一致的链表状态
    //   （例如，一个节点正在被删除的过程中）。
    // ============================================================
    bool contains(int value) {
        // 锁定头节点
        head_->lock.lock();

        FGLNode* prev = head_.get();
        FGLNode* cur = prev->next.get();

        // 锁定第一个真实节点（如果存在的话）
        if (cur) cur->lock.lock();

        // ==== 手递手遍历 ====
        // 遍历直到 cur 为空（链表末尾）或 cur->value >= value
        while (cur && cur->value < value) {
            FGLNode* old_prev = prev;
            prev = cur;
            cur = cur->next.get();

            // 如果下一个节点存在，锁定它
            if (cur) cur->lock.lock();

            // 释放旧节点的锁
            old_prev->lock.unlock();
        }

        // 检查是否找到了目标值
        bool found = (cur && cur->value == value);

        // 释放剩余的锁
        prev->lock.unlock();
        if (cur) cur->lock.unlock();

        return found;
    }

    // ============================================================
    // 打印链表内容（非线程安全，仅用于调试）
    //
    // 警告：此方法不获取任何锁，可能在并发环境下
    // 打印出不一致的链表状态。仅用于单线程调试场景。
    // ============================================================
    void print() const {
        FGLNode* cur = head_->next.get();
        std::cout << "链表内容：";
        while (cur) {
            std::cout << cur->value << " ";
            cur = cur->next.get();
        }
        std::cout << std::endl;
    }

private:
    // 链表头指针（哨兵节点）
    // unique_ptr 保证了链表的独占所有权和自动内存管理
    std::unique_ptr<FGLNode> head_;
};

// ============================================================
// 粗粒度链表类（单个全局锁）
// ============================================================
// 作为对比实现：最简单的线程安全链表。
// 所有操作（插入、删除、查找）都由同一个互斥锁保护。
//
// 特点对比：
//   粗粒度锁：
//     + 实现极简，不易出错
//     + 锁开销少（只获取一次锁）
//     - 完全串行化：任何时刻只能有一个线程访问链表
//     - 无法利用多核并行性
//
//   细粒度锁（手递手）：
//     + 允许对不同链表段的并发操作
//     + 在多核系统上可显著提升吞吐量
//     - 实现复杂，容易出错
//     - 锁获取/释放的开销更大（每个节点都要加锁/解锁）
//
// 使用场景选择：
//   - 低竞争或链表很小 → 粗粒度锁（简单可靠）
//   - 高竞争或链表很大 → 细粒度锁（更好的并行性）

class CoarseGrainedList {
public:
    // 公共接口：所有操作都先获取全局锁，然后委托给不安全的内部方法
    // std::lock_guard：RAII 风格的锁管理，自动在析构时释放锁
    bool insert(int value) {
        std::lock_guard<std::mutex> guard(lock_);
        return insert_unsafe(value);
    }

    bool remove(int value) {
        std::lock_guard<std::mutex> guard(lock_);
        return remove_unsafe(value);
    }

    bool contains(int value) {
        std::lock_guard<std::mutex> guard(lock_);
        return contains_unsafe(value);
    }

private:
    // 内部节点结构（不需要 per-node 锁）
    struct Node {
        int value;
        std::unique_ptr<Node> next;
        Node(int v) : value(v), next(nullptr) {}
    };

    // 哨兵头节点（value = -1）
    std::unique_ptr<Node> head_ = std::make_unique<Node>(-1);

    // 全局互斥锁：保护整个链表
    // 任何链表操作都必须先获取此锁
    std::mutex lock_;

    // ============================================================
    // 以下是不安全的内部方法（假定调用者已持有锁）
    // 命名规范：_unsafe 后缀表示"非线程安全，调用者负责加锁"
    // ============================================================

    // 不安全的插入：在排序链表中插入值（已持锁状态下调用）
    bool insert_unsafe(int value) {
        // 从头节点开始遍历，找到插入位置
        Node* prev = head_.get();
        while (prev->next && prev->next->value < value) {
            prev = prev->next.get();
        }

        // 检查重复值
        if (prev->next && prev->next->value == value) return false;

        // 创建新节点并插入
        auto new_node = std::make_unique<Node>(value);
        new_node->next.reset(prev->next.release());
        prev->next.reset(new_node.release());
        return true;
    }

    // 不安全的删除：从链表中删除指定值（已持锁状态下调用）
    bool remove_unsafe(int value) {
        Node* prev = head_.get();
        while (prev->next && prev->next->value < value) {
            prev = prev->next.get();
        }

        if (prev->next && prev->next->value == value) {
            // 绕过要删除的节点：prev->next = prev->next->next
            prev->next.reset(prev->next->next.release());
            return true;
        }
        return false;
    }

    // 不安全的查找：检查值是否存在（已持锁状态下调用）
    bool contains_unsafe(int value) {
        Node* cur = head_->next.get();
        while (cur) {
            if (cur->value == value) return true;
            cur = cur->next.get();
        }
        return false;
    }
};

// ============================================================
// 演示函数：并发插入操作
// ============================================================
// 每个线程从不同的范围插入值，以最小化线程间的竞争，
// 从而展示手递手锁定在不同链表段上的并行插入能力。
//
// 设计思路：
//   线程 0 插入值范围 [0, 999]
//   线程 1 插入值范围 [1000, 1999]
//   线程 2 插入值范围 [2000, 2999]
//   线程 3 插入值范围 [3000, 3999]
//
// 由于各线程操作的值范围完全不重叠，它们操作的链表区域也不同，
// 手递手锁定应该允许它们几乎完全并行执行。
void concurrent_inserts(FineGrainedList& list, int thread_id, int count) {
    // 每个线程使用不重叠的值范围来减少竞争
    int base = thread_id * 1000;
    for (int i = 0; i < count; ++i) {
        list.insert(base + i);
    }
}

// ============================================================
// 演示函数：混合操作（插入→查找→删除）
// ============================================================
// 每个线程对自己的值范围执行完整的生命周期：
//   插入一个值 → 验证它存在 → 删除它
//
// 这测试了手递手锁定在混合读写操作下的正确性。
// 每个操作完成后都使用 assert 验证预期结果。
void concurrent_mixed_ops(FineGrainedList& list, int thread_id, int count) {
    int base = thread_id * 1000;
    for (int i = 0; i < count; ++i) {
        int val = base + i;

        // 步骤 1：插入值
        list.insert(val);

        // 步骤 2：验证插入成功
        bool found = list.contains(val);
        assert(found && "插入后应该能找到该值");

        // 步骤 3：删除值
        bool removed = list.remove(val);
        assert(removed && "应该能够删除该值");
    }
}

int main() {
    std::cout << "=== CS149 第16讲：细粒度锁定（手递手锁定） ===" << std::endl;
    std::cout << std::endl;

    // ---- 演示 1：细粒度并发插入操作 ----
    // 4 个线程同时向链表的不同区域插入值
    // 由于值范围不重叠，手递手锁定应实现高度并行
    std::cout << "--- 演示 1：并发插入（细粒度锁） ---" << std::endl;
    {
        FineGrainedList list;
        const int num_threads = 4;
        const int inserts_per_thread = 500;
        std::vector<std::thread> threads;

        // 启动 4 个并发插入线程
        // 每个线程在值范围 [i*1000, i*1000+500) 内插入 500 个值
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back(concurrent_inserts, std::ref(list), i, inserts_per_thread);
        }
        // 等待所有线程完成
        for (auto& t : threads) t.join();

        std::cout << "已插入 " << (num_threads * inserts_per_thread) << " 个值。" << std::endl;

        // 抽查验证：检查每个线程的第一个和最后一个插入值
        for (int i = 0; i < num_threads; ++i) {
            assert(list.contains(i * 1000) && "每个线程的第一个插入值必须存在");
            assert(list.contains(i * 1000 + inserts_per_thread - 1) &&
                   "每个线程的最后一个插入值必须存在");
        }
        std::cout << "所有抽查验证通过！" << std::endl;
    }

    // ---- 演示 2：细粒度混合操作（插入→查找→删除） ----
    // 测试手递手锁定在读写混合场景下的正确性
    std::cout << std::endl;
    std::cout << "--- 演示 2：混合插入/查找/删除操作（细粒度锁） ---" << std::endl;
    {
        FineGrainedList list;
        const int num_threads = 4;
        const int ops_per_thread = 500;
        std::vector<std::thread> threads;

        // 启动 4 个线程，每个执行 500 组"插入→查找→删除"操作
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back(concurrent_mixed_ops, std::ref(list), i, ops_per_thread);
        }
        for (auto& t : threads) t.join();

        // 验证：所有线程完成后，链表应该恢复为空
        // （每个线程插入的值都被自己删除了）
        for (int i = 0; i < num_threads; ++i) {
            assert(!list.contains(i * 1000) && "所有值应该已经被删除");
        }
        std::cout << "所有值已正确删除。细粒度锁定（手递手锁定）工作正常！" << std::endl;
    }

    // ---- 演示 3：粗粒度链表对比测试 ----
    // 验证粗粒度链表的基本正确性（作为对比基准）
    std::cout << std::endl;
    std::cout << "--- 演示 3：粗粒度链表（正确性验证） ---" << std::endl;
    {
        CoarseGrainedList list;

        // 基本操作测试
        list.insert(5);
        list.insert(3);
        list.insert(7);
        list.insert(1);

        // 验证插入
        assert(list.contains(5) && "5 应该在链表中");
        assert(list.contains(3) && "3 应该在链表中");
        assert(!list.contains(99) && "99 不应该在链表中");

        // 验证删除
        list.remove(3);
        assert(!list.contains(3) && "3 应该已被删除");

        std::cout << "粗粒度链表操作正确。" << std::endl;
    }

    // ============================================================
    // 总结输出
    // ============================================================
    std::cout << std::endl;
    std::cout << "总结：" << std::endl;
    std::cout << "  - 细粒度（手递手）锁定允许对链表的不同部分进行并发操作，" << std::endl;
    std::cout << "    从而在适当的场景下获得比粗粒度锁更好的并行性。" << std::endl;
    std::cout << "  - 通过始终沿链表遍历方向获取锁，避免了死锁问题" << std::endl;
    std::cout << "    （不满足循环等待条件）。" << std::endl;
    std::cout << "  - 权衡：更多锁获取/释放开销 vs. 更好的并行性。" << std::endl;
    std::cout << "    在低竞争场景下，粗粒度锁可能因为开销更低而更快；" << std::endl;
    std::cout << "    在高竞争或大链表场景下，细粒度锁通常胜出。" << std::endl;

    return 0;
}
