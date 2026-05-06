/*
 * lecture16_part4.cpp - 无锁队列（Lock-Free Queues）
 * Stanford CS149, 2025年秋季 - 第16讲
 *
 * 本文件演示讲座中的无锁队列设计：
 *   1. 单读者、单写者有界队列（Single Reader, Single Writer Bounded Queue）
 *      基于环形缓冲区（ring buffer）实现
 *   2. 单读者、单写者无界队列（Single Reader, Single Writer Unbounded Queue）
 *      基于链表（linked-list）实现
 *
 * === 核心洞察：为什么这些队列是「无锁」的？ ===
 *
 * 关键在于「单生产者、单消费者（SPSC）」的设计约束：
 *   - head指针只由消费者（consumer）修改
 *   - tail指针只由生产者（producer）修改
 *   - 两个线程各自独占一个指针，不存在同时写入同一变量的竞争
 *   - 因此不需要CAS操作，甚至不需要锁！
 *
 * 这与多生产者/多消费者队列形成对比：
 *   - 多生产者需要CAS来竞争tail的写入权（如Michael-Scott队列）
 *   - 多消费者需要CAS来竞争head的写入权
 *
 * SPSC设计的优势：
 *   - 线程间永不互相阻塞（lock-free的无等待特性）
 *   - 内存开销最小（没有锁、没有CAS失败重试）
 *   - 在流水线（pipeline）架构中非常实用：
 *     线程A产出数据 → 队列 → 线程B消费数据，每个环节都是单线程
 *
 * === 内存顺序与可见性保证 ===
 *
 * 前提假设：顺序一致的内存系统，或者在C++11中使用恰当的
 * 内存顺序（memory order）和原子操作。
 *
 * 在C++11的std::atomic模型中：
 *   - push使用release语义：确保写入的数据在tail更新之前对其他线程可见
 *   - pop使用acquire语义：确保读取tail之前能看到生产者写入的所有数据
 *   - release/acquire配对保证了happens-before关系
 *
 * === 有界vs无界队列的设计权衡 ===
 *
 * 有界队列（环形缓冲区）：
 *   优点：无动态内存分配，缓存友好，性能最高
 *   缺点：容量固定，满时需要背压（back-pressure）处理
 *   适用场景：数据速率可预测，内存受限的嵌入式/实时系统
 *
 * 无界队列（链表）：
 *   优点：容量按需增长，没有背压问题
 *   缺点：动态内存分配开销，缓存局部性差，内存管理复杂
 *   适用场景：数据速率不可预测，允许偶尔的延迟峰值
 *
 * 编译命令: g++ -std=c++17 -pthread lecture16_part4.cpp -o lecture16_part4
 * 运行命令: ./lecture16_part4
 */

#include <iostream>
#include <thread>
#include <atomic>
#include <vector>
#include <cassert>
#include <memory>
#include <chrono>

// ============================================================
// 第一部分：单读者单写者有界队列（Bounded SPSC Queue）
// ============================================================
//
// 基于固定大小的环形缓冲区（ring buffer / circular buffer）。
// 生产者向tail位置写入数据，消费者从head位置读取数据。
//
// 关键设计：为什么不需要锁？
//   1. head只由消费者写入（pop时后移）
//   2. tail只由生产者写入（push时后移）
//   3. 生产者读取head仅用于判断队列是否已满（只读，不写入）
//   4. 消费者读取tail仅用于判断队列是否为空（只读，不写入）
//   5. 没有两个线程同时写入同一个变量 → 不需要锁！
//
// 满判断条件（Full Condition）：
//   有意浪费一个槽位（slot）来区分「满」和「空」两种状态。
//   公式：tail == MOD_N(head - 1)
//   即：tail的下一个位置是head → 队列已满
//   如果不浪费这个槽位，满(tail==head)和空(tail==head)将无法区分。
//
// 空判断条件（Empty Condition）：
//   head == tail → 队列为空（没有任何数据可读）
//
// 内存布局优化：使用alignas(64)将head和tail对齐到不同的
// 缓存行（cache line），避免「伪共享」（false sharing）问题：
//   如果head和tail在同一缓存行，一个线程写head会导致另一个线程
//   的缓存行失效，产生不必要的缓存一致性流量。

template <typename T, size_t N>
class BoundedSPSCQueue {
public:
    BoundedSPSCQueue() : head_(0), tail_(0) {}

    // 生产者push操作：向环形缓冲区的tail位置写入数据
    //
    // 执行流程：
    //   1. 读取当前tail位置
    //   2. 计算下一个tail（环形前进一位）
    //   3. 检查是否与head冲突（==满）
    //   4. 将数据写入data_[current_tail]
    //   5. 更新tail指针（release语义确保数据先于指针可见）
    //
    // 返回值：成功返回true，队列满时返回false
    //
    bool push(T value) {
        size_t current_tail = tail_.load(std::memory_order_relaxed);
        size_t next_tail = (current_tail + 1) % N;

        // 如果next_tail追上了head，说明队列已满
        // 使用acquire语义读取head，确保能看到消费者之前的所有更新
        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false; // 队列已满，push失败
        }

        // 将数据写入当前tail位置，然后原子地推进tail
        // release语义：确保data_[current_tail]的写入
        // 在tail_.store之前对所有线程可见
        data_[current_tail] = value;
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }

    // 消费者pop操作：从环形缓冲区的head位置读取数据
    //
    // 执行流程：
    //   1. 读取当前head位置
    //   2. 检查是否与tail相等（==空）
    //   3. 从data_[current_head]读取数据
    //   4. 更新head指针（release语义）
    //
    // 返回值：成功返回true并将值写入引用参数value，队列空时返回false
    //
    bool pop(T& value) {
        size_t current_head = head_.load(std::memory_order_relaxed);

        // 如果head追上了tail，说明队列为空
        // 使用acquire语义确保能看到生产者的所有写入
        if (current_head == tail_.load(std::memory_order_acquire)) {
            return false; // 队列为空，pop失败
        }

        value = data_[current_head];
        // release语义：确保消费者的读取完成后再推进head
        head_.store((current_head + 1) % N, std::memory_order_release);
        return true;
    }

private:
    T data_[N];  // 环形缓冲区数据数组（固定容量N）

    // 将head和tail对齐到不同的缓存行（64字节对齐），避免伪共享
    // 伪共享（False Sharing）：
    //   如果两个原子变量在同一缓存行中，一个线程写head会导致
    //   另一个线程的整个缓存行失效，即使它们访问的是不同变量。
    //   alignas(64)将它们强制放在不同缓存行的起始位置。
    alignas(64) std::atomic<size_t> head_;  // 消费者独占的读取位置
    alignas(64) std::atomic<size_t> tail_;  // 生产者独占的写入位置
};

// ============================================================
// 第二部分：单读者单写者无界队列（Unbounded SPSC Queue）
// ============================================================
//
// 基于链表的无界队列实现（源自Dr. Dobbs Journal的经典设计）。
//
// 核心设计理念：
//   head —— 指向队列「第一个真实节点之前」的哨兵节点（sentinel）
//   tail —— 指向队列「最后一个真实节点」
//
// 为什么head指向哨兵而不是第一个真实节点？
//   这是简化pop操作的巧妙设计：
//   - pop时不需要操作被删除节点的前驱
//   - 从head->next直接读取数据
//   - 然后将head推进到head->next
//   - 旧节点由生产者在push中负责回收
//
// 内存管理的关键分配：
//   - 所有节点的分配（new）由生产者线程完成
//   - 所有节点的回收（delete）也由生产者线程完成
//   - 消费者只移动head指针，不参与任何内存分配/释放
//
// 这种设计如何避免ABA问题？
//   - 消费者只读head指针，不写head（这里不适用CAS）
//   - 更重要的是：生产者在回收节点时，reclaim_指针追踪的是
//     消费者已经「越过」的节点（reclaim_在head之前），
//     这些节点消费者不会再访问 → 不会出现ABA问题
//   - 如果有多消费者，head的CAS竞争可能引发ABA，
//     但这个SPSC设计只有单消费者，避免了该问题
//
// 节点回收机制（reclaim）：
//   生产者在push新节点时，顺便回收已经被消费者消费掉的节点。
//   reclaim_指针从head开始追逐head：当reclaim_落后于head时，
//   说明reclaim_和head之间的节点已被消费者弹出，可以安全释放。
//   这个设计将内存管理工作集中到一个线程，简化了并发控制。

template <typename T>
class UnboundedSPSCQueue {
private:
    struct Node {
        T value;     // 节点存储的值
        Node* next;  // 链表指针，指向下一个节点
        Node() : next(nullptr) {}        // 哨兵节点构造函数
        Node(T v) : value(v), next(nullptr) {}  // 数据节点构造函数
    };

public:
    UnboundedSPSCQueue() {
        // 初始化：创建一个哨兵节点，head、tail、reclaim都指向它
        // 哨兵节点的作用：
        //   1. 作为一个「占位符」，简化边界条件处理
        //   2. pop时总是从head->next取数据，不需要特殊处理空队列
        //      （空队列时head->next == nullptr）
        Node* sentinel = new Node();
        head_ = sentinel;
        tail_ = sentinel;
        reclaim_ = sentinel;  // 回收指针从哨兵节点开始
    }

    // 析构函数：清理所有残留节点
    // 注意：生产者和消费者线程应该在此之前已经停止
    ~UnboundedSPSCQueue() {
        // 从reclaim_开始遍历链表，释放所有节点
        while (reclaim_ != nullptr) {
            Node* tmp = reclaim_;
            reclaim_ = reclaim_->next;
            delete tmp;
        }
    }

    // 生产者push操作：向队列尾部添加新值
    //
    // 步骤详解：
    //   1. 生产者分配新节点
    //   2. 将新节点链接到tail之后（tail_->next = n）
    //   3. 推进tail到新节点（tail_ = n）
    //   4. 顺便回收已被消费者消费的节点（reclaim循环）
    //
    // 回收逻辑说明：
    //   reclaim_追赶head_，释放被跨越的节点。
    //   为什么安全？因为head只被消费者推进，head之后的节点
    //   消费者可能还会访问，但head之前的节点（包括哨兵）
    //   消费者已经不会再访问了。
    //
    // 注意：这里tail_->next和tail_的写入顺序很重要。
    // 必须先设置next再更新tail，否则消费者可能看到一个
    // 不完整的链表。
    //
    void push(T value) {
        Node* n = new Node(value); // 生产者负责分配新节点
        n->next = nullptr;

        // 第一步：将新节点链接到链表末尾
        // tail_->next = n 必须在 tail_ = n 之前执行
        // 否则消费者看到新tail时可能发现next是nullptr（未初始化的状态）
        tail_->next = n;
        tail_ = n;

        // 第二步：回收已被消费者消费的节点
        // reclaim_在head之前（或等于head）的所有节点
        // 都是消费者已经弹出且不会再访问的节点，可以安全释放
        // 这个设计确保所有内存分配和释放都在生产者线程中完成
        while (reclaim_ != head_) {
            Node* tmp = reclaim_;
            reclaim_ = tmp->next;
            delete tmp;  // 生产者负责释放内存
        }
    }

    // 消费者pop操作：从队列头部取出值
    //
    // 步骤详解：
    //   1. 读取head指向的哨兵节点
    //   2. 检查head->next是否为空（空队列判断）
    //   3. 从head->next读取数据（因为head永远是哨兵）
    //   4. 推进head到head->next（旧的哨兵变为废弃节点）
    //   5. 注意：这里不delete旧head，内存回收由生产者在push中处理
    //
    // 为什么消费者不释放节点？
    //   如果消费者delete了head，生产者push中的reclaim循环
    //   可能访问到已释放的内存。将回收职责集中在生产者，
    //   避免了并发内存管理的复杂性。
    //
    bool pop(T& value) {
        Node* current_head = head_;
        Node* next = current_head->next;

        // 如果head->next为空：
        //   说明head和tail指向同一个节点（哨兵节点），
        //   队列中没有任何真实数据节点，队列为空
        if (next == nullptr) {
            return false;
        }

        // 从head的下一个节点读取数据（head永远是哨兵节点）
        value = next->value;
        head_ = next;  // 推进head，旧的哨兵节点现在可以被回收了

        // 注意：我们在这里不delete旧的head节点。
        // 生产者线程在push()中通过reclaim循环负责回收。
        // 这种分工避免了ABA问题和双重释放（double-free）的风险。
        return true;
    }

private:
    // 三个指针的职责划分：
    //
    // head_:   永远指向一个哨兵节点（sentinel node），
    //          该节点位于第一个真实数据节点的前面。
    //          只由消费者线程修改（pop时推进）。
    //
    // tail_:   指向链表中最后一个真实数据节点。
    //          只由生产者线程修改（push时推进）。
    //
    // reclaim_: 生产者用来追踪哪些节点可以安全释放。
    //           reclaim_追赶head_，释放head之前的所有节点。
    //           只由生产者线程修改（push时推进）。
    //
    // 这种「三指针」设计将并发操作降到最低：
    //   - 消费者只写head_
    //   - 生产者只写tail_和reclaim_
    //   - 没有任何两个线程同时写同一个指针
    //
    Node* head_;      // 只由消费者修改
    Node* tail_;      // 只由生产者修改
    Node* reclaim_;   // 只由生产者修改（内存回收追踪器）
};

// ============================================================
// 演示部分（Demonstration）
// ============================================================
//
// 测试设计说明：
//   - 使用100000个元素来展示无锁队列的性能特征
//   - 测量每次传输的耗时，用于对比有界和无界队列的性能差距
//   - 正确性测试验证FIFO（先进先出）的顺序语义

// 有界队列的生产者函数
// 注意：由于队列容量只有16，生产者需要自旋等待空位
// 在生产环境中应该用条件变量或信号量替代忙等
void producer_bounded(BoundedSPSCQueue<int, 16>& q, int count) {
    int pushed = 0;
    while (pushed < count) {
        if (q.push(pushed)) {
            ++pushed;
        }
        // 如果队列满了，生产者自旋等待（spin-wait）
        // 在生产代码中这会浪费CPU，但在SPSC场景下通常是短等待
    }
    std::cout << "有界队列生产者完成：压入了 " << pushed << " 个元素。" << std::endl;
}

// 有界队列的消费者函数
// 计算所有弹出值的总和，用于验证数据完整性
void consumer_bounded(BoundedSPSCQueue<int, 16>& q, int count) {
    int popped = 0;
    int sum = 0;  // 累加和用于数据完整性校验
    while (popped < count) {
        int val;
        if (q.pop(val)) {
            sum += val;
            ++popped;
        }
    }
    std::cout << "有界队列消费者完成：弹出了 " << popped << " 个元素，累加和=" << sum << std::endl;
}

// 无界队列的生产者函数
// 无界队列不需要检查满条件，实现更简单
void producer_unbounded(UnboundedSPSCQueue<int>& q, int count) {
    for (int i = 0; i < count; ++i) {
        q.push(i);
    }
    std::cout << "无界队列生产者完成：压入了 " << count << " 个元素。" << std::endl;
}

// 无界队列的消费者函数
void consumer_unbounded(UnboundedSPSCQueue<int>& q, int count) {
    int popped = 0;
    int sum = 0;
    while (popped < count) {
        int val;
        if (q.pop(val)) {
            sum += val;
            ++popped;
        }
    }
    std::cout << "无界队列消费者完成：弹出了 " << popped << " 个元素，累加和=" << sum << std::endl;
}

int main() {
    std::cout << "=== CS149 第16讲：无锁队列 ===" << std::endl;
    std::cout << std::endl;

    const int num_items = 100000;  // 传输测试的元素数量

    // ---- 演示1：有界SPSC队列 ----
    std::cout << "--- 演示1：有界SPSC队列（环形缓冲区，容量=16） ---" << std::endl;
    std::cout << "  说明：固定大小环形缓冲区。生产者可能需要等待空位。" << std::endl;
    {
        BoundedSPSCQueue<int, 16> q;
        auto start = std::chrono::high_resolution_clock::now();

        std::thread producer(producer_bounded, std::ref(q), num_items);
        std::thread consumer(consumer_bounded, std::ref(q), num_items);
        producer.join();
        consumer.join();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "耗时: " << duration.count() << " 毫秒" << std::endl;
    }

    // ---- 演示2：无界SPSC队列 ----
    std::cout << std::endl;
    std::cout << "--- 演示2：无界SPSC队列（基于链表） ---" << std::endl;
    std::cout << "  说明：动态分配节点，无容量限制，不需要背压处理。" << std::endl;
    {
        UnboundedSPSCQueue<int> q;
        auto start = std::chrono::high_resolution_clock::now();

        std::thread producer(producer_unbounded, std::ref(q), num_items);
        std::thread consumer(consumer_unbounded, std::ref(q), num_items);
        producer.join();
        consumer.join();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "耗时: " << duration.count() << " 毫秒" << std::endl;
    }

    // ---- 演示3：正确性验证 ----
    std::cout << std::endl;
    std::cout << "--- 演示3：FIFO顺序正确性验证 ---" << std::endl;
    std::cout << "  验证FIFO（先进先出）的顺序语义。" << std::endl;
    {
        BoundedSPSCQueue<int, 8> q;
        // 依次压入1, 2, 3, 4, 5
        for (int i = 1; i <= 5; ++i) {
            assert(q.push(i) && "push应该成功（队列有足够容量）");
        }
        std::cout << "  压入: 1, 2, 3, 4, 5" << std::endl;

        // 按FIFO顺序依次弹出，验证顺序正确性
        for (int expected = 1; expected <= 5; ++expected) {
            int val;
            assert(q.pop(val) && "pop应该成功（队列非空）");
            assert(val == expected && "FIFO顺序验证失败：期望值与实际值不匹配");
        }
        std::cout << "  弹出顺序: 1, 2, 3, 4, 5 （FIFO正确）" << std::endl;

        // 验证队列现在应该为空
        int dummy;
        assert(!q.pop(dummy) && "队列应该为空，pop应该失败");
        std::cout << "  有界队列FIFO顺序验证通过。✓" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "总结：本讲核心知识点" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "  1. SPSC（单生产者单消费者）队列无需锁或CAS操作。" << std::endl;
    std::cout << "     - 原因：head只由消费者写，tail只由生产者写，" << std::endl;
    std::cout << "       没有两个线程同时写入同一个变量" << std::endl;
    std::cout << "     - 生产者读取head仅用于「满」判断（只读）" << std::endl;
    std::cout << "     - 消费者读取tail仅用于「空」判断（只读）" << std::endl;
    std::cout << "     - 但需要正确的内存顺序（release/acquire配对）" << std::endl;
    std::cout << "       来保证跨线程的数据可见性" << std::endl;
    std::cout << std::endl;
    std::cout << "  2. 有界队列（环形缓冲区）：" << std::endl;
    std::cout << "     - 固定大小的环形缓冲区，无动态内存分配" << std::endl;
    std::cout << "     - 需要内存屏障（memory fences）保证可见性" << std::endl;
    std::cout << "     - 故意浪费一个槽位来区分「满」和「空」" << std::endl;
    std::cout << "     - 使用alignas(64)避免伪共享（false sharing）" << std::endl;
    std::cout << "     - 优势：缓存友好，适合高频小数据量传输" << std::endl;
    std::cout << std::endl;
    std::cout << "  3. 无界队列（链表实现）：" << std::endl;
    std::cout << "     - 动态分配节点，容量可以无限增长" << std::endl;
    std::cout << "     - 使用哨兵节点（sentinel）简化代码逻辑" << std::endl;
    std::cout << "     - 生产者负责所有内存管理（分配+回收）" << std::endl;
    std::cout << "       以此避免ABA问题和use-after-free问题" << std::endl;
    std::cout << "     - 劣势：动态分配开销、缓存局部性差" << std::endl;
    std::cout << std::endl;
    std::cout << "  4. 对于多生产者/多消费者场景，请参考Michael-Scott队列：" << std::endl;
    std::cout << "     - 使用两个独立的锁（head锁和tail锁）" << std::endl;
    std::cout << "     - 或使用CAS竞争操作head和tail指针" << std::endl;
    std::cout << "     - 是无锁并发数据结构领域的经典实现" << std::endl;
    std::cout << "     - 在java.util.concurrent中作为ConcurrentLinkedQueue的核心" << std::endl;

    return 0;
}
