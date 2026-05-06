/*
 * lecture16_part3.cpp - 无锁栈与ABA问题（Lock-Free Stack with ABA Problem）
 * Stanford CS149, 2025年秋季 - 第16讲
 *
 * 本文件演示无锁数据结构中的核心概念：
 *   1. 使用CAS（Compare-And-Swap，比较并交换）操作顶部指针的简单无锁栈
 *   2. ABA问题：为什么仅靠CAS不足以保证正确性
 *   3. ABA问题的解决方案：使用计数器（pop_count）+ 双宽CAS（Double-Width CAS, DWCAS）
 *
 * === ABA问题详解（回顾讲座内容） ===
 *
 * 什么是否锁（Lock-Free）数据结构？
 *   - 锁-Free意味着系统中至少有一个线程能保证在有限步数内完成操作，
 *     不会因为其他线程的挂起或崩溃而导致整个系统停滞。
 *   - 与「无等待（Wait-Free）」的区别：Wait-Free要求每个线程都能在有限步数内完成，
 *     而Lock-Free允许某些线程因重试而延迟，但至少有一个线程在推进。
 *
 * CAS（比较并交换）原语：
 *   - CAS(ptr, expected, desired)：如果ptr指向的值等于expected，
 *     则原子地将ptr指向的值改为desired并返回true；否则返回false。
 *   - 这是构建无锁数据结构的基石，由CPU硬件指令（如x86的cmpxchg）直接支持。
 *
 * ABA问题的产生过程（经典场景）：
 *   假设栈当前状态为：top -> A -> B -> C
 *
 *   步骤1: 线程0读取top=A，计算new_top = A->next = B，
 *          准备执行CAS(&top, A, B)。但此时线程0被抢占。
 *
 *   步骤2: 线程1执行 pop(A)：将A弹出，top变为 B -> C
 *          线程1执行 pop(B)：将B弹出，top变为 C
 *          线程1执行 push(A)：将A重新压入，top变为 A -> C
 *          （注意：此时A被删除后又重新分配，或从空闲链表复用，
 *           其地址可能与原来相同，但next指针已指向C而非B！）
 *
 *   步骤3: 线程0恢复执行，执行CAS(&top, A, B)。
 *          此时top确实等于A（CAS的期望值匹配），
 *          所以CAS成功将top设为B —— 但B已经被线程1弹出并删除了！
 *          现在top指向已释放/无效的内存区域，数据结构被破坏。
 *
 * ABA问题的本质：
 *   - CAS只检查「值是否相等」，不检查「值在中间是否被修改过」。
 *   - 在并发环境下，指针的值（内存地址）可能被回收并重新使用，
 *     导致CAS误判为「没有变化」。
 *   - 这个问题不仅出现在栈中，任何依赖CAS的链表结构都可能受影响。
 *
 * === 解决方案 ===
 *
 * 方案1：双宽CAS（Double-Width CAS, DWCAS）—— 本文件演示的方案
 *   - 在指针旁边附加一个版本号/计数器（pop_count），
 *     每次修改时同时递增计数器。
 *   - 使用128位CAS（x86的cmpxchg16b指令）原子地比较和更新
 *     指针+计数器的组合值。
 *   - 即使地址被复用，计数器不同也会让CAS失败，从而识别ABA。
 *   - 需要编译选项：-mcx16 或 -march=native
 *
 * 方案2：风险指针（Hazard Pointers）—— 讲座中讨论的方案
 *   - 线程在访问指针前将其发布到「风险指针」列表中。
 *   - 内存回收线程在释放内存前检查该地址是否在任何线程的风险指针中。
 *   - 这样可以从根本上防止「访问已释放内存」的问题，
 *     但并不直接防止CAS错误成功（仍需要计数器配合）。
 *
 * 编译命令: g++ -std=c++17 -pthread lecture16_part3.cpp -o lecture16_part3
 * 运行命令: ./lecture16_part3
 */

#include <iostream>
#include <thread>
#include <atomic>
#include <vector>
#include <cassert>
#include <cstdint>
#include <memory>
#include <mutex>

// ============================================================
// 第一部分：简单无锁栈（存在ABA漏洞）
// ============================================================
// 使用原子CAS操作栈顶指针。此实现在单生产者场景下正确，
// 但在多线程pop场景下存在ABA漏洞。
//
// 核心思路：乐观地（speculatively）计算新的top，
// 然后通过CAS原子地安装。如果CAS失败（说明其他线程修改了top），则重试。
// 这种「尝试-失败-重试」模式是无锁编程的标准范式。

struct LFNode {
    int value;      // 节点存储的值
    LFNode* next;   // 指向下一个节点的指针（链表结构）
    LFNode(int v) : value(v), next(nullptr) {}
};

class SimpleLockFreeStack {
public:
    SimpleLockFreeStack() : top_(nullptr) {}

    // Push操作：使用CAS循环原子地更新top指针
    // 工作流程：
    //   1. 创建新节点new_node
    //   2. 读取当前top
    //   3. 让new_node->next指向当前top
    //   4. 尝试CAS：如果top仍然等于old_top，就将top更新为new_node
    //   5. 如果CAS失败，说明有其他线程修改了top，回到步骤2重试
    void push(int value) {
        LFNode* n = new LFNode(value);
        while (true) {
            LFNode* old_top = top_.load(std::memory_order_relaxed);
            n->next = old_top;
            // 原子操作：如果top == old_top，则设置top = n
            // compare_exchange_weak可能因spurious failure而失败，
            // 但放在循环中使用是正确的（比strong版本在某些架构上更高效）
            if (top_.compare_exchange_weak(old_top, n,
                    std::memory_order_release, std::memory_order_relaxed)) {
                return; // 成功！新节点已原子地成为新的栈顶
            }
            // CAS失败：另一个线程修改了top。此时old_top
            // 已被compare_exchange_weak更新为当前top值。继续循环重试。
        }
    }

    // Pop操作：使用CAS循环 —— 但警告：存在ABA漏洞！
    //
    // ABA漏洞具体场景：
    //   假设当前栈为 top -> X -> Y -> Z
    //   1. 线程A读取top=X（第69行），然后被抢占
    //   2. 线程B执行pop(X)，删除X；执行pop(Y)，删除Y；
    //      执行push(X)，将X重新压入栈顶（X可能被分配到相同地址）。
    //      此时栈为 top -> X -> Z
    //   3. 线程A恢复，执行CAS(&top, X, X->next=Y)：
    //      top确实是X（CAS成功！），但X->next已经被修改或Y已被释放！
    //      结果：top指向了无效内存。
    //
    // 这就是为什么这个pop实现在多线程pop场景下不安全。
    int pop() {
        while (true) {
            LFNode* old_top = top_.load(std::memory_order_acquire);
            if (old_top == nullptr) {
                return -1; // 栈为空，返回哨兵值
            }
            LFNode* new_top = old_top->next;
            // ABA风险：在读取old_top和CAS之间，old_top可能被另一个线程
            // pop删除、释放内存、然后又被重新push回来。
            // CAS只检查地址是否相等，无法检测这种「中间被修改过」的情况。
            if (top_.compare_exchange_weak(old_top, new_top,
                    std::memory_order_release, std::memory_order_relaxed)) {
                int val = old_top->value;
                delete old_top; // 内存回收 —— 在ABA场景下这也可能不安全
                                 // （因为old_top可能在CAS之前就被其他线程删除了）
                return val;
            }
        }
    }

    bool empty() const {
        return top_.load(std::memory_order_relaxed) == nullptr;
    }

private:
    std::atomic<LFNode*> top_;  // 原子栈顶指针
};

// ============================================================
// 第二部分：使用Mutex降级方案实现ABA安全栈
// ============================================================
//
// 讲座中的理想方案：在top旁边维护pop_count计数器，
// 使用128位CAS（双宽CAS）原子地更新top和counter两个值。
// 如果pop_count发生了变化，即使top匹配，CAS也会失败 —— 
// 这样就能防止ABA问题。
//
// 关于128位CAS的说明：
//   在x86架构上，cmpxchg16b指令提供了16字节（128位）的CAS操作。
//   需要std::atomic对128位类型的支持，以及编译选项-mcx16或-march=native
//   来启用编译器的__atomic_load_16 / __atomic_compare_exchange_16内建函数。
//   如果不加这些编译选项，128位std::atomic操作将无法链接。
//
// 降级方案说明：
//   由于128位原子操作的编译依赖性，这里使用互斥锁（mutex）
//   来同时保护top和pop_count。这不是真正的无锁实现，
//   但保留了算法的结构以及用于追踪ABA历史的pop_count计数器。
//   在生产环境中，真正的无锁实现应使用128位CAS。
//
// pop_count的作用（在真正的128位CAS实现中）：
//   - 每次pop时递增计数器
//   - CAS同时检查(top, pop_count)元组是否与期望值匹配
//   - 即使top地址被复用（ABA），pop_count的不同也会导致CAS失败
//   - 这是对抗ABA的关键机制

class ABASafeLockFreeStack {
public:
    ABASafeLockFreeStack() : top_(nullptr), pop_count_(0) {}

    // Push操作：mutex保护，仅更新top（push时不改变counter）
    void push(int value) {
        LFNode* n = new LFNode(value);
        std::lock_guard<std::mutex> lock(mutex_);
        n->next = top_;
        top_ = n;
    }

    // Pop操作：mutex保护，原子地同时更新top和counter。
    //
    // 计数器的作用：即使在基于mutex的实现中counter并非严格必要
    // （mutex已经防止了竞争），但保留counter可以展示在真正的无锁实现中，
    // pop_count如何作为「版本号」来防御ABA场景。
    //
    // 在真正的无锁DWCAS实现中，pop_count的工作原理：
    //   1. 线程读取(old_top, old_count)元组
    //   2. 准备new_top = old_top->next, new_count = old_count + 1
    //   3. 执行128位CAS：如果(top, count) == (old_top, old_count)，
    //      则原子地设置为(new_top, new_count)
    //   4. 如果CAS失败：
    //      - 要么top变了（其他线程push/pop）
    //      - 要么count变了（其他线程pop了——即使top地址没变）
    //   5. 这两种情况都会触发重试，从而避免ABA
    //
    int pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (top_ == nullptr) {
            return -1; // 栈为空
        }
        LFNode* old_top = top_;
        top_ = old_top->next;
        ++pop_count_;           // 每次pop递增计数器 —— 版本号的物理意义
        int val = old_top->value;
        delete old_top;
        return val;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return top_ == nullptr;
    }

private:
    LFNode* top_;               // 栈顶指针
    uint64_t pop_count_;        // pop计数器（版本号），用于检测ABA
    mutable std::mutex mutex_;  // 互斥锁（真正的无锁实现会用128位原子代替）
};

// ============================================================
// 第三部分：演示 —— ABA问题场景
// ============================================================
//
// 这个演示展示了ABA问题可能发生的场景。
// 在实际的受控测试中，简单栈在大多数运行中不会出现问题，
// 因为我们的测试时间和线程调度不一定会产生触发ABA所需的精确时序。
// 但在负载高、线程多、运行时间长的生产环境中，ABA问题会以
// 极低的概率出现，导致难以调试的数据损坏。
//
// 基于计数器的方案在数学上是可以证明免疫ABA的：
//   每次pop递增计数器 → 即使地址被复用，计数器的不同也会导致CAS失败。
//
// 测试设计说明：
//   - 2个生产者/2个消费者模拟适度的并发场景
//   - 每个生产者push 10000个元素，足以展示并发行为
//   - 先启动消费者再启动生产者，模拟实际场景中消费者在等待数据到达
//   - 消费者在栈为空时主动yield，避免忙等浪费CPU

void producer(SimpleLockFreeStack& stack, int thread_id, int items) {
    for (int i = 0; i < items; ++i) {
        stack.push(thread_id * 1000 + i);  // 为每个线程生成唯一的值范围
    }
}

void consumer(SimpleLockFreeStack& stack, std::atomic<int>& total_popped, int items) {
    int popped = 0;
    while (popped < items) {
        int val = stack.pop();
        if (val != -1) { // 成功弹出
            ++popped;
        }
        // 如果栈为空，让出CPU给生产者（避免忙等待）
        if (val == -1) {
            std::this_thread::yield();
        }
    }
    total_popped.fetch_add(popped, std::memory_order_relaxed);
}

void producer_aba_safe(ABASafeLockFreeStack& stack, int thread_id, int items) {
    for (int i = 0; i < items; ++i) {
        stack.push(thread_id * 1000 + i);
    }
}

void consumer_aba_safe(ABASafeLockFreeStack& stack, std::atomic<int>& total_popped, int items) {
    int popped = 0;
    while (popped < items) {
        int val = stack.pop();
        if (val != -1) {
            ++popped;
        }
        if (val == -1) {
            std::this_thread::yield();
        }
    }
    total_popped.fetch_add(popped, std::memory_order_relaxed);
}

int main() {
    std::cout << "=== CS149 第16讲：无锁栈与ABA问题 ===" << std::endl;
    std::cout << std::endl;

    const int num_producers = 2;      // 生产者线程数
    const int num_consumers = 2;      // 消费者线程数
    const int items_per_producer = 10000;  // 每个生产者push的元素数量

    // ---- 演示1：简单无锁栈（存在ABA漏洞） ----
    std::cout << "--- 演示1：简单无锁栈（ABA漏洞版本） ---" << std::endl;
    std::cout << "  说明：此实现在底层使用CAS操作，无锁但在pop时存在ABA风险。" << std::endl;
    {
        SimpleLockFreeStack stack;
        std::atomic<int> total_popped{0};
        std::vector<std::thread> threads;

        // 先启动消费者（它们会在栈为空时自旋等待）
        for (int i = 0; i < num_consumers; ++i) {
            threads.emplace_back(consumer, std::ref(stack),
                                 std::ref(total_popped), items_per_producer);
        }
        for (int i = 0; i < num_producers; ++i) {
            threads.emplace_back(producer, std::ref(stack), i, items_per_producer);
        }
        for (auto& t : threads) t.join();  // 等待所有线程完成

        std::cout << "压入数量: " << (num_producers * items_per_producer)
                  << ", 弹出数量: " << total_popped.load()
                  << ", 栈是否为空: " << (stack.empty() ? "是" : "否（数据丢失！）") << std::endl;
        std::cout << "  （注意：ABA风险存在但可能在短时间运行中不会显现，" << std::endl;
        std::cout << "   在长时间、高并发的生产环境中可能以低概率出现数据损坏。）" << std::endl;
    }

    // ---- 演示2：ABA安全栈（基于计数器+mutex） ----
    std::cout << std::endl;
    std::cout << "--- 演示2：ABA安全栈（带pop_count计数器的版本） ---" << std::endl;
    std::cout << "  说明：使用pop_count计数器追踪每次pop操作。" << std::endl;
    std::cout << "  当前实现使用mutex保证原子性，真正的无锁版本需要128位CAS。" << std::endl;
    {
        ABASafeLockFreeStack stack;
        std::atomic<int> total_popped{0};
        std::vector<std::thread> threads;

        for (int i = 0; i < num_consumers; ++i) {
            threads.emplace_back(consumer_aba_safe, std::ref(stack),
                                 std::ref(total_popped), items_per_producer);
        }
        for (int i = 0; i < num_producers; ++i) {
            threads.emplace_back(producer_aba_safe, std::ref(stack), i, items_per_producer);
        }
        for (auto& t : threads) t.join();

        std::cout << "压入数量: " << (num_producers * items_per_producer)
                  << ", 弹出数量: " << total_popped.load()
                  << ", 栈是否为空: " << (stack.empty() ? "是" : "否（数据丢失！）") << std::endl;
        std::cout << "  基于计数器的CAS可以防止ABA问题 —— 数学上可证明其正确性！" << std::endl;
        std::cout << "  pop_count作用：即使top地址被复用，不同的计数器值也会导致CAS失败。" << std::endl;
    }

    // ---- 演示3：单线程顺序正确性验证 ----
    std::cout << std::endl;
    std::cout << "--- 演示3：单线程顺序正确性验证 ---" << std::endl;
    std::cout << "  验证LIFO（后进先出）的栈语义。" << std::endl;
    {
        ABASafeLockFreeStack stack;
        stack.push(10);
        stack.push(20);
        stack.push(30);

        int v1 = stack.pop();
        int v2 = stack.pop();
        int v3 = stack.pop();

        std::cout << "弹出顺序: " << v1 << ", " << v2 << ", " << v3
                  << " （期望: 30, 20, 10 —— LIFO后进先出顺序）" << std::endl;

        // assert用于验证LIFO语义和栈为空的条件
        assert(v1 == 30 && "LIFO验证失败：最后压入的应该最先弹出");
        assert(v2 == 20 && "LIFO验证失败：第二个弹出的应为20");
        assert(v3 == 10 && "LIFO验证失败：第一个压入的最后弹出");
        assert(stack.empty() && "栈应该为空");
        std::cout << "单线程LIFO顺序验证通过。✓" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "总结：本讲核心知识点" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "  1. 无锁栈使用CAS重试循环代替锁（mutex），避免线程阻塞。" << std::endl;
    std::cout << "     - CAS循环模式：读取 → 计算新值 → CAS尝试 → 失败则重试" << std::endl;
    std::cout << "     - 优势：不会发生死锁，优先级反转问题较轻" << std::endl;
    std::cout << "     - 代价：忙等（busy-waiting）浪费CPU，CAS失败率高时性能下降" << std::endl;
    std::cout << std::endl;
    std::cout << "  2. ABA问题：top指针经历 A→B→A 的变化后，CAS检测不到中间的变化，" << std::endl;
    std::cout << "     CAS成功但数据结构已被破坏。" << std::endl;
    std::cout << "     - 本质：CAS只检查「当前值是否等于期望值」，" << std::endl;
    std::cout << "       不检查「值在中间是否被修改过」" << std::endl;
    std::cout << "     - 触发条件：内存被释放后重新分配/复用，地址恰好与之前相同" << std::endl;
    std::cout << "     - 为什么难调试：极低概率出现，需要高并发+精确时序" << std::endl;
    std::cout << std::endl;
    std::cout << "  3. 解决方案：在top指针旁边附加pop_count计数器，" << std::endl;
    std::cout << "     使用双宽CAS（DWCAS）原子地更新(top, count)组合值" << std::endl;
    std::cout << "     - 硬件支持：x86的cmpxchg16b提供16字节（128位）的CAS" << std::endl;
    std::cout << "     - 编译支持：需要-mcx16或-march=native选项" << std::endl;
    std::cout << "     - 原理：每次pop时counter+1，即使地址相同计数器也不同，" << std::endl;
    std::cout << "       CAS因此会失败"  << std::endl;
    std::cout << std::endl;
    std::cout << "  4. 讲座还讨论了风险指针（Hazard Pointers）用于安全的内存回收：" << std::endl;
    std::cout << "     - 线程在访问指针前将其声明为「风险指针」" << std::endl;
    std::cout << "     - 内存回收线程检查风险指针列表后才释放内存" << std::endl;
    std::cout << "     - 防止「use-after-free」（释放后使用）问题" << std::endl;

    return 0;
}
