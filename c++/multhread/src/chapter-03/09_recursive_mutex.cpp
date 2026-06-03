// 09_recursive_mutex.cpp — std::recursive_mutex 递归锁
//
// recursive_mutex 允许同一线程多次锁定同一个 mutex:
//  - 每次 lock() 计数 +1
//  - 每次 unlock() 计数 -1
//  - 计数归零时真正释放锁
//
// 典型场景: 递归函数需要锁保护、公共接口调用内部加锁方法

#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// ===== 1. 基础: recursive_mutex 允许多次加锁 =====
void demo_basic_recursive() {
    std::cout << "=== 1. recursive_mutex 基础 ===\n";

    std::recursive_mutex rmtx;
    int depth = 0;

    auto recursive_function = [&](int n, auto&& self) -> void {
        std::lock_guard lock(rmtx);
        ++depth;

        std::cout << "  深度 " << depth << ": n=" << n << "\n";

        if (n > 0) {
            self(n - 1, self); // 递归调用，再次加锁 (OK!)
        }

        --depth;
    };

    recursive_function(3, recursive_function);
    std::cout << "  recursive_mutex 允许同一线程递归加锁\n";
}

// ===== 2. 对比: 普通 mutex 递归加锁会死锁 =====
void demo_normal_mutex_deadlock() {
    std::cout << "\n=== 2. 普通 mutex 递归死锁 (演示) ===\n";
    std::cout << "  以下代码如果运行会导致死锁 (已注释):\n";
    std::cout << "  std::mutex mtx;\n";
    std::cout << "  auto bad = [&](int n, auto&& self) {\n";
    std::cout << "      std::lock_guard lock(mtx);  // 第1次加锁 OK\n";
    std::cout << "      if (n > 0) self(n-1, self); // 第2次加锁: DEADLOCK!\n";
    std::cout << "  };\n";
}

// ===== 3. 实际场景: 公共接口 + 内部实现 =====
class RecursiveCounter {
public:
    void increment() {
        std::lock_guard lock(mtx_);
        ++value_;
    }

    // reset_and_increment 调用 increment (两者都需要锁)
    void reset_and_increment() {
        std::lock_guard lock(mtx_); // 外层加锁
        value_ = 0;
        increment(); // 内层再加锁: 普通 mutex 会死锁!
        // 使用 recursive_mutex 则 OK
    }

    int value() const {
        std::lock_guard lock(mtx_);
        return value_;
    }

private:
    mutable std::recursive_mutex mtx_;
    int value_ = 0;
};

void demo_recursive_pattern() {
    std::cout << "\n=== 3. 公共接口调用内部方法 ===\n";

    RecursiveCounter counter;
    counter.reset_and_increment();
    counter.increment();

    std::cout << "  value = " << counter.value() << " (期望 2)\n";
    std::cout << "  recursive_mutex 使公共方法可以安全调用内部方法\n";
}

// ===== 4. recursive_mutex 的坑 =====
void demo_pitfalls() {
    std::cout << "\n=== 4. recursive_mutex 的注意事项 ===\n";

    std::cout << "  坑1: 隐藏设计问题, 需要递归锁通常意味着锁粒度太大\n";
    std::cout << "  坑2: 性能开销, 比普通 mutex 略慢 (需要维护计数)\n";
    std::cout << "  坑3: 难以推理, 不清楚当前线程持锁计数是多少\n";
    std::cout << "  坑4: 不能与 condition_variable 搭配使用\n";
    std::cout << "       (cv.wait 只释放一次锁，计数可能不是 0)\n\n";

    std::cout << "  替代方案:\n";
    std::cout << "    1. 提取不加锁的内部实现函数 (前缀 _impl)\n";
    std::cout << "    2. 减小锁粒度，在调用前释放锁\n";
    std::cout << "    3. 重新设计 API 避免嵌套加锁\n";
}

// ===== 5. 该用还是不该用 =====
void demo_guidelines() {
    std::cout << "\n=== 5. 使用建议 ===\n";

    std::cout << "  适用场景:\n";
    std::cout << "    - 递归数据结构 (如树的遍历)\n";
    std::cout << "    - 无法轻易修改的遗留代码\n";
    std::cout << "    - 公共 API + 内部方法都需锁的场景\n\n";

    std::cout << "  不适用:\n";
    std::cout << "    - 新代码中能用普通 mutex 解决的\n";
    std::cout << "    - 需要 condition_variable 的场景\n";
    std::cout << "    - 对性能敏感的热路径\n";
}

int main() {
    demo_basic_recursive();
    demo_normal_mutex_deadlock();
    demo_recursive_pattern();
    demo_pitfalls();
    demo_guidelines();

    std::cout << "\n经验法则: 如果代码需要 recursive_mutex,\n";
    std::cout << "先问自己: 能否重构 API 避免嵌套加锁? 90% 的情况答案是 Yes。\n";
    return 0;
}
