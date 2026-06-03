// 02_stop_token.cpp — stop_token 协作式取消机制详解
// 演示: stop_source, stop_token, stop_callback

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <syncstream>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// ===== 1. 基础: 通过 stop_source 请求停止 =====
void demo_basic_stop() {
    std::cout << "=== 1. 基础停止机制 ===\n";

    std::stop_source source;
    auto token = source.get_token();

    std::jthread worker([token]() {
        int count = 0;
        while (!token.stop_requested()) {
            std::this_thread::sleep_for(10ms);
            ++count;
        }
        std::osyncstream(std::cout)
            << "  Worker 迭代 " << count << " 次后停止\n";
    });

    std::this_thread::sleep_for(100ms);
    bool result = source.request_stop();
    std::cout << "  request_stop() 返回 " << std::boolalpha << result << "\n";

    // 重复请求停止
    result = source.request_stop();
    std::cout << "  再次 request_stop() 返回 " << result << " (已停止)\n";
}

// ===== 2. stop_callback: 停止时自动回调 =====
void demo_stop_callback() {
    std::cout << "\n=== 2. stop_callback 停止回调 ===\n";

    std::stop_source source;
    auto token = source.get_token();

    // 注册回调: 当停止被请求时，自动调用
    int callback_invoked = 0;
    std::stop_callback cb(token, [&]() {
        callback_invoked = 42;
        std::osyncstream(std::cout) << "  [stop_callback] 清理资源中...\n";
    });

    // 回调也可用于一次性初始化
    std::stop_callback cb2(token, [&]() {
        std::osyncstream(std::cout) << "  [stop_callback2] 第二个回调\n";
    });

    std::cout << "  发出停止请求...\n";
    source.request_stop();

    std::cout << "  回调执行完成，callback_invoked = "
              << callback_invoked << " (期望 42)\n";
}

// ===== 3. condition_variable_any 与 stop_token =====
void demo_cv_with_stop() {
    std::cout << "\n=== 3. condition_variable 可中断等待 ===\n";

    std::mutex mtx;
    std::condition_variable_any cv;
    bool ready = false;

    std::stop_source source;
    auto token = source.get_token();

    std::jthread worker([&](std::stop_token stoken) {
        std::unique_lock lock(mtx);
        // wait 可被 stop_token 中断
        bool result = cv.wait(lock, stoken, [&] { return ready; });
        if (stoken.stop_requested()) {
            std::osyncstream(std::cout) << "  Worker: 被 stop_token 中断等待\n";
        } else {
            std::osyncstream(std::cout) << "  Worker: 条件满足，继续执行\n";
        }
    });

    std::this_thread::sleep_for(50ms);
    // 不设置 ready，而是请求停止——中断等待
    std::cout << "  请求停止 worker...\n";
    source.request_stop();
    cv.notify_all(); // 需要唤醒以检查 stop_token
}

// ===== 4. 多 stop_token 组合 =====
void demo_multi_token() {
    std::cout << "\n=== 4. 多 stop_token 监听 ===\n";

    std::stop_source source1, source2;
    std::jthread worker([t1 = source1.get_token(),
                          t2 = source2.get_token()]() {
        int iter = 0;
        while (!t1.stop_requested() && !t2.stop_requested()) {
            std::this_thread::sleep_for(10ms);
            ++iter;
        }
        std::osyncstream(std::cout)
            << "  Worker 停止 (t1=" << t1.stop_requested()
            << ", t2=" << t2.stop_requested()
            << "), 迭代 " << iter << " 次\n";
    });

    std::this_thread::sleep_for(50ms);
    source2.request_stop(); // 通过 source2 停止
    std::cout << "  通过 source2 请求停止\n";
}

int main() {
    demo_basic_stop();
    demo_stop_callback();
    demo_cv_with_stop();
    demo_multi_token();

    std::cout << "\nstop_token 提供了线程安全的协作式取消机制。\n";
    return 0;
}
