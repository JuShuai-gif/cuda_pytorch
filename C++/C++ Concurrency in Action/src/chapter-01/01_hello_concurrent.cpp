// 01_hello_concurrent.cpp
// 知识点: 创建线程、join、基本的线程并发运行
// 演示: 创建2个线程分别执行不同的任务，主线程等待它们完成

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

void fct_task_a(const std::string& name) {
    for (int i = 0; i < 3; ++i) {
        std::cout << "[线程 " << name << "] 正在执行第 " << i + 1 << " 次任务\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cout << "[线程 " << name << "] 完成\n";
}

void fct_task_b(const std::string& name) {
    for (int i = 0; i < 3; ++i) {
        std::cout << "[线程 " << name << "] 正在处理数据块 #" << i + 1 << "\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
    std::cout << "[线程 " << name << "] 完成\n";
}

int main() {
    std::cout << "=== 并发 Hello World ===\n";
    std::cout << "主线程 ID: " << std::this_thread::get_id() << "\n\n";

    // 创建线程: 传入函数和参数
    std::thread t1(fct_task_a, "Worker-A");
    std::thread t2(fct_task_b, "Worker-B");

    std::cout << "线程 Worker-A ID: " << t1.get_id() << "\n";
    std::cout << "线程 Worker-B ID: " << t2.get_id() << "\n\n";

    t1.join();  // 等待 t1 完成
    t2.join();  // 等待 t2 完成

    std::cout << "\n所有线程已完成，主线程退出\n";
    return 0;
}
