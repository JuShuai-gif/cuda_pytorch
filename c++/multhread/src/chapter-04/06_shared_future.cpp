// 06_shared_future.cpp - std::shared_future 多线程等待同一结果
// 场景：一次性配置加载后通知所有工作线程

#include <chrono>
#include <future>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// 模拟：从文件/网络加载配置
std::string load_configuration() {
    std::cout << "[Loader] 开始加载配置...\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::string config = "server=192.168.1.1;port=8080;timeout=30";
    std::cout << "[Loader] 配置加载完成: " << config << "\n";
    return config;
}

// 工作线程：等待 shared_future 以获取配置
void worker(int id, std::shared_future<std::string> config_future) {
    std::cout << "[Worker " << id << "] 等待配置...\n";
    // shared_future::get() 可被多个线程调用，返回 const 引用
    const std::string& config = config_future.get();
    std::cout << "[Worker " << id << "] 获得配置: " << config << "\n";
}

int main() {
    // promise 设置值后，通过 share() 获得 shared_future
    std::promise<std::string> config_promise;
    std::shared_future<std::string> config_future = config_promise.get_future().share();

    // 启动配置加载线程
    std::jthread loader([&config_promise]() {
        std::string config = load_configuration();
        config_promise.set_value(std::move(config));
    });

    // 启动多个工作线程，共享同一个 shared_future
    const int kNumWorkers = 5;
    std::vector<std::jthread> workers;
    for (int i = 0; i < kNumWorkers; ++i) {
        workers.emplace_back(worker, i, config_future);
    }

    std::cout << "[Main] 所有线程已结束\n";
    return 0;
}
