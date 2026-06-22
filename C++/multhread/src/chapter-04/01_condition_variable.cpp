// 01_condition_variant.cpp - 经典生产者-消费者模型
// 演示 std::condition_variable + std::unique_lock + std::mutex 的基本用法
// 核心：使用 while 循环等待条件，防止虚假唤醒 (spurious wakeup)

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

class SimpleProducerConsumer {
public:
    // 生产者：向队列添加数据并通知消费者
    void produce(const std::string& item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(item);
            std::cout << "[Producer] 生产: " << item << " (队列大小: " << queue_.size() << ")\n";
        }
        // 通知一个等待的消费者（在锁外通知可减少阻塞）
        cond_var_.notify_one();
    }

    // 消费者：从队列取出数据；队列空则阻塞等待
    std::string consume() {
        std::unique_lock<std::mutex> lock(mutex_);

        // ⚠️ 必须用 while 而非 if：防止虚假唤醒及惊群效应
        //    当线程被唤醒时，队列可能已被其他消费者清空
        cond_var_.wait(lock, [this]() { return !queue_.empty(); });

        std::string item = queue_.front();
        queue_.pop();
        lock.unlock(); // 尽早释放锁

        std::cout << "[Consumer] 消费: " << item << " (队列大小: " << queue_.size() << ")\n";
        return item;
    }

    // 通知所有等待线程（用于关闭场景）
    void notify_all() { cond_var_.notify_all(); }

private:
    std::mutex              mutex_;
    std::condition_variable cond_var_;
    std::queue<std::string> queue_;
};

int main() {
    SimpleProducerConsumer pc;

    // 启动 3 个消费者线程
    std::vector<std::jthread> consumers;
    for (int i = 0; i < 3; ++i) {
        consumers.emplace_back([&pc, i]() {
            for (int j = 0; j < 5; ++j) {
                pc.consume();
            }
            std::cout << "[Consumer " << i << "] 完成工作\n";
        });
    }

    // 生产者线程
    std::jthread producer([&pc]() {
        for (int i = 0; i < 15; ++i) {
            pc.produce("任务-" + std::to_string(i));
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });

    // jthread 自动 join，无需手动管理
    std::cout << "[Main] 所有线程已结束\n";
    return 0;
}
