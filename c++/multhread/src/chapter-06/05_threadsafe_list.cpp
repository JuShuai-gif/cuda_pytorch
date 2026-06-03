// 05_threadsafe_list.cpp - 线程安全链表
// 支持并发安全的遍历、插入、删除操作
// 使用 shared_mutex：遍历用读锁，修改用写锁
// 也可使用逐节点加锁的手递手（hand-over-hand locking）策略

#include <algorithm>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

template <typename T>
class ThreadSafeList {
    struct Node {
        mutable std::mutex  mutex;
        std::unique_ptr<Node> next;
        T                   data;

        Node() = default;
        explicit Node(T value) : data(std::move(value)) {}
    };

public:
    ThreadSafeList() {
        // 头尾哨兵节点，简化边界处理
        head_       = std::make_unique<Node>();
        auto tail   = std::make_unique<Node>();
        tail_       = tail.get();
        head_->next = std::move(tail);
    }

    ThreadSafeList(const ThreadSafeList&) = delete;
    ThreadSafeList& operator=(const ThreadSafeList&) = delete;

    // 在链表头部插入
    void push_front(T value) {
        auto new_node = std::make_unique<Node>(std::move(value));
        std::lock_guard<std::mutex> lock(head_->mutex);
        new_node->next = std::move(head_->next);
        head_->next    = std::move(new_node);
    }

    // 查找链表中是否存在某个值
    bool contains(const T& value) const {
        // 手递手加锁：每次只锁两个节点，释放前一个
        Node* prev = head_.get();
        std::unique_lock<std::mutex> prev_lock(prev->mutex);

        Node* curr = prev->next.get();
        while (curr != tail_) {
            std::unique_lock<std::mutex> curr_lock(curr->mutex);
            // 释放前一个节点的锁（缩小锁粒度）
            prev_lock.unlock();

            if (curr->data == value) {
                return true;
            }

            prev      = curr;
            prev_lock = std::move(curr_lock);
            curr      = curr->next.get();
        }
        return false;
    }

    // 删除第一个匹配的值
    bool remove(const T& value) {
        Node* prev = head_.get();
        std::unique_lock<std::mutex> prev_lock(prev->mutex);

        Node* curr = prev->next.get();
        while (curr != tail_) {
            std::unique_lock<std::mutex> curr_lock(curr->mutex);

            if (curr->data == value) {
                // 找到目标，修改链表指针
                prev->next = std::move(curr->next);
                // curr 节点离开作用域后自动删除
                return true;
            }

            prev_lock.unlock();
            prev      = curr;
            prev_lock = std::move(curr_lock);
            curr      = curr->next.get();
        }
        return false;
    }

    // 打印链表内容
    void print() const {
        std::lock_guard<std::mutex> lock(head_->mutex);
        auto* curr = head_->next.get();
        std::cout << "List: ";
        while (curr != tail_) {
            std::cout << curr->data << " ";
            curr = curr->next.get();
        }
        std::cout << "\n";
    }

    // 计算节点数（不含哨兵）
    size_t size() const {
        size_t count = 0;
        std::lock_guard<std::mutex> lock(head_->mutex);
        auto* curr = head_->next.get();
        while (curr != tail_) {
            ++count;
            curr = curr->next.get();
        }
        return count;
    }

private:
    std::unique_ptr<Node> head_;  // 头哨兵
    Node*                 tail_;  // 尾哨兵（裸指针，无数据）
};

// ===== 全锁共享版本（简单但粒度粗） =====
template <typename T>
class SimpleThreadSafeList {
public:
    void push_front(T value) {
        std::lock_guard<std::shared_mutex> lock(mutex_);
        data_.push_front(std::move(value));
    }

    bool contains(const T& value) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return std::find(data_.begin(), data_.end(), value) != data_.end();
    }

    bool remove(const T& value) {
        std::lock_guard<std::shared_mutex> lock(mutex_);
        auto it = std::find(data_.begin(), data_.end(), value);
        if (it != data_.end()) {
            data_.erase(it);
            return true;
        }
        return false;
    }

    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return data_.size();
    }

private:
    mutable std::shared_mutex mutex_;
    std::list<T>              data_;
};

// ===== 测试 =====
int main() {
    std::cout << "=== ThreadSafeList (手递手锁) ===\n";

    ThreadSafeList<int> list;

    // 并发插入
    {
        std::vector<std::jthread> threads;
        for (int i = 0; i < 4; ++i) {
            threads.emplace_back([&, i]() {
                for (int v = 0; v < 25; ++v) {
                    list.push_front(i * 100 + v);
                }
            });
        }
    }

    std::cout << "  插入后大小: " << list.size() << "\n";

    // 并发查询
    {
        std::atomic<int> found{0};
        std::vector<std::jthread> threads;
        for (int i = 0; i < 4; ++i) {
            threads.emplace_back([&]() {
                for (int v = 0; v < 100; ++v) {
                    if (list.contains(v)) {
                        found.fetch_add(1);
                    }
                }
            });
        }
        threads.clear();
        std::cout << "  找到次数: " << found.load() << "\n";
    }

    // 并发删除
    {
        std::vector<std::jthread> threads;
        for (int i = 0; i < 4; ++i) {
            threads.emplace_back([&, i]() {
                for (int v = 0; v < 25; ++v) {
                    list.remove(i * 100 + v);
                }
            });
        }
    }

    std::cout << "  删除后大小: " << list.size() << "\n";
    list.print();

    // ===== SimpleThreadSafeList 测试 =====
    std::cout << "\n=== SimpleThreadSafeList (全锁) ===\n";

    SimpleThreadSafeList<std::string> simple_list;

    std::vector<std::jthread> threads;
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back([&, i]() {
            simple_list.push_front("item_" + std::to_string(i));
        });
    }
    threads.clear();

    std::cout << "  大小: " << simple_list.size() << "\n";
    std::cout << "  contains 'item_3': " << simple_list.contains("item_3") << "\n";
    simple_list.remove("item_3");
    std::cout << "  contains 'item_3' after remove: " << simple_list.contains("item_3") << "\n";

    return 0;
}
