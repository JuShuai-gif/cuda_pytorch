// 04_threadsafe_lookup_table.cpp - 线程安全哈希查找表
// 分桶锁设计：每个桶独立互斥量，支持高并发读写
// 类似 Java ConcurrentHashMap 的简化版

#include <algorithm>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

template <typename Key, typename Value, typename Hash = std::hash<Key>>
class ThreadSafeLookupTable {
public:
    explicit ThreadSafeLookupTable(size_t num_buckets = 19)
        : buckets_(num_buckets) {}

    ThreadSafeLookupTable(const ThreadSafeLookupTable&) = delete;
    ThreadSafeLookupTable& operator=(const ThreadSafeLookupTable&) = delete;

    // 查找键对应的值
    std::optional<Value> find(const Key& key) const {
        auto& bucket = get_bucket(key);
        std::shared_lock<std::shared_mutex> lock(bucket.mutex); // 读锁（共享）
        return bucket.find(key);
    }

    // 插入或更新键值对
    void insert_or_update(const Key& key, const Value& value) {
        auto& bucket = get_bucket(key);
        std::unique_lock<std::shared_mutex> lock(bucket.mutex); // 写锁（独占）
        bucket.insert_or_update(key, value);
    }

    // 删除键
    void erase(const Key& key) {
        auto& bucket = get_bucket(key);
        std::unique_lock<std::shared_mutex> lock(bucket.mutex);
        bucket.erase(key);
    }

    // 获取当前条目数（近似值，遍历所有桶）
    size_t size() const {
        size_t total = 0;
        for (auto& bucket : buckets_) {
            std::shared_lock<std::shared_mutex> lock(bucket.mutex);
            total += bucket.size();
        }
        return total;
    }

private:
    struct Bucket {
        mutable std::shared_mutex mutex;

        using Entry = std::pair<Key, Value>;
        std::list<Entry> data; // list 有利于并发迭代（不会因插入导致指针失效）

        std::optional<Value> find(const Key& key) const {
            auto it = std::find_if(data.begin(), data.end(),
                [&](const auto& e) { return e.first == key; });
            if (it != data.end()) return it->second;
            return std::nullopt;
        }

        void insert_or_update(const Key& key, const Value& value) {
            auto it = std::find_if(data.begin(), data.end(),
                [&](const auto& e) { return e.first == key; });
            if (it != data.end()) {
                it->second = value; // 更新
            } else {
                data.emplace_back(key, value); // 插入
            }
        }

        void erase(const Key& key) {
            data.remove_if([&](const auto& e) { return e.first == key; });
        }

        size_t size() const { return data.size(); }
    };

    Bucket& get_bucket(const Key& key) const {
        size_t idx = hasher_(key) % buckets_.size();
        return buckets_[idx];
    }

    Hash                         hasher_;
    mutable std::vector<Bucket>   buckets_; // mutable: 读操作仍需对桶加锁
};

// ===== 测试 =====
int main() {
    std::cout << "=== ThreadSafeLookupTable (分桶锁) ===\n";

    const int kNumBuckets = 7;
    ThreadSafeLookupTable<int, std::string> table(kNumBuckets);

    const int kNumWriters = 4;
    const int kNumReaders = 8;
    const int kOpsPerThread = 1000;

    // 写线程：插入数据
    std::vector<std::jthread> writers;
    for (int w = 0; w < kNumWriters; ++w) {
        writers.emplace_back([&, w]() {
            for (int i = 0; i < kOpsPerThread; ++i) {
                int key = w * kOpsPerThread + i;
                table.insert_or_update(key, "value_" + std::to_string(key));
            }
        });
    }

    // 读线程：查找数据
    std::atomic<int> found_count{0};
    std::atomic<int> miss_count{0};
    std::vector<std::jthread> readers;
    for (int r = 0; r < kNumReaders; ++r) {
        readers.emplace_back([&, r]() {
            for (int i = 0; i < kOpsPerThread / 2; ++i) {
                // 随机查找已插入范围的 key
                int key = (r * 137 + i * 73) % (kNumWriters * kOpsPerThread);
                auto result = table.find(key);
                if (result.has_value()) {
                    found_count.fetch_add(1, std::memory_order_relaxed);
                } else {
                    miss_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    // jthread 自动 join
    writers.clear();
    readers.clear();

    std::cout << "  表大小: " << table.size() << "\n";
    std::cout << "  命中: " << found_count.load()
              << ", 未命中: " << miss_count.load() << "\n";
    std::cout << "  (未命中是因为读线程可能在写线程完成前查询)\n";
    return 0;
}
