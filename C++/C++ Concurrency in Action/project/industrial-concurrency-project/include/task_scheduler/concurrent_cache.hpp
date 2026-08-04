#pragma once
// Ch3.3.2：使用 std::shared_mutex 保护数据（读写锁）
// 实现一个面向读密集型工作负载优化的线程安全 LRU 缓存。
// 使用 std::shared_mutex（C++17）实现：
//   - 多个并发读者（shared_lock）
//   - 单个独占写者（unique_lock）
// Ch3.3.2 以 DNS 缓存为例演示此模式；我们将其泛化。

// 保护数据：使用 std::shared_mutex（读写锁）
// 多个并发读者（shared_lock），单个独占写者（unique_lock）

#include <shared_mutex>
#include <unordered_map>
#include <list>
#include <optional>
#include <utility>
#include <mutex>
#include <chrono>
#include <cstddef>

namespace task_scheduler {

template <typename Key, typename Value>
class ConcurrentCache {
public:
    explicit ConcurrentCache(size_t max_size = 1024) : max_size_(max_size) {}

    ConcurrentCache(const ConcurrentCache&) = delete;
    ConcurrentCache& operator=(const ConcurrentCache&) = delete;

    // Ch3.3.2：使用共享（读）锁获取——允许多个并发读。
    // 如果未找到 key，返回 std::nullopt。
    // 使用共享读锁获取，允许多个并发读取
    std::optional<Value> get(const Key& key) {
        // Ch3.3.2：std::shared_lock 用于读访问——多个线程可以同时持有。
        // 读访问使用 shared_lock，允许多个线程同时持有
        std::shared_lock lock(mutex_);
        auto it = cache_.find(key);
        if (it == cache_.end()) {
            return std::nullopt;
        }
        // 将被访问项移到 LRU 列表前面需要独占锁。
        // 为简化起见，我们返回值而不在读取时更新 LRU。
        // 生产版本会使用无锁近似方法或延迟更新。
        return it->second.first;
    }

    // Ch3.3.1：使用独占（写）锁写入——同一时间只有一个写者。
    // 写操作使用 unique_lock，同一时间只有一个写者
    void put(const Key& key, const Value& value) {
        std::unique_lock lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            // 更新已存在项：移到链表前端
            lru_list_.erase(it->second.second);
            lru_list_.push_front(key);
            it->second = {value, lru_list_.begin()};
            return;
        }
        // 如需要则淘汰（LRU 淘汰，Ch6.2：在独占锁下安全）。
        if (cache_.size() >= max_size_) {
            auto last = lru_list_.back();
            lru_list_.pop_back();
            cache_.erase(last);
        }
        // 插入新项
        lru_list_.push_front(key);
        cache_[key] = {value, lru_list_.begin()};
    }

    // 使用移动语义写入（Ch3.2.8：在热点路径上避免拷贝）。
    // 移动语义版本：在热点路径上避免拷贝
    void put(const Key& key, Value&& value) {
        std::unique_lock lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            lru_list_.erase(it->second.second);
            lru_list_.push_front(key);
            it->second = {std::move(value), lru_list_.begin()};
            return;
        }
        if (cache_.size() >= max_size_) {
            auto last = lru_list_.back();
            lru_list_.pop_back();
            cache_.erase(last);
        }
        lru_list_.push_front(key);
        cache_[key] = {std::move(value), lru_list_.begin()};
    }

    // Ch3.3.2：使用共享锁检查是否包含。
    // 使用共享锁检查键是否存在
    [[nodiscard]] bool contains(const Key& key) {
        std::shared_lock lock(mutex_);
        return cache_.find(key) != cache_.end();
    }

    // Ch3.3.2：使用共享锁查询大小。
    // 使用共享锁查询缓存大小
    [[nodiscard]] size_t size() const {
        std::shared_lock lock(mutex_);
        return cache_.size();
    }

    // 使用独占锁清空。
    // 清空缓存，需要独占锁
    void clear() {
        std::unique_lock lock(mutex_);
        cache_.clear();
        lru_list_.clear();
    }

private:
    size_t max_size_;
    // Ch3.3.2：std::shared_mutex 允许多个读者、单个写者。
    // mutable 允许在 const 方法中加锁
    mutable std::shared_mutex mutex_;
    using LruIter = typename std::list<Key>::iterator;
    // 哈希表存储键到 (值, LRU迭代器) 的映射
    std::unordered_map<Key, std::pair<Value, LruIter>> cache_;
    // LRU 链表：头部是最近使用的，尾部是最久未使用的
    std::list<Key> lru_list_;
};

} // namespace task_scheduler
