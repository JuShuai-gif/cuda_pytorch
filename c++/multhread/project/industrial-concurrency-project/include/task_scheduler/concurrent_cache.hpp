#pragma once
// Chapter 3.3.2: Protecting Data with std::shared_mutex (Read-Write Lock)
// Implements a thread-safe LRU cache optimized for read-heavy workloads.
// Uses std::shared_mutex (C++17) for:
//   - Multiple concurrent readers (shared_lock)
//   - Single exclusive writer (unique_lock)
// Ch3.3.2 demonstrates this pattern for DNS cache; we generalize it.

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

    // Ch3.3.2: Get with shared (read) lock - allows concurrent reads.
    // Returns std::nullopt if key is not found.
    std::optional<Value> get(const Key& key) {
        // Ch3.3.2: std::shared_lock for read access - multiple threads can hold.
        std::shared_lock lock(mutex_);
        auto it = cache_.find(key);
        if (it == cache_.end()) {
            return std::nullopt;
        }
        // Move the accessed item to front of LRU list (requires exclusive lock).
        // For simplicity, we return value without updating LRU on read.
        // A production version would use a lock-free approximation or defer updates.
        return it->second.first;
    }

    // Ch3.3.1: Put with exclusive (write) lock - only one writer at a time.
    void put(const Key& key, const Value& value) {
        std::unique_lock lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            // Update existing: move to front
            lru_list_.erase(it->second.second);
            lru_list_.push_front(key);
            it->second = {value, lru_list_.begin()};
            return;
        }
        // Evict if needed (LRU eviction, Ch6.2: safe under exclusive lock).
        if (cache_.size() >= max_size_) {
            auto last = lru_list_.back();
            lru_list_.pop_back();
            cache_.erase(last);
        }
        // Insert new
        lru_list_.push_front(key);
        cache_[key] = {value, lru_list_.begin()};
    }

    // Put with move semantics (Ch3.2.8: avoid copies on hot paths).
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

    // Ch3.3.2: Contains check with shared lock.
    [[nodiscard]] bool contains(const Key& key) {
        std::shared_lock lock(mutex_);
        return cache_.find(key) != cache_.end();
    }

    // Ch3.3.2: Size query with shared lock.
    [[nodiscard]] size_t size() const {
        std::shared_lock lock(mutex_);
        return cache_.size();
    }

    // Clear with exclusive lock.
    void clear() {
        std::unique_lock lock(mutex_);
        cache_.clear();
        lru_list_.clear();
    }

private:
    size_t max_size_;
    // Ch3.3.2: std::shared_mutex enables multiple readers, single writer.
    mutable std::shared_mutex mutex_;
    using LruIter = typename std::list<Key>::iterator;
    std::unordered_map<Key, std::pair<Value, LruIter>> cache_;
    std::list<Key> lru_list_;
};

} // namespace task_scheduler
