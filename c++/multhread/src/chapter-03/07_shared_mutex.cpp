// 07_shared_mutex.cpp
// 知识点: std::shared_mutex (C++17) 实现读写锁
// 演示: 读多写少的线程安全缓存 (Reader-Writer Lock)
// 对应书中 3.3.2 节

#include <chrono>
#include <iostream>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

// =============================================================================
// ThreadSafeCache: 使用 std::shared_mutex 的线程安全缓存
//
// 策略:
//   - 读操作: shared_lock (多个线程可同时读)
//   - 写操作: unique_lock (独占访问)
// =============================================================================
class ThreadSafeCache {
public:
    // 读操作: 使用 shared_lock (允许多个线程并发读)
    [[nodiscard]] std::string get(const std::string& key) const {
        std::shared_lock lock(m_mutex);  // 共享锁: 多个读者可同时持有
        auto             it = m_cache.find(key);
        return (it != m_cache.end()) ? it->second : "NOT_FOUND";
    }

    // 检查 key 是否存在
    [[nodiscard]] bool contains(const std::string& key) const {
        std::shared_lock lock(m_mutex);
        return m_cache.find(key) != m_cache.end();
    }

    // 写操作: 使用 unique_lock (独占访问)
    void put(const std::string& key, const std::string& value) {
        std::unique_lock lock(m_mutex);  // 独占锁: 写者独占
        m_cache[key] = value;
    }

    // 删除
    void remove(const std::string& key) {
        std::unique_lock lock(m_mutex);
        m_cache.erase(key);
    }

    // 获取缓存大小
    [[nodiscard]] size_t size() const {
        std::shared_lock lock(m_mutex);
        return m_cache.size();
    }

    // 批量读取: 获取所有 key
    [[nodiscard]] std::vector<std::string> keys() const {
        std::shared_lock lock(m_mutex);
        std::vector<std::string> result;
        result.reserve(m_cache.size());
        for (const auto& [key, _] : m_cache) {
            result.push_back(key);
        }
        return result;
    }

    // 清空缓存
    void clear() {
        std::unique_lock lock(m_mutex);
        m_cache.clear();
    }

private:
    mutable std::shared_mutex m_mutex;  // mutable 允许 const 方法中锁定
    std::map<std::string, std::string> m_cache;
};

// =============================================================================
// 性能对比: shared_mutex vs 普通 mutex
// =============================================================================

class SimpleMutexCache {
public:
    [[nodiscard]] std::string get(const std::string& key) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto                        it = m_cache.find(key);
        return (it != m_cache.end()) ? it->second : "NOT_FOUND";
    }

    void put(const std::string& key, const std::string& value) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_cache[key] = value;
    }

private:
    mutable std::mutex m_mutex;
    std::map<std::string, std::string> m_cache;
};

// =============================================================================
// 测试函数
// =============================================================================

void fct_read_heavy_test(ThreadSafeCache& cache, int thread_id, int num_reads) {
    for (int i = 0; i < num_reads; ++i) {
        std::string key = "key_" + std::to_string(i % 100);
        std::string val = cache.get(key);  // 多个线程可同时读取
        (void)val;  // 抑制 nodiscard 警告
    }
}

void fct_write_test(ThreadSafeCache& cache, int thread_id, int num_writes) {
    for (int i = 0; i < num_writes; ++i) {
        cache.put("key_" + std::to_string(i % 100),
                  "value_" + std::to_string(thread_id) + "_" +
                      std::to_string(i));
    }
}

int main() {
    std::cout << "=== std::shared_mutex 读写锁 ===\n\n";

    // --- 测试1: 基本读写 ---
    std::cout << "--- 测试1: 基本读写操作 ---\n";
    {
        ThreadSafeCache cache;

        // 写入初始数据
        cache.put("name", "Alice");
        cache.put("role", "Engineer");
        cache.put("lang", "C++");

        // 并发读: 多个读者同时访问
        auto reader = [&cache](int id) {
            std::cout << "  [Reader " << id
                      << "] name=" << cache.get("name")
                      << " role=" << cache.get("role") << "\n";
        };

        std::thread t1(reader, 1);
        std::thread t2(reader, 2);
        std::thread t3(reader, 3);

        t1.join();
        t2.join();
        t3.join();

        std::cout << "  缓存大小: " << cache.size() << "\n";

        auto all_keys = cache.keys();
        std::cout << "  所有 key: ";
        for (const auto& k : all_keys) {
            std::cout << k << " ";
        }
        std::cout << "\n";
    }

    // --- 测试2: 读写混合 ---
    std::cout << "\n--- 测试2: 读写混合并发 ---\n";
    {
        ThreadSafeCache cache;

        // 预填充数据
        for (int i = 0; i < 100; ++i) {
            cache.put("key_" + std::to_string(i),
                      "initial_" + std::to_string(i));
        }

        const int             num_readers   = 8;
        const int             num_writers   = 2;
        const int             reads_per_th  = 10'000;
        const int             writes_per_th = 1'000;
        std::vector<std::jthread> threads;

        auto start = std::chrono::high_resolution_clock::now();

        // 启动读者线程
        for (int i = 0; i < num_readers; ++i) {
            threads.emplace_back(fct_read_heavy_test, std::ref(cache), i,
                                 reads_per_th);
        }

        // 启动写者线程
        for (int i = 0; i < num_writers; ++i) {
            threads.emplace_back(fct_write_test, std::ref(cache), i,
                                 writes_per_th);
        }

        threads.clear();  // jthread 自动 join

        auto end = std::chrono::high_resolution_clock::now();
        auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                        start);

        std::cout << "  读者线程: " << num_readers
                  << " (各 " << reads_per_th << " 次读)\n";
        std::cout << "  写者线程: " << num_writers
                  << " (各 " << writes_per_th << " 次写)\n";
        std::cout << "  总耗时: " << ms.count() << "ms\n";
        std::cout << "  缓存大小: " << cache.size() << "\n";
        std::cout << "  shared_mutex 允许读者并发，提升读多写少场景性能\n";
    }

    // --- 测试3: contains 和 remove ---
    std::cout << "\n--- 测试3: contains 和 remove ---\n";
    {
        ThreadSafeCache cache;
        cache.put("temp", "data");

        std::cout << "  contains('temp'): " << cache.contains("temp") << "\n";
        std::cout << "  contains('nope'): " << cache.contains("nope") << "\n";

        cache.remove("temp");
        std::cout << "  remove 后 contains('temp'): "
                  << cache.contains("temp") << "\n";
    }

    std::cout << "\n=== shared_mutex 使用要点 ===\n";
    std::cout << "1. shared_lock:  共享锁，多个读者可同时持有\n";
    std::cout << "2. unique_lock:  独占锁，写者独占 (或 lock_guard)\n";
    std::cout << "3. mutable:      const 方法中使用 shared_lock\n";
    std::cout << "4. 适用场景:     读多写少的缓存、配置\n";
    std::cout << "5. 注意:         写者优先级问题 (可能饿死)\n";
    std::cout << "6. C++14:        用 std::shared_timed_mutex\n";
    std::cout << "7. C++17:        用 std::shared_mutex (更高性能)\n";

    return 0;
}
