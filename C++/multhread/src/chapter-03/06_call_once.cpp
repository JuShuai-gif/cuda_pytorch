// 06_call_once.cpp
// 知识点: std::once_flag + std::call_once 线程安全单次初始化
// 演示: 延迟初始化、单例模式、资源初始化
// 对应书中 3.3.1 节

#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// =============================================================================
// 使用 std::call_once 实现线程安全的单例
// =============================================================================
class ConfigManager {
public:
    // 获取单例
    static ConfigManager& instance() {
        // call_once 保证 init() 只被调用一次
        // 即使多个线程同时调用 instance()
        std::call_once(s_once_flag, &ConfigManager::init_singleton);
        return *s_instance;
    }

    void set(const std::string& key, const std::string& value) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_config[key] = value;
    }

    [[nodiscard]] std::string get(const std::string& key) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto                        it = m_config.find(key);
        return (it != m_config.end()) ? it->second : "NOT_FOUND";
    }

    [[nodiscard]] bool is_initialized() const { return m_initialized; }

public:
    ConfigManager() {
        std::cout << "[ConfigManager] 构造 (应该只看到一次)\n";
        // 模拟昂贵的初始化
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    ~ConfigManager() = default;

    static void init_singleton() {
        s_instance.reset(new ConfigManager());
    }

    static std::once_flag            s_once_flag;
    static std::unique_ptr<ConfigManager> s_instance;

    mutable std::mutex                   m_mutex;
    std::map<std::string, std::string>   m_config;
    bool                                 m_initialized = true;
};

std::once_flag                         ConfigManager::s_once_flag;
std::unique_ptr<ConfigManager>             ConfigManager::s_instance;

// =============================================================================
// 使用 call_once 进行连接的延迟初始化
// =============================================================================
class DatabaseConnection {
public:
    void initialize(const std::string& conn_string) {
        std::cout << "[DB] 正在初始化连接: " << conn_string << "\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        m_conn_string = conn_string;
        m_connected   = true;
        std::cout << "[DB] 初始化完成\n";
    }

    void query(const std::string& sql) {
        // 在使用前确保已初始化
        std::call_once(m_init_flag, &DatabaseConnection::initialize, this,
                       "default://localhost:5432");

        std::lock_guard<std::mutex> lock(m_mutex);
        std::cout << "[DB] 执行查询: " << sql
                  << " (conn: " << m_conn_string << ")\n";
    }

private:
    std::once_flag  m_init_flag;
    std::mutex      m_mutex;
    std::string     m_conn_string = "uninitialized";
    bool            m_connected   = false;
};

// =============================================================================
// 使用 call_once 的延迟初始化包装器
// =============================================================================
class LazyResource {
public:
    explicit LazyResource(int id) : m_id(id) {}

    void use() {
        // call_once 保证 initialize 只执行一次
        std::call_once(m_init_flag, &LazyResource::do_heavy_init, this);

        std::lock_guard<std::mutex> lock(m_mutex);
        std::cout << "[资源 " << m_id << "] 使用中, data=" << m_data << "\n";
    }

private:
    void do_heavy_init() {
        std::cout << "[资源 " << m_id << "] 正在执行重量级初始化...\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
        m_data = m_id * 1000;
        std::cout << "[资源 " << m_id << "] 初始化完成\n";
    }

    int            m_id;
    std::once_flag m_init_flag;
    std::mutex     m_mutex;
    int            m_data = 0;
};

int main() {
    std::cout << "=== std::call_once 线程安全单次初始化 ===\n\n";

    // --- 测试1: 单例模式 ---
    std::cout << "--- 测试1: 线程安全单例 ---\n";
    {
        const int             num_threads = 8;
        std::vector<std::jthread> threads;
        threads.reserve(num_threads);

        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([i]() {
                auto& cfg = ConfigManager::instance();
                cfg.set("thread_" + std::to_string(i),
                        "value_" + std::to_string(i));
            });
        }
        // jthread 自动 join

        auto& cfg = ConfigManager::instance();
        std::cout << "  单例已初始化: " << cfg.is_initialized() << "\n";
        std::cout << "  配置项 thread_0 = " << cfg.get("thread_0") << "\n";
        std::cout << "  配置项 thread_7 = " << cfg.get("thread_7") << "\n";
    }

    // --- 测试2: 延迟初始化 ---
    std::cout << "\n--- 测试2: 延迟初始化 (用前才初始化) ---\n";
    {
        DatabaseConnection db;
        std::cout << "  数据库对象已创建，但尚未连接\n";

        const int             num_threads = 4;
        std::vector<std::jthread> threads;
        threads.reserve(num_threads);

        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&db, i]() {
                db.query("SELECT * FROM users WHERE id = " +
                         std::to_string(i));
            });
        }
        // jthread 自动 join
    }

    // --- 测试3: 多资源延迟初始化 ---
    std::cout << "\n--- 测试3: 多资源延迟初始化 ---\n";
    {
        LazyResource          res1(1);
        LazyResource          res2(2);
        const int             num_threads = 4;
        std::vector<std::jthread> threads;
        threads.reserve(num_threads);

        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&res1, &res2, i]() {
                res1.use();
                res2.use();
            });
        }
        // jthread 自动 join
    }

    // --- 测试4: 异常处理 ---
    std::cout << "\n--- 测试4: call_once 异常处理 ---\n";
    {
        std::once_flag flag;
        int            attempts = 0;

        auto fct_init_may_fail = [&attempts]() {
            ++attempts;
            std::cout << "  初始化尝试 #" << attempts << "\n";
            if (attempts < 2) {
                throw std::runtime_error("初始化失败");
            }
            std::cout << "  初始化成功!\n";
        };

        // 第一次调用: 抛出异常
        try {
            std::call_once(flag, fct_init_may_fail);
        } catch (const std::exception& e) {
            std::cout << "  第1次 call_once 失败: " << e.what() << "\n";
        }

        // 异常后，once_flag 未被设置，可以再次尝试
        std::cout << "  once_flag 未被标记，可以重试\n";

        // 第二次调用: 仍然失败
        try {
            std::call_once(flag, fct_init_may_fail);
        } catch (const std::exception& e) {
            std::cout << "  第2次 call_once 失败: " << e.what() << "\n";
        }

        // 第三次: 成功
        std::call_once(flag, fct_init_may_fail);

        // 第四次调用: 不会执行 (once_flag 已标记)
        std::call_once(flag, fct_init_may_fail);
        std::cout << "  第4次 call_once: 函数不再执行 (flag 已设置)\n";
    }

    std::cout << "\n=== call_once 使用场景 ===\n";
    std::cout << "1. 单例模式: 替代双重检查锁定(DCLP)\n";
    std::cout << "2. 延迟初始化: 连接池、配置加载\n";
    std::cout << "3. 缓存预热: 首次使用时初始化\n";
    std::cout << "4. 异常处理: 抛异常时 flag 不设置，可重试\n";
    std::cout << "5. 比手动旗标+互斥量更简洁高效\n";

    return 0;
}
