# 第6章：基于锁的并发数据结构设计

> 锁是最直观的并发控制手段，但"加锁"本身不是目的，如何在保证正确性的前提下最大化并发度才是关键。

---

## 6.1 并发设计的意义

### 线程安全 ≠ 真正并发

一个类是"线程安全"的，只意味着多个线程同时调用它的方法不会导致数据损坏。但这**不等于**它支持高并发——如果所有方法都用一个全局大锁保护，同一时刻只有一个线程能干活，其他全部阻塞。

**生活类比**：一栋写字楼只有一个卫生间（全局大锁）——确实安全（不会出现两个人用一个坑位的尴尬），但高峰期大家排长队，效率极低。改进方案：每层楼都有卫生间（细粒度锁/分桶锁），让不同楼层的人同时使用。

### 正确性条件

设计并发数据结构时，必须思考两个问题：

- **不变量（Invariant）**：数据结构在"外部看来"始终合法的性质。例如链表的"每个节点要么被 head 可达，要么已释放"。
- **序列化点（Serialization Point）**：所有操作看起来像在某一个瞬间原子完成。一个有锁的 `push()` 方法，序列化点就在获得锁之后、释放锁之前。

---

## 6.2 基于锁的并发数据结构

### 6.2.1 线程安全栈

最简单的实现：用 `std::mutex` 包裹 `std::stack` 的所有操作。

```cpp
#include <mutex>
#include <stack>
#include <memory>
#include <exception>

template<typename T>
class ThreadSafeStack {
    std::stack<T> data_;
    mutable std::mutex mtx_;

public:
    ThreadSafeStack() = default;

    // 拷贝构造：同时锁住两个栈（按地址排序避免死锁）
    ThreadSafeStack(const ThreadSafeStack& other) {
        std::lock_guard<std::mutex> lock(other.mtx_);
        data_ = other.data_;
    }

    void push(T value) {
        std::lock_guard<std::mutex> lock(mtx_);
        data_.push(std::move(value));
    }

    // 关键设计：pop 不返回值，而是通过引用参数
    // 避免 top() + pop() 的 TOCTOU 竞争
    void pop(T& result) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (data_.empty()) {
            throw std::runtime_error("stack is empty");
        }
        result = std::move(data_.top());
        data_.pop();
    }

    std::shared_ptr<T> pop() {
        std::lock_guard<std::mutex> lock(mtx_);
        if (data_.empty()) {
            return std::shared_ptr<T>();
        }
        auto res = std::make_shared<T>(std::move(data_.top()));
        data_.pop();
        return res;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return data_.empty();
    }
};
```

**为什么不用 `top()` + `pop()` 分开？**

因为在这两步之间，另一个线程可能也 `pop()` 了同一个元素。这被称为 **TOCTOU（Time Of Check, Time Of Use）** 问题。解决方案：把"查看"和"移除"合并为一个原子操作。

### 6.2.2 线程安全队列

队列比栈复杂，因为涉及"队头"和"队尾"两个竞争点。

#### 方案一：单锁队列（简单但并发度低）

```cpp
template<typename T>
class SingleLockQueue {
    std::queue<T> data_;
    mutable std::mutex mtx_;
    std::condition_variable cv_;

public:
    void push(T value) {
        std::lock_guard<std::mutex> lock(mtx_);
        data_.push(std::move(value));
        cv_.notify_one();  // 通知等待的消费者
    }

    void wait_and_pop(T& result) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return !data_.empty(); });
        result = std::move(data_.front());
        data_.pop();
    }

    bool try_pop(T& result) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (data_.empty()) return false;
        result = std::move(data_.front());
        data_.pop();
        return true;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return data_.empty();
    }
};
```

**问题**：`push()` 和 `pop()` 操作的是队列不同端，理论上可以并行，但共享同一把锁导致完全串行化。

#### 方案二：双锁分离队列

**核心思想**：head 和 tail 各用一把锁，生产者和消费者互不阻塞。

```cpp
#include <memory>
#include <mutex>

template<typename T>
class DualLockQueue {
    struct Node {
        std::shared_ptr<T> data;
        std::unique_ptr<Node> next;
    };

    std::unique_ptr<Node> head_;
    Node* tail_;
    std::mutex head_mtx_;
    std::mutex tail_mtx_;

    Node* get_tail() {
        std::lock_guard<std::mutex> lock(tail_mtx_);
        return tail_;
    }

public:
    DualLockQueue() : head_(new Node), tail_(head_.get()) {}

    void push(T value) {
        auto new_data = std::make_shared<T>(std::move(value));
        auto new_node = std::make_unique<Node>();

        {
            std::lock_guard<std::mutex> lock(tail_mtx_);
            tail_->data = new_data;
            Node* const new_tail = new_node.get();
            tail_->next = std::move(new_node);
            tail_ = new_tail;
        }
    }

    std::shared_ptr<T> try_pop() {
        std::lock_guard<std::mutex> lock(head_mtx_);
        if (head_.get() == get_tail()) {
            return std::shared_ptr<T>();  // 空队列
        }
        std::shared_ptr<T> res = head_->data;
        head_ = std::move(head_->next);
        return res;
    }
};
```

**关键**：push 只锁 tail_mtx，pop 只锁 head_mtx。只有当队列几乎为空（只剩 dummy node）时才需要同时触及两边。

**生活类比**：排队买奶茶。收银员（生产者）在队尾接单，出餐台（消费者）在队头叫号。两个位置互不干扰，可以同时工作。只有队尾刚好在队头位置时（队列为空），才需要协调。

### 6.2.3 线程安全链表

单锁链表太简单，这里展示**手递手锁（Hand-over-hand Locking）**。

```cpp
template<typename T>
class LockedList {
    struct Node {
        T data;
        std::unique_ptr<Node> next;
        std::mutex mtx;
        Node(T val) : data(std::move(val)) {}
    };

    Node head_;  // 哨兵节点

public:
    LockedList() : head_(T{}) {}

    // 手递手锁查找：先锁当前节点，再锁下一节点，然后释放当前节点
    // 就像登山时确保始终有一只手抓住岩壁
    bool contains(const T& value) const {
        std::unique_lock<std::mutex> prev_lock(head_.mtx);
        Node* prev = &head_;

        while (Node* const curr = prev->next.get()) {
            std::unique_lock<std::mutex> curr_lock(curr->mtx);
            prev_lock.unlock();  // 放手 prev，抓住 curr
            if (curr->data == value) return true;
            prev = curr;
            prev_lock = std::move(curr_lock);
        }
        return false;
    }
};
```

---

## 6.3 更复杂的数据结构

### 6.3.1 线程安全哈希表（分桶锁）

每个桶（bucket）单独加锁，不同桶之间可以并行操作。

```cpp
#include <vector>
#include <list>
#include <mutex>
#include <functional>
#include <shared_mutex>  // C++17 读写锁

template<typename Key, typename Value,
         typename Hash = std::hash<Key>>
class ThreadSafeHashMap {
    struct Bucket {
        using BucketData = std::list<std::pair<Key, Value>>;
        BucketData data;
        mutable std::shared_mutex mtx;  // 读共享，写独占

        Value get(const Key& key) const {
            std::shared_lock lock(mtx);  // 读锁，多线程并发
            for (auto& [k, v] : data) {
                if (k == key) return v;
            }
            return Value{};
        }

        void set(const Key& key, Value value) {
            std::unique_lock lock(mtx);  // 写锁，独占
            for (auto& [k, v] : data) {
                if (k == key) { v = std::move(value); return; }
            }
            data.emplace_back(key, std::move(value));
        }
    };

    std::vector<Bucket> buckets_;
    Hash hasher_;

    Bucket& get_bucket(const Key& key) {
        return buckets_[hasher_(key) % buckets_.size()];
    }

public:
    ThreadSafeHashMap(size_t num_buckets = 32)
        : buckets_(num_buckets) {}

    Value get(const Key& key) const {
        return get_bucket(key).get(key);
    }

    void set(const Key& key, Value value) {
        get_bucket(key).set(key, std::move(value));
    }
};
```

**生活类比——图书馆的书架管理**：

- **一把大锁锁整个图书馆**（粗粒度锁）：一次只能进一个人，即使他只需要 3 楼的一本书，所有人都得在门外等。
- **每个书架一把锁**（分桶锁）：找小说的人集中在 A 区，查资料的人在 B 区，互不干扰。这才是真正的"并发"。

---

## 6.4 锁粒度的权衡

| 维度 | 粗粒度锁 | 细粒度锁 |
|------|---------|----------|
| 实现复杂度 | 简单 | 较高 |
| 并发度 | 低 | 高 |
| 死锁风险 | 低 | 高（需精心设计加锁顺序） |
| 锁开销（单线程） | 低 | 较高（多次加锁解锁） |
| 适用场景 | 操作简单、竞争少 | 操作复杂、高竞争 |

**经验法则**：
- 先写粗粒度版本，用性能分析工具定位瓶颈
- 只在确有必要时才细化锁
- 细化时要警惕死锁——始终按固定顺序加锁

---

## 6.5 工业场景

### 任务队列（Task Queue）

线程池的核心组件。通常使用双锁队列或多生产者多消费者无锁队列。C++ 标准库的 `std::async` 和自定义线程池都依赖它。

### LRU 缓存

最常用的实现：`std::unordered_map` + 双向链表，外层加读写锁。读多写少的场景用 `std::shared_mutex`（C++17）大幅提升吞吐。

### 并发哈希表

`folly::ConcurrentHashMap`（Facebook）、`tbb::concurrent_hash_map`（Intel TBB）、`absl::flat_hash_map`（Google Abseil）都是工业级的实现，核心思想都是分桶/分段锁。

---

## 6.6 常见坑点

1. **接口设计不当导致竞争条件**：`empty()` + `top()` + `pop()` 三步分开——经典 TOCTOU。合并为一步。

2. **异常安全问题**：`pop()` 如果先移除元素再返回，中间抛异常会导致元素丢失。先用 `shared_ptr` 持有数据，再执行修改操作。

3. **条件变量虚假唤醒**：**必须**在 `wait` 中使用带谓词的版本或显式检查循环：`cv.wait(lock, []{ return !queue.empty(); })`。

4. **死锁**：粗粒度锁也有死锁——两个对象互相持有对方需要的锁。用 `std::lock()` 同时锁多个 mutex，或始终按固定顺序加锁。

5. **锁粒度过细的低效**：频繁的加锁/解锁开销可能超过粗粒度锁的等待时间。**Profile before optimize。**

---

## 6.7 面试常问

| 问题 | 要点 |
|------|------|
| 如何设计线程安全的栈？ | 用 mutex 包裹 push/pop；合并 empty+top+pop；异常安全 |
| 如何设计线程安全的队列？ | 单锁简单；双锁提升并发；用条件变量支持阻塞等待 |
| 什么是 TOCTOU？ | 检查和使用的间隙被其他线程修改；解决：原子化操作 |
| 粗粒度锁 vs 细粒度锁 | 粗：简单，并发度低；细：复杂，并发度高，死锁风险大 |
| 如何避免死锁？ | 固定加锁顺序；使用 `std::lock()` 同时获取多锁；避免嵌套锁 |
| 为什么 `pop()` 不该返回值？ | 拷贝构造可能抛异常，导致元素已删除但数据丢失 |

---

## 我应该掌握什么

- [ ] 能写出 `empty()`、`push()`、`pop()` 线程安全且异常安全的栈实现
- [ ] 理解单锁队列与双锁队列的区别和使用场景
- [ ] 能用 `std::condition_variable` 实现阻塞等待的线程安全队列
- [ ] 理解手递手锁的工作方式及其适用场景
- [ ] 能用分桶锁实现简单的线程安全哈希表
- [ ] 能分析给定方案的锁粒度是否合理
- [ ] 知道 `pop()` 的异常安全陷阱及解决方案
- [ ] 理解 TOCTOU 问题并能避免
