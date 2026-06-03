# Chapter 06 - 无锁并发数据结构设计

C++ Concurrency in Action 第6章示例代码。

## 内容概览

| 文件 | 主题 | 关键知识点 |
|------|------|-----------|
| `01_threadsafe_stack.cpp` | 线程安全栈 | 合并 empty+top+pop 消除竞态；shared_ptr 返回避免异常 |
| `02_threadsafe_queue.cpp` | 线程安全队列（单锁） | 一把 mutex + condition_variable；阻塞/非阻塞接口 |
| `03_threadsafe_queue_fine.cpp` | 分离锁队列 | 头尾各一把锁，push/pop 并发；unique_ptr 节点管理 |
| `04_threadsafe_lookup_table.cpp` | 分桶锁哈希表 | 每个桶独立 shared_mutex；读写锁分离高并发 |
| `05_threadsafe_list.cpp` | 线程安全链表 | 手递手加锁（细粒度）；shared_mutex 全锁（粗粒度）对比 |

## 编译运行

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
./01_threadsafe_stack
./02_threadsafe_queue
./03_threadsafe_queue_fine
./04_threadsafe_lookup_table
./05_threadsafe_list
```
