#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <vector>

// ROS2 生命周期状态机
// 模拟 ros2_control 中节点的生命周期管理
//
// 状态转换图:
//   UNCONFIGURED ──on_configure──▶ INACTIVE
//        ▲                          │  ▼
//        │                          │  on_activate
//        │                          │  ▼
//        │◄──on_cleanup─── ACTIVE ──│
//        │                          │
//        │◄────on_shutdown──────────│
//        │                          │
//   FINALIZED ◄──on_shutdown────────────────

enum class LifecycleState : int {
    UNCONFIGURED = 0, // 节点刚创建，尚未分配资源
    INACTIVE = 1,     // 已配置资源，但未启动控制循环
    ACTIVE = 2,       // 控制循环运行中，提供数据输出
    FINALIZED = 3,    // 已销毁资源，不可恢复
    ERROR = 4,        // 错误状态，需要人工干预
};

// 状态名称字符串
const char *state_to_string(LifecycleState state);

// 状态转换结果
struct TransitionResult {
    bool success = false;
    LifecycleState from_state;
    LifecycleState to_state;
    std::string error_msg;
};

// 生命周期节点：实现状态转换回调
class LifecycleNode {
public:
    explicit LifecycleNode(const std::string &name);
    virtual ~LifecycleNode();

    LifecycleNode(const LifecycleNode &) = delete;
    LifecycleNode &operator=(const LifecycleNode &) = delete;

    // === 状态转换接口 ===

    // 配置节点：预分配内存、创建线程
    TransitionResult configure();

    // 激活节点：启动 RT 控制循环
    TransitionResult activate();

    // 停用节点：停止控制循环，保留资源
    TransitionResult deactivate();

    // 清理节点：释放所有资源
    TransitionResult cleanup();

    // 关闭节点：销毁节点，不可恢复
    TransitionResult shutdown();

    // === 状态查询 ===
    LifecycleState get_state() const {
        return state_.load();
    }
    const std::string &get_name() const {
        return name_;
    }
    bool is_active() const {
        return state_.load() == LifecycleState::ACTIVE;
    }

protected:
    // === 子类需要实现的状态转换回调 ===
    // 返回 true 表示成功，false 表示失败

    virtual bool on_configure() {
        return true;
    }
    virtual bool on_activate() {
        return true;
    }
    virtual bool on_deactivate() {
        return true;
    }
    virtual bool on_cleanup() {
        return true;
    }
    virtual bool on_shutdown() {
        return true;
    }

    // 状态转换时可能发生的错误处理
    virtual bool on_error(const std::string & /*error*/) {
        return true;
    }

private:
    // 验证状态转换是否合法
    TransitionResult validate_transition(LifecycleState target);

    std::string name_;
    std::atomic<LifecycleState> state_{LifecycleState::UNCONFIGURED};
};

// 生命周期管理器：协调多个 LifecycleNode
// 按依赖顺序激活/停用节点
class LifecycleManager {
public:
    LifecycleManager();
    ~LifecycleManager();

    // 添加节点，可选设置依赖 (被依赖的节点必须先激活)
    void add_node(LifecycleNode *node,
                  const std::vector<std::string> &depends_on = {});

    // 激活所有节点 (按依赖顺序)
    bool activate_all();

    // 停用所有节点 (按依赖逆序)
    bool deactivate_all();

    // 清理所有节点
    bool cleanup_all();

    // 关闭所有节点
    bool shutdown_all();

    // 获取节点的当前状态
    LifecycleState get_node_state(const std::string &name) const;

private:
    struct ManagedNode {
        LifecycleNode *node;
        std::vector<std::string> depends_on;
    };

    std::vector<ManagedNode> nodes_;
    LifecycleNode *find_node(const std::string &name);

    // 拓扑排序：返回按依赖顺序排列的节点列表
    std::vector<LifecycleNode *> topological_order();
    std::vector<LifecycleNode *> reverse_topological_order();
};

// 示例节点：带错误注入的测试节点
class TestLifecycleNode : public LifecycleNode {
public:
    TestLifecycleNode(const std::string &name,
                      bool fail_on_activate = false,
                      bool fail_on_configure = false);

protected:
    bool on_configure() override;
    bool on_activate() override;
    bool on_deactivate() override;
    bool on_cleanup() override;
    bool on_shutdown() override;

private:
    bool fail_on_activate_;
    bool fail_on_configure_;
};
