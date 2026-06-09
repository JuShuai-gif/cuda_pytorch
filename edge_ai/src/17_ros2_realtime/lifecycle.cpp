#include "lifecycle.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <queue>
#include <set>

// ============================================================
// 辅助函数
// ============================================================

const char *state_to_string(LifecycleState state) {
    switch (state) {
    case LifecycleState::UNCONFIGURED: return "UNCONFIGURED";
    case LifecycleState::INACTIVE: return "INACTIVE";
    case LifecycleState::ACTIVE: return "ACTIVE";
    case LifecycleState::FINALIZED: return "FINALIZED";
    case LifecycleState::ERROR: return "ERROR";
    default: return "UNKNOWN";
    }
}

// ============================================================
// LifecycleNode 实现
// ============================================================

LifecycleNode::LifecycleNode(const std::string &name) : name_(name) {
}

LifecycleNode::~LifecycleNode() {
    if (state_.load() != LifecycleState::FINALIZED) {
        shutdown();
    }
}

TransitionResult LifecycleNode::validate_transition(LifecycleState target) {
    TransitionResult result;
    result.from_state = state_.load();

    bool valid = false;
    switch (target) {
    case LifecycleState::INACTIVE:
        // UNCONFIGURED -> INACTIVE, ACTIVE -> INACTIVE (via deactivate)
        valid = (result.from_state == LifecycleState::UNCONFIGURED
                 || result.from_state == LifecycleState::ACTIVE);
        break;
    case LifecycleState::ACTIVE:
        // 只能从 INACTIVE 激活
        valid = (result.from_state == LifecycleState::INACTIVE);
        break;
    case LifecycleState::FINALIZED:
        // 可以从任意状态关闭 (除了已关闭)
        valid = (result.from_state != LifecycleState::FINALIZED);
        break;
    case LifecycleState::UNCONFIGURED:
        // INACTIVE -> UNCONFIGURED (via cleanup)
        valid = (result.from_state == LifecycleState::INACTIVE);
        break;
    default:
        break;
    }

    if (!valid) {
        result.success = false;
        result.to_state = result.from_state;
        result.error_msg = "非法状态转换: "
                           + std::string(state_to_string(result.from_state))
                           + " -> "
                           + std::string(state_to_string(target));
        printf("[%s] ❌ %s\n", name_.c_str(), result.error_msg.c_str());
        return result;
    }

    result.success = true;
    result.to_state = target;
    return result;
}

TransitionResult LifecycleNode::configure() {
    auto result = validate_transition(LifecycleState::INACTIVE);
    if (!result.success) return result;

    printf("[%s] 正在配置... (UNCONFIGURED -> INACTIVE)\n", name_.c_str());

    if (!on_configure()) {
        result.success = false;
        result.to_state = state_.load();
        result.error_msg = "配置失败";
        printf("[%s] ❌ 配置失败，保持在 %s\n", name_.c_str(),
               state_to_string(state_.load()));
        return result;
    }

    state_.store(LifecycleState::INACTIVE, std::memory_order_release);
    printf("[%s] ✅ 配置完成，当前状态: INACTIVE\n", name_.c_str());
    return result;
}

TransitionResult LifecycleNode::activate() {
    auto result = validate_transition(LifecycleState::ACTIVE);
    if (!result.success) return result;

    printf("[%s] 正在激活... (INACTIVE -> ACTIVE)\n", name_.c_str());

    if (!on_activate()) {
        // 激活失败：保持 INACTIVE 状态
        result.success = false;
        result.to_state = LifecycleState::INACTIVE;
        result.error_msg = "激活失败，回退到 INACTIVE";
        printf("[%s] ❌ %s\n", name_.c_str(), result.error_msg.c_str());
        on_error(result.error_msg);
        return result;
    }

    state_.store(LifecycleState::ACTIVE, std::memory_order_release);
    printf("[%s] ✅ 已激活，当前状态: ACTIVE\n", name_.c_str());
    return result;
}

TransitionResult LifecycleNode::deactivate() {
    auto result = validate_transition(LifecycleState::INACTIVE);
    if (!result.success) return result;

    printf("[%s] 正在停用... (ACTIVE -> INACTIVE)\n", name_.c_str());

    if (!on_deactivate()) {
        result.success = false;
        result.to_state = LifecycleState::ACTIVE;
        result.error_msg = "停用失败";
        printf("[%s] ❌ %s\n", name_.c_str(), result.error_msg.c_str());
        return result;
    }

    state_.store(LifecycleState::INACTIVE, std::memory_order_release);
    printf("[%s] ✅ 已停用，当前状态: INACTIVE\n", name_.c_str());
    return result;
}

TransitionResult LifecycleNode::cleanup() {
    if (state_.load() != LifecycleState::INACTIVE) {
        TransitionResult result;
        result.success = false;
        result.from_state = state_.load();
        result.to_state = state_.load();
        result.error_msg = "只能从 INACTIVE 状态清理";
        printf("[%s] ❌ %s\n", name_.c_str(), result.error_msg.c_str());
        return result;
    }

    printf("[%s] 正在清理... (INACTIVE -> UNCONFIGURED)\n", name_.c_str());

    if (!on_cleanup()) {
        TransitionResult result;
        result.success = false;
        result.from_state = LifecycleState::INACTIVE;
        result.to_state = LifecycleState::INACTIVE;
        result.error_msg = "清理失败";
        printf("[%s] ❌ %s\n", name_.c_str(), result.error_msg.c_str());
        return result;
    }

    state_.store(LifecycleState::UNCONFIGURED, std::memory_order_release);
    printf("[%s] ✅ 清理完成，当前状态: UNCONFIGURED\n", name_.c_str());
    return {true, LifecycleState::INACTIVE, LifecycleState::UNCONFIGURED, ""};
}

TransitionResult LifecycleNode::shutdown() {
    if (state_.load() == LifecycleState::FINALIZED) {
        return {true, LifecycleState::FINALIZED, LifecycleState::FINALIZED, "已关闭"};
    }

    LifecycleState prev = state_.load();
    printf("[%s] 正在关闭... (%s -> FINALIZED)\n",
           name_.c_str(), state_to_string(prev));

    // 如果是 ACTIVE，先停用
    if (prev == LifecycleState::ACTIVE) {
        on_deactivate();
    }

    on_shutdown();
    state_.store(LifecycleState::FINALIZED, std::memory_order_release);
    printf("[%s] ✅ 已关闭，当前状态: FINALIZED\n", name_.c_str());
    return {true, prev, LifecycleState::FINALIZED, ""};
}

// ============================================================
// LifecycleManager 实现
// ============================================================

LifecycleManager::LifecycleManager() = default;
LifecycleManager::~LifecycleManager() = default;

void LifecycleManager::add_node(LifecycleNode *node,
                                const std::vector<std::string> &depends_on) {
    nodes_.push_back({node, depends_on});
}

LifecycleNode *LifecycleManager::find_node(const std::string &name) {
    for (auto &mn : nodes_) {
        if (mn.node->get_name() == name) return mn.node;
    }
    return nullptr;
}

std::vector<LifecycleNode *> LifecycleManager::topological_order() {
    std::vector<LifecycleNode *> result;
    std::set<LifecycleNode *> visited;
    std::set<LifecycleNode *> visiting; // 检测循环依赖

    std::function<bool(LifecycleNode *)> dfs = [&](LifecycleNode *node) -> bool {
        if (visited.count(node)) return true;
        if (visiting.count(node)) {
            printf("[LifecycleManager] ❌ 检测到循环依赖: %s\n",
                   node->get_name().c_str());
            return false;
        }

        visiting.insert(node);

        // 先处理依赖
        for (auto &mn : nodes_) {
            if (mn.node == node) {
                for (const auto &dep_name : mn.depends_on) {
                    LifecycleNode *dep = find_node(dep_name);
                    if (dep && !dfs(dep)) return false;
                }
                break;
            }
        }

        visiting.erase(node);
        visited.insert(node);
        result.push_back(node);
        return true;
    };

    for (auto &mn : nodes_) {
        if (!dfs(mn.node)) return {};
    }

    return result;
}

std::vector<LifecycleNode *> LifecycleManager::reverse_topological_order() {
    auto order = topological_order();
    std::reverse(order.begin(), order.end());
    return order;
}

bool LifecycleManager::activate_all() {
    printf("\n=== 激活所有节点 (按依赖顺序) ===\n");

    auto order = topological_order();
    if (order.empty()) {
        printf("[LifecycleManager] ❌ 拓扑排序失败\n");
        return false;
    }

    // 第一步：配置所有节点
    for (auto *node : order) {
        auto result = node->configure();
        if (!result.success) {
            printf("[LifecycleManager] ❌ 配置失败: %s\n",
                   node->get_name().c_str());
            // 清理已配置的节点
            for (auto *n : order) {
                if (n == node) break;
                n->cleanup();
            }
            return false;
        }
    }

    // 第二步：激活所有节点
    for (auto *node : order) {
        auto result = node->activate();
        if (!result.success) {
            printf("[LifecycleManager] ❌ 激活失败: %s\n",
                   node->get_name().c_str());
            // 停用并清理已激活的节点
            for (auto *n : order) {
                if (n == node) break;
                n->deactivate();
            }
            for (auto *n : order) {
                if (n == node) break;
                n->cleanup();
            }
            return false;
        }
    }

    printf("[LifecycleManager] ✅ 所有节点已激活\n");
    return true;
}

bool LifecycleManager::deactivate_all() {
    printf("\n=== 停用所有节点 (按依赖逆序) ===\n");

    auto order = reverse_topological_order();
    for (auto *node : order) {
        if (node->get_state() == LifecycleState::ACTIVE) {
            node->deactivate();
        }
    }
    return true;
}

bool LifecycleManager::cleanup_all() {
    printf("\n=== 清理所有节点 ===\n");

    auto order = reverse_topological_order();
    for (auto *node : order) {
        if (node->get_state() == LifecycleState::INACTIVE) {
            node->cleanup();
        }
    }
    return true;
}

bool LifecycleManager::shutdown_all() {
    printf("\n=== 关闭所有节点 ===\n");

    auto order = reverse_topological_order();
    for (auto *node : order) {
        node->shutdown();
    }
    return true;
}

LifecycleState LifecycleManager::get_node_state(const std::string &name) const {
    for (auto &mn : nodes_) {
        if (mn.node->get_name() == name) return mn.node->get_state();
    }
    return LifecycleState::FINALIZED;
}

// ============================================================
// TestLifecycleNode 实现
// ============================================================

TestLifecycleNode::TestLifecycleNode(const std::string &name,
                                     bool fail_on_activate,
                                     bool fail_on_configure) : LifecycleNode(name), fail_on_activate_(fail_on_activate), fail_on_configure_(fail_on_configure) {
}

bool TestLifecycleNode::on_configure() {
    if (fail_on_configure_) {
        printf("[%s] ℹ  模拟配置失败 (硬件不可用)\n", get_name().c_str());
        return false;
    }
    printf("[%s] ℹ  配置：预分配内存、创建线程\n", get_name().c_str());
    return true;
}

bool TestLifecycleNode::on_activate() {
    if (fail_on_activate_) {
        printf("[%s] ℹ  模拟激活失败 (驱动加载失败)\n", get_name().c_str());
        return false;
    }
    printf("[%s] ℹ  激活：启动 RT 控制循环\n", get_name().c_str());
    return true;
}

bool TestLifecycleNode::on_deactivate() {
    printf("[%s] ℹ  停用：停止控制循环，保留资源\n", get_name().c_str());
    return true;
}

bool TestLifecycleNode::on_cleanup() {
    printf("[%s] ℹ  清理：释放所有资源\n", get_name().c_str());
    return true;
}

bool TestLifecycleNode::on_shutdown() {
    printf("[%s] ℹ  关闭：销毁节点\n", get_name().c_str());
    return true;
}
