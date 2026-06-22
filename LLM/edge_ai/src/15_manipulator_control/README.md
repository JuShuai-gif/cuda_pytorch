# 15_manipulator_control — 机械臂实时控制性能分析

7-DOF 机械臂的完整运动学 + 轨迹规划 + 1kHz 关节控制闭环，聚焦实时性能。

## 功能模块

| 模块 | 文件 | 功能 |
|------|------|------|
| **运动学** | `kinematics.h/.cpp` | DH 正运动学、6×7 几何雅可比、DLS/NR 逆运动学、SVD/Cholesky、奇异点检测 |
| **轨迹生成** | `trajectory.h/.cpp` | 梯形速度曲线、S 曲线(7段)、三次样条 via-point、1kHz 在线插补 |
| **关节控制** | `joint_controller.h/.cpp` | PID+抗饱和、前馈力矩、双积分器关节模型、无锁双缓冲共享状态、1kHz 控制闭环 |
| **主程序** | `main.cpp` | 5 个演示：FK 验证、IK 收敛测试、轨迹生成、控制闭环、奇异点检测 |

## 文件结构

```
15_manipulator_control/
|-- kinematics.h            # DH 参数、Mat4、运动学声明
|-- kinematics.cpp          # FK、Jacobian、DLS/NR IK、SVD、Cholesky 实现
|-- trajectory.h            # 轨迹类型声明
|-- trajectory.cpp          # 梯形/S 曲线/样条实现
|-- joint_controller.h      # PID、关节控制器、实时循环声明
|-- joint_controller.cpp    # PID+抗饱和、双积分器、原子共享状态实现
|-- main.cpp                # 入口 + 5 个演示
|-- CMakeLists.txt
|-- README.md
```

## 推荐阅读顺序

1. **`kinematics.h` + `kinematics.cpp`** — 基础：DH 正运动学是 IK 和其他一切的前提，所有组件均依赖运动学引擎
2. **`trajectory.h` + `trajectory.cpp`** — 轨迹生成（梯形/S 曲线/三次样条），建立在运动学之上但独立于控制器
3. **`joint_controller.h` + `joint_controller.cpp`** — 控制闭环（PID+抗饱和+前馈），使用运动学和轨迹进行插补
4. **`main.cpp`** — 最后阅读，演示 1(FK)→2(IK)→3(轨迹)→4(控制)→5(奇异点)，正是自然的阅读顺序

## 构建

```bash
cd edge_ai/src/15_manipulator_control
mkdir build && cd build
cmake ..
make -j$(nproc)
```

**无需外部依赖**。所有 4×4/6×7/7×7 矩阵运算内联实现，不使用 Eigen3。

## 运行

```bash
./manipulator_control
```

输出 5 个演示的结果，并将指标写入 `manipulator_metrics.json`。

## 演示说明

### 1. 正运动学验证
零位配置和典型工作配置的末端位姿计算，验证 DH 参数链条正确性。

### 2. IK 收敛测试
200 组随机目标位姿，对比 DLS (Cholesky) 和 NR (SVD 伪逆) 的：
- 收敛率
- P50/P99 收敛时间
- 最终位姿误差

### 3. 轨迹生成
- 梯形速度曲线：验证速度/加速度限制
- S 曲线：7 段 jerk-limited 轮廓
- 三次样条：通过 5 个路径点的平滑插值

### 4. 实时控制闭环
7-DOF PID 控制模拟 1kHz 循环 500ms，统计：
- 错过截止时间（>1ms）的次数
- 平均/最大/最小/P99 循环时间
- 有/无前馈力矩对比

### 5. 奇异点检测
正常 vs 全伸展配置的雅可比最小奇异值对比。

## 技术要点

- **DLS IK**：使用 Cholesky 分解解 (JᵀJ + λ²I)Δθ = Jᵀe，避免完整 SVD 开销
- **NR IK**：使用 Jacobi SVD 计算伪逆，用于对比 DLS 的稳定性和速度
- **零动态分配**：所有工作数组使用栈上的固定大小数组（`std::array`/`double[49]`）
- **无锁共享状态**：双缓冲 + `std::atomic` 指针交换，感知线程（低频率）与控制线程（1kHz）解耦
- **1kHz 截止时间**：控制循环严格计时，报告超期比例

## 学习笔记

详见 `../../../notes/15_manipulator_control.md`（中文，~4000 字），涵盖：
- 运动学基础与 DH 参数
- 数值 IK 算法对比（NR vs DLS）
- 轨迹规划理论（梯形/S 曲线/样条）
- 实时控制闭环架构
- 性能优化实践
