# 16_humanoid_balance — 人形机器人全身控制与平衡

## 概述

本项目实现人形机器人（34-DOF）的**全身控制（WBC）**和**平衡控制**核心算法，
聚焦于性能关键路径上的数值计算挑战。

### 核心模块

| 文件 | 功能 |
|------|------|
| `wbc_core.h/.cpp` | 层级零空间投影 WBC 求解器（伪逆、阻尼最小二乘、零空间投影） |
| `balance_control.h/.cpp` | LIPM 模型、ZMP 计算、支撑多边形判定、摩擦锥约束 |
| `task_stack.h/.cpp` | 34-DOF 人形机器人任务栈构建器（P0-P4 优先级） |
| `main.cpp` | 4 项演示 + 性能指标输出 |

### 任务优先级栈

```
P0: 关节限位（硬约束）     — 接近限位时激活，不可违反
P1: 平衡（CoM/ZMP）       — 保持 ZMP 在支撑多边形内
P2: 足底接触              — 支撑脚静止 + 摩擦锥约束
P3: 摆动脚 + 手末端       — 步态跟踪 + 操作任务
P4: 姿态                  — 维持直立站立姿态
```

## 推荐阅读顺序

1. **`wbc_core.h` + `wbc_core.cpp`** — WBC 求解器：定义核心数学工具（伪逆、阻尼最小二乘、零空间投影）
2. **`balance_control.h` + `balance_control.cpp`** — LIPM 模型、ZMP 计算、支撑多边形、摩擦锥约束
3. **`task_stack.h` + `task_stack.cpp`** — 构建 P0→P4 任务栈的构建器，组合 WBC 求解器和平衡控制
4. **`main.cpp`** — 最后阅读，演示 1(WBC)→2(ZMP/LIPM)→3(完整平衡回路)→4(可扩展性)

## 构建

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./humanoid_balance
```

### 依赖

- **C++17** 编译器（GCC 9+ / Clang 10+）
- **CMake >= 3.14**
- **Eigen3**（`sudo apt install libeigen3-dev`）
- **Linux**（pthread）

## 运行演示

| 演示 | 内容 |
|------|------|
| 1 | WBC 求解器验证（34-DOF，4 任务，1000 次迭代测量） |
| 2 | ZMP 计算 + LIPM 步行模式生成 + 摩擦锥验证 |
| 3 | 完整平衡控制回路模拟（100ms 站立 + 100ms 迈步） |
| 4 | 可扩展性测试（7 → 14 → 28 → 34 DOF 求解时间曲线） |

运行结束后自动生成 `balance_metrics.json` 性能指标文件。

## 关键算法

### 层级零空间投影

```
N₀ = I,  q̇₀ = 0
for k = 0, 1, 2, ...:
    Ĵₖ = Jₖ · Nₖ₋₁              (增广雅可比)
    q̇ₖ = q̇ₖ₋₁ + Ĵₖ⁺(vₖ − Jₖq̇ₖ₋₁)  (速度增量)
    Nₖ = Nₖ₋₁ − Ĵₖ⁺ Ĵₖ           (零空间更新)
```

### 阻尼最小二乘伪逆

```
J⁺ = Jᵀ (J Jᵀ + λ² I)⁻¹
```

### LIPM ZMP 公式

```
p_zmp = x − (z_c / g) · ẍ
  ẍ  = (g / z_c) · (x − p_zmp)
```

## 性能目标

- **WBC 求解**：< 500μs（34-DOF，6 个任务）
- **ZMP 检查**：< 1μs（点-多边形射线法）
- **控制回路**：1kHz（1ms 周期）
- **截止时间违约率**：< 1%
