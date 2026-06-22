#!/usr/bin/env python3
"""
MIT 6.5940 第23讲：量子机器学习模拟

涵盖主题：
  - 使用 numpy 实现简单的参数化量子电路（PQC）模拟器
  - 构建量子二分类器（将量子比特模拟为复向量）
  - 展示纠缠和表达能力的概念
  - 说明量子优势仍处于理论阶段（代码注释中讨论其局限性）

所有计算均在 CPU 上运行，无需 GPU。使用 numpy 对复值态向量进行线性代数运算。
"""

from __future__ import annotations

import math
from typing import List, Tuple, Callable

import numpy as np


# ===========================================================================
# 1. 量子门定义
# ===========================================================================

# 泡利矩阵
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)  # σ_x: 比特翻转
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)  # σ_y
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)  # σ_z: 相位翻转
HADAMARD = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(
    2
)  # 哈达玛门：产生叠加态

# 2x2 单位矩阵
I2 = np.eye(2, dtype=complex)


def rotation_x(theta: float) -> np.ndarray:
    """绕 X 轴旋转门：exp(-i * theta/2 * σ_x)。"""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def rotation_y(theta: float) -> np.ndarray:
    """绕 Y 轴旋转门：exp(-i * theta/2 * σ_y)。"""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rotation_z(theta: float) -> np.ndarray:
    """绕 Z 轴旋转门：exp(-i * theta/2 * σ_z)。"""
    return np.array(
        [[math.e ** (-1j * theta / 2), 0], [0, math.e ** (1j * theta / 2)]],
        dtype=complex,
    )


def cnot_gate() -> np.ndarray:
    """受控非门（CNOT，4x4 矩阵）：控制比特为 |1> 时翻转目标比特。"""
    return np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )


# ===========================================================================
# 2. 张量积工具函数
# ===========================================================================


def kronecker_product(*mats: np.ndarray) -> np.ndarray:
    """计算多个矩阵的克罗内克积（Kronecker product）。

    对于量子电路，这用于将单量子比特门扩展到完整的希尔伯特空间。
    """
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(result, m)
    return result


def apply_gate(
    state: np.ndarray, gate: np.ndarray, target_qubits: List[int], num_qubits: int
) -> np.ndarray:
    """将量子门作用于态向量的特定量子比特上。

    通过在非目标量子比特上与单位矩阵做张量积，
    构建完整的 2^n × 2^n 算子。

    参数:
        state: 形状为 (2**num_qubits,) 的态向量
        gate: 作用于目标量子比特的门矩阵
        target_qubits: 门作用的量子比特索引
        num_qubits: 量子比特总数

    返回:
        更新后的态向量。
    """
    # 使用张量积构建完整的算子
    full_op = np.eye(1, dtype=complex)
    for q in range(num_qubits):
        if q in target_qubits:
            # 找到门中的位置
            pos = target_qubits.index(q)
            # 对于多量子比特门，需要正确的排序
            full_op = np.kron(full_op, np.eye(2, dtype=complex))
        else:
            full_op = np.kron(full_op, I2)

    # 备选方案：使用基于置换的方法以保证正确性
    # 我们将正确地构建完整算子
    return _apply_gate_correct(state, gate, target_qubits, num_qubits)


def _apply_gate_correct(
    state: np.ndarray, gate: np.ndarray, targets: List[int], n: int
) -> np.ndarray:
    """通过排列量子比特来正确应用量子门。

    策略:
      1. 置换态向量，使目标量子比特成为最高有效位
      2. 应用门（在剩余量子比特上与单位矩阵做张量积）
      3. 逆置换还原
    """
    if len(targets) == 1:
        # 单量子比特门：更简单的路径
        full_gate = np.eye(1, dtype=complex)
        for q in range(n):
            if q in targets:
                full_gate = np.kron(full_gate, gate)  # 目标比特上放置门
            else:
                full_gate = np.kron(full_gate, I2)  # 其他比特上放置单位矩阵
        return full_gate @ state

    elif len(targets) == 2 and gate.shape == (4, 4):
        # 双量子比特门（例如 CNOT）
        full_gate = np.eye(1, dtype=complex)
        for q in range(n):
            if q == targets[0]:
                # 从此处开始构建双量子比特算子
                full_gate = np.kron(full_gate, gate)
                # 跳过下一个目标比特（已包含在双量子比特门中）
                continue
            elif q == targets[1]:
                # 已在上面处理
                continue
            else:
                full_gate = np.kron(full_gate, I2)
        return full_gate @ state

    else:
        raise ValueError(
            f"不支持的门形状 {gate.shape}，作用于 {len(targets)} 个目标比特"
        )


# ===========================================================================
# 3. 参数化量子电路（PQC）
# ===========================================================================


class ParameterizedQuantumCircuit:
    """带有 Ry 旋转和 CNOT 纠缠层的简单 PQC。

    架构:
      - 每个量子比特上的 Ry(θ_i) 旋转层
      - CNOT 梯式纠缠（近邻连接）
      - 重复 L 次以提高表达能力
    """

    def __init__(self, num_qubits: int, num_layers: int = 2):
        """
        参数:
            num_qubits: 电路中的量子比特数
            num_layers: 旋转+纠缠的层数
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        # 每层参数：每个量子比特一个旋转角度
        self.num_params = num_qubits * num_layers

    def _build_circuit(self, params: np.ndarray) -> np.ndarray:
        """构建表示整个电路的完整酉矩阵。

        参数:
            params: 形状为 (num_params,) 的 numpy 数组 -- 旋转角度

        返回:
            形状为 (2**n, 2**n) 的酉矩阵。
        """
        n = self.num_qubits
        dim = 2**n
        U = np.eye(dim, dtype=complex)  # 从单位矩阵开始

        for layer in range(self.num_layers):
            # 旋转层：对每个量子比特应用 Ry 旋转
            for q in range(n):
                idx = layer * n + q
                theta = params[idx] if idx < len(params) else 0.0
                Ry_q = rotation_y(theta)
                full_Ry = np.eye(1, dtype=complex)
                for qi in range(n):
                    if qi == q:
                        full_Ry = np.kron(full_Ry, Ry_q)
                    else:
                        full_Ry = np.kron(full_Ry, I2)
                U = full_Ry @ U  # 左乘表示时间上靠后的操作

            # 纠缠层：CNOT 梯式连接
            for q in range(n - 1):
                cnot_full = np.eye(1, dtype=complex)
                for qi in range(n):
                    if qi == q:
                        # 在位置 q 和 q+1 上放置 CNOT 门
                        cnot_full = np.kron(cnot_full, cnot_gate())
                        continue
                    elif qi == q + 1:
                        continue  # 已经包含在 CNOT 中
                    else:
                        cnot_full = np.kron(cnot_full, I2)
                U = cnot_full @ U

        return U

    def run(
        self, params: np.ndarray, initial_state: np.ndarray | None = None
    ) -> np.ndarray:
        """在初始态上执行电路。

        参数:
            params: 旋转角度 (num_params,)
            initial_state: 初始态向量（默认: |0...0>）

        返回:
            最终态向量。
        """
        if initial_state is None:
            # 默认初始化为全零态 |0...0>
            initial_state = np.zeros(2**self.num_qubits, dtype=complex)
            initial_state[0] = 1.0

        U = self._build_circuit(params)
        return U @ initial_state

    def measure_expectation(self, params: np.ndarray, observable: str = "Z") -> float:
        """测量第一个量子比特上可观测量的期望值。

        参数:
            params: 旋转角度
            observable: "Z" 表示量子比特 0 上的泡利-Z 测量

        返回:
            期望值 <Z_0>，范围在 [-1, 1] 之间。
        """
        state = self.run(params)
        n = self.num_qubits

        # 在量子比特 0 上构建完整的 Z 可观测量
        Z_full = np.eye(1, dtype=complex)
        for q in range(n):
            if q == 0:
                Z_full = np.kron(Z_full, PAULI_Z)
            else:
                Z_full = np.kron(Z_full, I2)

        # 期望值 = <ψ|Z|ψ>，取实部
        expectation = (state.conj().T @ Z_full @ state).real
        return float(expectation)


# ===========================================================================
# 4. 量子二分类器
# ===========================================================================


class QuantumBinaryClassifier:
    """使用 PQC 作为特征映射的简单量子二分类器。

    核心思想：将经典数据编码为旋转角度，通过 PQC 处理，
    然后测量一个量子比特以获得二分类预测。
    """

    def __init__(self, num_qubits: int = 3, num_layers: int = 2):
        self.num_qubits = num_qubits
        self.circuit = ParameterizedQuantumCircuit(num_qubits, num_layers)
        # 可训练参数，随机初始化在 [0, 2π) 范围内
        self.params = np.random.uniform(0, 2 * math.pi, self.circuit.num_params)

    def _encode_data(self, features: np.ndarray) -> np.ndarray:
        """将经典特征编码为旋转参数。

        使用角度编码：每个特征通过 arctan 或直接缩放映射为旋转角度。

        参数:
            features: 经典特征向量

        返回:
            电路参数（编码角度 + 可训练参数的拼接）。
        """
        # 归一化并映射到角度范围
        norm = np.linalg.norm(features) + 1e-8
        encoded = features / norm * math.pi
        # 填充/截断以匹配 num_params
        if len(encoded) < self.circuit.num_params:
            encoded = np.pad(encoded, (0, self.circuit.num_params - len(encoded)))
        else:
            encoded = encoded[: self.circuit.num_params]
        # 与可训练参数组合（加法编码）
        return encoded + self.params

    def predict(self, features: np.ndarray) -> int:
        """二分类预测：返回 0 或 1。

        使用量子比特 0 上 Z 的期望值：sign(expectation) -> 类别。
        """
        params = self._encode_data(features)
        exp = self.circuit.measure_expectation(params, "Z")
        return 1 if exp > 0 else 0

    def predict_proba(self, features: np.ndarray) -> float:
        """返回 [0, 1] 范围内类似概率的分数。"""
        params = self._encode_data(features)
        exp = self.circuit.measure_expectation(params, "Z")
        # 将 [-1, 1] 映射到 [0, 1]
        return float((exp + 1) / 2)


# ===========================================================================
# 5. 纠缠演示
# ===========================================================================


def demonstrate_entanglement() -> None:
    """演示量子电路如何创建纠缠态。

    创建贝尔态：(|00> + |11>) / sqrt(2)，然后展示
    测量一个量子比特如何决定另一个量子比特的状态。
    """
    print("  创建贝尔态：对 q0 施加 H，然后 CNOT(q0, q1)")
    # 从 |00> 开始
    state = np.zeros(4, dtype=complex)
    state[0] = 1.0

    # 对 q0 施加哈达玛门：H ⊗ I |00> → (|00> + |10>) / √2
    H0 = np.kron(HADAMARD, I2)
    state = H0 @ state

    # 以 q0 为控制比特、q1 为目标比特施加 CNOT：CNOT @ (H ⊗ I) |00> → 贝尔态
    CNOT = cnot_gate()  # |00>,|01>,|10>,|11> 基 — 控制比特 q0，目标比特 q1
    state = CNOT @ state

    print(f"  贝尔态: {np.round(state, 4)}")
    probs = np.abs(state) ** 2  # 测量概率 = |振幅|^2
    print(
        f"  测量概率: |00>={probs[0]:.2f}, "
        f"|01>={probs[1]:.2f}, |10>={probs[2]:.2f}, |11>={probs[3]:.2f}"
    )
    print("  -> 量子比特处于纠缠态：测量 q0=0 则必然 q1=0，反之亦然。")


# ===========================================================================
# 6. 表达能力分析
# ===========================================================================


def analyze_expressivity(
    num_qubits: int, num_layers_list: List[int], num_samples: int = 200
) -> None:
    """分析不同层数的 PQC 的表达能力。

    表达能力：电路在随机参数下产生多样化输出态的能力。
    通过期望值在随机参数上的方差来衡量。

    参数:
        num_qubits: 量子比特数
        num_layers_list: 要测试的层数列表
        num_samples: 每种配置的随机参数采样数
    """
    print(f"\n  表达能力分析（{num_qubits} 个量子比特）：")
    print(f"  {'层数':>6} {'<Z> 均值':>10} {'<Z> 标准差':>10} {'态空间':>12}")
    print(f"  {'-' * 40}")

    for layers in num_layers_list:
        circuit = ParameterizedQuantumCircuit(num_qubits, layers)
        expectations = []
        for _ in range(num_samples):
            params = np.random.uniform(0, 2 * math.pi, circuit.num_params)
            exp = circuit.measure_expectation(params, "Z")
            expectations.append(exp)
        expectations = np.array(expectations)
        print(
            f"  {layers:>6} {expectations.mean():>10.4f} {expectations.std():>10.4f} "
            f"{'2^' + str(num_qubits) + ' 希尔伯特空间':>12}"
        )
        print(
            f"          -> 更多层数 = 更高的表达能力（更大的标准差） "
            f"[当前标准差={expectations.std():.3f}]"
        )


# ===========================================================================
# 7. 经典模拟代价分析
# ===========================================================================


def analyze_scaling():
    """分析经典计算机模拟量子电路的指数级代价。

    这是量子优势仍属理论范畴的核心论据：
    经典模拟需要 O(2^n) 的内存和时间。
    """
    print("\n--- 经典模拟代价分析 ---")
    print("  模拟 n 个量子比特需要存储一个 2^n 的复向量。")
    print("  （每个元素使用 complex128 类型需 16 字节）")
    print()
    print(f"  {'量子比特':>6} {'态大小':>10} {'内存':>12} {'可行?':>10}")
    print(f"  {'-' * 42}")
    for n in [5, 10, 15, 20, 25, 30, 35, 40, 50]:
        dim = 2**n  # 希尔伯特空间维度
        mem_bytes = dim * 16  # 每个复数值 16 字节
        # 格式化内存大小，选择合适的单位
        if mem_bytes < 1024:
            mem_str = f"{mem_bytes} B"
        elif mem_bytes < 1024**2:
            mem_str = f"{mem_bytes / 1024:.1f} KB"
        elif mem_bytes < 1024**3:
            mem_str = f"{mem_bytes / 1024**2:.1f} MB"
        elif mem_bytes < 1024**4:
            mem_str = f"{mem_bytes / 1024**3:.1f} GB"
        else:
            mem_str = f"{mem_bytes / 1024**4:.1f} TB"
        feasible = "Yes" if n <= 30 else "No"  # 约 30 量子比特是经典模拟的极限
        print(f"  {n:>6} {dim:>10,} {mem_str:>12} {feasible:>10}")


# ===========================================================================
# 8. 主演示
# ===========================================================================


def main() -> None:
    """主入口：运行量子机器学习模拟演示。"""
    print("=" * 72)
    print("MIT 6.5940 第23讲：量子机器学习模拟")
    print("=" * 72)

    # ---------- 量子门展示 ----------
    print("\n--- 1. 量子门 ---")
    print(f"  哈达玛门:\n{HADAMARD}")
    print(f"  CNOT 门:\n{cnot_gate()}")
    print(f"  R_y(π/4):\n{np.round(rotation_y(math.pi / 4), 4)}")

    # ---------- 单量子比特旋转 ----------
    print("\n--- 2. 单量子比特态演化 ---")
    state0 = np.array([1.0, 0.0], dtype=complex)  # 初始态 |0>
    for angle in [0, math.pi / 4, math.pi / 2, math.pi]:
        Ry = rotation_y(angle)
        final = Ry @ state0
        prob0 = abs(final[0]) ** 2  # 测量到 |0> 的概率
        prob1 = abs(final[1]) ** 2  # 测量到 |1> 的概率
        print(
            f"  R_y({angle:.2f})|0> -> [{final[0]:.3f}, {final[1]:.3f}], "
            f"p(0)={prob0:.3f}, p(1)={prob1:.3f}"
        )

    # ---------- 纠缠 ----------
    print("\n--- 3. 纠缠演示 ---")
    demonstrate_entanglement()

    # ---------- PQC 和二分类器 ----------
    print("\n--- 4. 参数化量子电路 ---")
    pqc = ParameterizedQuantumCircuit(num_qubits=3, num_layers=2)
    print(f"  量子比特数: {pqc.num_qubits}, 层数: {pqc.num_layers}")
    print(f"  可训练参数: {pqc.num_params}")

    # 示例电路执行
    params = np.array([0.5, 1.0, 0.3, 0.8, 0.2, 1.5])
    final_state = pqc.run(params)
    print(f"  输入参数: {params}")
    print(f"  最终态（前4个振幅）: {np.round(final_state[:4], 4)}")
    exp_z = pqc.measure_expectation(params, "Z")
    print(f"  <Z_0> = {exp_z:.4f}")

    # ---------- 量子二分类器 ----------
    print("\n--- 5. 量子二分类器 ---")
    qbc = QuantumBinaryClassifier(num_qubits=3, num_layers=2)
    print(f"  可训练参数: {qbc.circuit.num_params}")

    # 在随机特征上测试
    test_features = np.array(
        [
            [1.0, 0.5, -0.3],
            [0.2, -0.8, 0.6],
            [-1.0, 0.2, 0.9],
            [0.7, -0.1, -0.5],
        ]
    )
    print(f"  {'特征':<25} {'预测':>10} {'分数':>8}")
    print(f"  {'-' * 45}")
    for feat in test_features:
        pred = qbc.predict(feat)
        proba = qbc.predict_proba(feat)
        print(f"  {str(np.round(feat, 2)):<25} {pred:>10} {proba:>8.4f}")

    # ---------- 表达能力 ----------
    print("\n--- 6. 表达能力分析 ---")
    analyze_expressivity(num_qubits=3, num_layers_list=[1, 2, 4, 8])

    # ---------- 经典代价 ----------
    analyze_scaling()

    # ---------- 局限性讨论 ----------
    print("\n--- 7. 讨论：量子优势的局限性 ---")
    print("""
  量子机器学习是一个令人着迷但尚处萌芽阶段的领域。当前局限：

  1. 噪声量子比特：当前的 NISQ（含噪中等规模量子）设备有较高的错误率
     （每个门约 1%）。纠错需要每个逻辑量子比特配备 1000+ 物理量子比特。

  2. 量子比特数量有限：最先进水平约 1000 个物理量子比特
     （IBM Condor 2023）。经典模拟器可轻松处理 30+ 量子比特。

  3. 输入/输出瓶颈：将经典数据编码到量子态在最坏情况下需 O(2^n)。
     读出结果需要重复测量（散粒噪声）。

  4. 贫瘠高原：随机 PQC 的梯度呈指数级消失，使训练难度与经典模拟相当。

  5. 去量子化：许多提出的量子算法已被"去量子化"——
     即有类似保证的经典算法（例如 Tang 2019 的推荐系统算法）。

  6. 机器学习尚无证明的指数级加速：与 Shor 算法（因子分解）或
     Grover 算法（搜索）不同，目前没有任何量子机器学习算法对实际
     学习问题具有可证明的指数级优势。

  7. 经典基线很强：调优良好的经典模型（Transformer、CNN、GNN）
     在许多基准上达到接近完美的结果，留给量子改进的空间很小。

  该领域对长期研究很重要，但量子机器学习在实际应用中的优势
  仍然是一个开放问题。
    """)

    # ---------- 总结 ----------
    print("--- 8. 总结 ---")
    print("  演示的概念：")
    print("    - 量子门和态演化（numpy 模拟）")
    print("    - 纠缠（贝尔态创建）")
    print("    - 参数化量子电路（PQC）")
    print("    - 使用 PQC 特征映射的量子二分类器")
    print("    - 表达能力随电路深度的变化")
    print("    - 经典模拟的指数级代价")
    print("    - 量子机器学习的当前局限性")

    print("\n完成。所有计算均在 CPU（numpy）上执行。\n")


if __name__ == "__main__":
    main()
