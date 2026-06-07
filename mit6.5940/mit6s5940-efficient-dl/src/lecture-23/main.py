#!/usr/bin/env python3
"""
MIT 6.5940 Lecture 23: Quantum ML Simulation

Topics covered:
  - Implement a simple Parameterized Quantum Circuit (PQC) simulator using
    numpy
  - Build a quantum binary classifier (simulating qubits as complex vectors)
  - Demonstrate entanglement and expressivity concepts
  - Show quantum advantage is still theoretical (limitations discussion in
    code comments)

All computation runs on CPU.  No GPU required.  Uses numpy for linear
algebra on complex-valued state vectors.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Callable

import numpy as np


# ===========================================================================
# 1. Quantum Gate Definitions
# ===========================================================================

# Pauli matrices
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
HADAMARD = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)

# Identity
I2 = np.eye(2, dtype=complex)


def rotation_x(theta: float) -> np.ndarray:
    """Rotation around X axis: exp(-i * theta/2 * sigma_x)."""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def rotation_y(theta: float) -> np.ndarray:
    """Rotation around Y axis: exp(-i * theta/2 * sigma_y)."""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rotation_z(theta: float) -> np.ndarray:
    """Rotation around Z axis: exp(-i * theta/2 * sigma_z)."""
    return np.array(
        [[math.e ** (-1j * theta / 2), 0], [0, math.e ** (1j * theta / 2)]],
        dtype=complex,
    )


def cnot_gate() -> np.ndarray:
    """Controlled-NOT gate (4x4 matrix)."""
    return np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )


# ===========================================================================
# 2. Tensor Product Utilities
# ===========================================================================


def kronecker_product(*mats: np.ndarray) -> np.ndarray:
    """Compute the Kronecker product of multiple matrices.

    For quantum circuits, this expands single-qubit gates to the full
    Hilbert space.
    """
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(result, m)
    return result


def apply_gate(
    state: np.ndarray, gate: np.ndarray, target_qubits: List[int], num_qubits: int
) -> np.ndarray:
    """Apply a gate to specific qubits of a state vector.

    Builds the full 2^n x 2^n operator by tensor products with
    identity matrices on non-target qubits.

    Args:
        state: state vector of shape (2**num_qubits,)
        gate: gate matrix acting on target qubits
        target_qubits: indices of qubits the gate acts on
        num_qubits: total number of qubits

    Returns:
        Updated state vector.
    """
    # Build full operator using tensor products
    full_op = np.eye(1, dtype=complex)
    for q in range(num_qubits):
        if q in target_qubits:
            # Find position in gate
            pos = target_qubits.index(q)
            # For multi-qubit gates, we need correct ordering
            full_op = np.kron(full_op, np.eye(2, dtype=complex))
        else:
            full_op = np.kron(full_op, I2)

    # Alternative: use permutation-based approach for correctness
    # We'll construct the full operator properly
    return _apply_gate_correct(state, gate, target_qubits, num_qubits)


def _apply_gate_correct(
    state: np.ndarray, gate: np.ndarray, targets: List[int], n: int
) -> np.ndarray:
    """Apply gate correctly by permuting qubits.

    Strategy:
      1. Permute state so target qubits are the most significant
      2. Apply gate (tensor with identity on remaining qubits)
      3. Inverse permute back
    """
    if len(targets) == 1:
        # Single-qubit gate: simpler path
        full_gate = np.eye(1, dtype=complex)
        for q in range(n):
            if q in targets:
                full_gate = np.kron(full_gate, gate)
            else:
                full_gate = np.kron(full_gate, I2)
        return full_gate @ state

    elif len(targets) == 2 and gate.shape == (4, 4):
        # Two-qubit gate (e.g. CNOT)
        full_gate = np.eye(1, dtype=complex)
        for q in range(n):
            if q == targets[0]:
                # Build two-qubit operator starting here
                full_gate = np.kron(full_gate, gate)
                # Skip the next target (already included)
                continue
            elif q == targets[1]:
                # Already handled above
                continue
            else:
                full_gate = np.kron(full_gate, I2)
        return full_gate @ state

    else:
        raise ValueError(
            f"Unsupported gate shape {gate.shape} on {len(targets)} target(s)"
        )


# ===========================================================================
# 3. Parameterized Quantum Circuit (PQC)
# ===========================================================================


class ParameterizedQuantumCircuit:
    """A simple PQC with Ry rotations and CNOT entangling layers.

    Architecture:
      - Layer of Ry(θ_i) on each qubit
      - CNOT ladder for entanglement (nearest-neighbor)
      - Repeated L times for expressivity
    """

    def __init__(self, num_qubits: int, num_layers: int = 2):
        """
        Args:
            num_qubits: number of qubits in the circuit
            num_layers: number of rotation+entanglement layers
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        # Parameters per layer: one rotation per qubit
        self.num_params = num_qubits * num_layers

    def _build_circuit(self, params: np.ndarray) -> np.ndarray:
        """Build the full unitary matrix representing the circuit.

        Args:
            params: numpy array of shape (num_params,) -- rotation angles

        Returns:
            Unitary matrix of shape (2**n, 2**n).
        """
        n = self.num_qubits
        dim = 2**n
        U = np.eye(dim, dtype=complex)

        for layer in range(self.num_layers):
            # Rotation layer
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
                U = full_Ry @ U

            # Entanglement layer: CNOT ladder
            for q in range(n - 1):
                cnot_full = np.eye(1, dtype=complex)
                for qi in range(n):
                    if qi == q:
                        cnot_full = np.kron(cnot_full, cnot_gate())
                        continue
                    elif qi == q + 1:
                        continue  # already in the CNOT
                    else:
                        cnot_full = np.kron(cnot_full, I2)
                U = cnot_full @ U

        return U

    def run(
        self, params: np.ndarray, initial_state: np.ndarray | None = None
    ) -> np.ndarray:
        """Execute the circuit on an initial state.

        Args:
            params: rotation angles (num_params,)
            initial_state: initial state vector (default: |0...0>)

        Returns:
            Final state vector.
        """
        if initial_state is None:
            initial_state = np.zeros(2**self.num_qubits, dtype=complex)
            initial_state[0] = 1.0

        U = self._build_circuit(params)
        return U @ initial_state

    def measure_expectation(self, params: np.ndarray, observable: str = "Z") -> float:
        """Measure expectation value of observable on first qubit.

        Args:
            params: rotation angles
            observable: "Z" for Pauli-Z on qubit 0

        Returns:
            Expectation value <Z_0> in [-1, 1].
        """
        state = self.run(params)
        n = self.num_qubits

        # Build full Z observable on qubit 0
        Z_full = np.eye(1, dtype=complex)
        for q in range(n):
            if q == 0:
                Z_full = np.kron(Z_full, PAULI_Z)
            else:
                Z_full = np.kron(Z_full, I2)

        expectation = (state.conj().T @ Z_full @ state).real
        return float(expectation)


# ===========================================================================
# 4. Quantum Binary Classifier
# ===========================================================================


class QuantumBinaryClassifier:
    """A simple quantum binary classifier using a PQC as feature map.

    The idea: encode classical data into rotation angles, process
    through a PQC, and measure a qubit to get a binary prediction.
    """

    def __init__(self, num_qubits: int = 3, num_layers: int = 2):
        self.num_qubits = num_qubits
        self.circuit = ParameterizedQuantumCircuit(num_qubits, num_layers)
        # Trainable parameters
        self.params = np.random.uniform(0, 2 * math.pi, self.circuit.num_params)

    def _encode_data(self, features: np.ndarray) -> np.ndarray:
        """Encode classical features into rotation parameters.

        Uses angle encoding: each feature maps to a rotation angle
        via arctan or direct scaling.

        Args:
            features: classical feature vector

        Returns:
            Circuit parameters (concatenation of encoded angles + trainable).
        """
        # Normalize and map to angles
        norm = np.linalg.norm(features) + 1e-8
        encoded = features / norm * math.pi
        # Pad/truncate to fit num_params
        if len(encoded) < self.circuit.num_params:
            encoded = np.pad(encoded, (0, self.circuit.num_params - len(encoded)))
        else:
            encoded = encoded[: self.circuit.num_params]
        # Combine with trainable parameters
        return encoded + self.params

    def predict(self, features: np.ndarray) -> int:
        """Binary classification: returns 0 or 1.

        Uses expectation value of Z on qubit 0: sign(expectation) -> class.
        """
        params = self._encode_data(features)
        exp = self.circuit.measure_expectation(params, "Z")
        return 1 if exp > 0 else 0

    def predict_proba(self, features: np.ndarray) -> float:
        """Return probability-like score in [0, 1]."""
        params = self._encode_data(features)
        exp = self.circuit.measure_expectation(params, "Z")
        # Map [-1, 1] -> [0, 1]
        return float((exp + 1) / 2)


# ===========================================================================
# 5. Entanglement Demonstration
# ===========================================================================


def demonstrate_entanglement() -> None:
    """Demonstrate that quantum circuits can create entangled states.

    Creates a Bell state: (|00> + |11>) / sqrt(2), then shows that
    measuring one qubit determines the other.
    """
    print("  Creating Bell state: H on q0, CNOT(q0, q1)")
    # Start with |00>
    state = np.zeros(4, dtype=complex)
    state[0] = 1.0

    # Apply Hadamard on q0
    H0 = np.kron(HADAMARD, I2)
    state = H0 @ state

    # Apply CNOT with q0 as control, q1 as target: CNOT @ (H ⊗ I) |00>
    CNOT = cnot_gate()  # |00>,|01>,|10>,|11> -- control q0, target q1
    state = CNOT @ state

    print(f"  Bell state: {np.round(state, 4)}")
    probs = np.abs(state) ** 2
    print(
        f"  Measurement probabilities: |00>={probs[0]:.2f}, "
        f"|01>={probs[1]:.2f}, |10>={probs[2]:.2f}, |11>={probs[3]:.2f}"
    )
    print("  -> Qubits are entangled: measuring q0=0 forces q1=0, etc.")


# ===========================================================================
# 6. Expressivity Analysis
# ===========================================================================


def analyze_expressivity(
    num_qubits: int, num_layers_list: List[int], num_samples: int = 200
) -> None:
    """Analyze the expressivity of PQCs with different numbers of layers.

    Expressivity: the ability of a circuit to produce diverse output
    states. Measured by the variance of expectation values over random
    parameters.

    Args:
        num_qubits: number of qubits
        num_layers_list: list of layer counts to test
        num_samples: random parameter samples per configuration
    """
    print(f"\n  Expressivity analysis ({num_qubits} qubits):")
    print(f"  {'Layers':>6} {'<Z> Mean':>10} {'<Z> Std':>10} {'State Space':>12}")
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
            f"{'2^' + str(num_qubits) + ' Hilbert space':>12}"
        )
        print(
            f"          -> More layers = higher expressivity (larger std) "
            f"[current std={expectations.std():.3f}]"
        )


# ===========================================================================
# 7. Classical Simulation Cost Analysis
# ===========================================================================


def analyze_scaling():
    """Analyze the exponential classical cost of simulating quantum circuits.

    This is the core argument for why quantum advantage is theoretical:
    classical simulation requires O(2^n) memory and time.
    """
    print("\n--- Classical Simulation Cost Analysis ---")
    print("  Simulating n qubits requires storing a 2^n complex vector.")
    print("  (Each element is 16 bytes for complex128)")
    print()
    print(f"  {'Qubits':>6} {'State Size':>10} {'Memory':>12} {'Feasible?':>10}")
    print(f"  {'-' * 42}")
    for n in [5, 10, 15, 20, 25, 30, 35, 40, 50]:
        dim = 2**n
        mem_bytes = dim * 16
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
        feasible = "Yes" if n <= 30 else "No"
        print(f"  {n:>6} {dim:>10,} {mem_str:>12} {feasible:>10}")


# ===========================================================================
# 8. Main Demonstration
# ===========================================================================


def main() -> None:
    print("=" * 72)
    print("MIT 6.5940 Lecture 23: Quantum ML Simulation")
    print("=" * 72)

    # ---------- Quantum Gate Showcase ----------
    print("\n--- 1. Quantum Gates ---")
    print(f"  Hadamard gate:\n{HADAMARD}")
    print(f"  CNOT gate:\n{cnot_gate()}")
    print(f"  R_y(pi/4):\n{np.round(rotation_y(math.pi / 4), 4)}")

    # ---------- Single Qubit Rotation ----------
    print("\n--- 2. Single-Qubit State Evolution ---")
    state0 = np.array([1.0, 0.0], dtype=complex)
    for angle in [0, math.pi / 4, math.pi / 2, math.pi]:
        Ry = rotation_y(angle)
        final = Ry @ state0
        prob0 = abs(final[0]) ** 2
        prob1 = abs(final[1]) ** 2
        print(
            f"  R_y({angle:.2f})|0> -> [{final[0]:.3f}, {final[1]:.3f}], "
            f"p(0)={prob0:.3f}, p(1)={prob1:.3f}"
        )

    # ---------- Entanglement ----------
    print("\n--- 3. Entanglement Demonstration ---")
    demonstrate_entanglement()

    # ---------- PQC and Binary Classifier ----------
    print("\n--- 4. Parameterized Quantum Circuit ---")
    pqc = ParameterizedQuantumCircuit(num_qubits=3, num_layers=2)
    print(f"  Qubits: {pqc.num_qubits}, Layers: {pqc.num_layers}")
    print(f"  Trainable params: {pqc.num_params}")

    # Sample circuit execution
    params = np.array([0.5, 1.0, 0.3, 0.8, 0.2, 1.5])
    final_state = pqc.run(params)
    print(f"  Input params: {params}")
    print(f"  Final state (first 4 amplitudes): {np.round(final_state[:4], 4)}")
    exp_z = pqc.measure_expectation(params, "Z")
    print(f"  <Z_0> = {exp_z:.4f}")

    # ---------- Quantum Binary Classifier ----------
    print("\n--- 5. Quantum Binary Classifier ---")
    qbc = QuantumBinaryClassifier(num_qubits=3, num_layers=2)
    print(f"  Trainable parameters: {qbc.circuit.num_params}")

    # Test on random features
    test_features = np.array(
        [
            [1.0, 0.5, -0.3],
            [0.2, -0.8, 0.6],
            [-1.0, 0.2, 0.9],
            [0.7, -0.1, -0.5],
        ]
    )
    print(f"  {'Features':<25} {'Prediction':>10} {'Score':>8}")
    print(f"  {'-' * 45}")
    for feat in test_features:
        pred = qbc.predict(feat)
        proba = qbc.predict_proba(feat)
        print(f"  {str(np.round(feat, 2)):<25} {pred:>10} {proba:>8.4f}")

    # ---------- Expressivity ----------
    print("\n--- 6. Expressivity Analysis ---")
    analyze_expressivity(num_qubits=3, num_layers_list=[1, 2, 4, 8])

    # ---------- Classical Cost ----------
    analyze_scaling()

    # ---------- Limitations Discussion ----------
    print("\n--- 7. Discussion: Quantum Advantage Limitations ---")
    print("""
  Quantum ML is a fascinating but nascent field. Current limitations:

  1. NOISY QUBITS: Current NISQ devices have high error rates (~1%
     per gate). Error correction requires 1000+ physical qubits per
     logical qubit.

  2. LIMITED QUBIT COUNT: State-of-the-art: ~1000 physical qubits
     (IBM Condor 2023). Classical simulators easily handle 30+ qubits.

  3. INPUT/OUTPUT BOTTLENECK: Encoding classical data into quantum
     states is O(2^n) in the worst case. Reading out results requires
     repeated measurements (shot noise).

  4. BARREN PLATEAUS: Random PQCs have exponentially vanishing
     gradients, making training as hard as classical simulation.

  5. DEQUANTIZATION: Many proposed quantum algorithms have been
     "dequantized" -- classical algorithms with similar guarantees
     (e.g., Tang 2019 for recommendation systems).

  6. NO PROVEN EXPONENTIAL SPEEDUP FOR ML: Unlike Shor's (factoring)
     or Grover's (search), no quantum ML algorithm has a provable
     exponential advantage for a practical learning problem.

  7. CLASSICAL BASELINES ARE STRONG: Well-tuned classical models
     (transformers, CNNs, GNNs) achieve near-perfect results on
     many benchmarks, leaving little room for quantum improvement.

  The field is important for long-term research, but practical
  quantum advantage for ML remains an open question.
    """)

    # ---------- Summary ----------
    print("--- 8. Summary ---")
    print("  Demonstrated concepts:")
    print("    - Quantum gates and state evolution (numpy simulation)")
    print("    - Entanglement (Bell state creation)")
    print("    - Parameterized Quantum Circuits (PQC)")
    print("    - Quantum binary classifier using PQC feature map")
    print("    - Expressivity scaling with circuit depth")
    print("    - Exponential classical simulation cost")
    print("    - Current limitations of quantum ML")

    print("\nDone. All computations on CPU (numpy).\n")


if __name__ == "__main__":
    main()
