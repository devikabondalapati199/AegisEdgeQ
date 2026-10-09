"""
src/quantum_circuit.py
Quantum Circuit Engine for AegisEdge-Q using Qiskit and Qiskit Aer.

Provides:
- Parameterized 3-qubit quantum state encoding via rotation gates (RY, RZ).
- Entanglement topology (CX / CZ gates) creating non-classical correlations.
- Local quantum circuit execution on Qiskit Aer Simulator (AerSimulator).
- Measurement counts, bitstring probability distribution, and expectation values.
- Circuit visualization helpers (Matplotlib figures and Unicode/ASCII text).
- Clear scientific demarcation: Circuit execution vs. clinical diagnostics.

DISCLAIMER:
Raw quantum measurements or state sampling do NOT constitute a medical diagnosis.
In AegisEdge-Q, the quantum circuit demonstrates state preparation, feature mapping,
and entanglement simulation on local quantum simulators without external hardware.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Qiskit imports
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt


def normalize_features_for_angles(
    features: List[float],
    min_vals: Optional[List[float]] = None,
    max_vals: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Min-Max scale numerical vitals into rotation angles [0, pi] for RY gate encoding.
    Default bounds map standard human ranges:
    - Age: [18, 90]
    - Heart Rate: [45, 145]
    - Systolic BP: [85, 200]
    """
    if min_vals is None:
        min_vals = [18.0, 45.0, 85.0]
    if max_vals is None:
        max_vals = [90.0, 145.0, 200.0]

    feats = np.array(features[:3], dtype=float)
    mins = np.array(min_vals[: len(feats)], dtype=float)
    maxs = np.array(max_vals[: len(feats)], dtype=float)

    # Prevent division by zero
    ranges = np.where(maxs - mins == 0, 1.0, maxs - mins)
    scaled = np.clip((feats - mins) / ranges, 0.0, 1.0)
    angles = scaled * np.pi
    return angles


def build_aegis_quantum_circuit(
    angles: List[float],
    entanglement_type: str = "cx",
    add_variational_layer: bool = True,
    var_angles: Optional[List[float]] = None,
) -> QuantumCircuit:
    """
    Construct a parameterized 3-qubit healthcare quantum feature circuit.

    Architecture:
    1. Layer 1 (State Prep / Angle Encoding):
       RY(theta_i) on each qubit q_i, mapping classical vitals to Bloch sphere rotations.
    2. Layer 2 (Entanglement):
       Creates multipartite superposition and entanglement using CX (CNOT) or CZ rings:
       - q0 -> q1
       - q1 -> q2
       - q2 -> q0
    3. Layer 3 (Phase / Variational Modulation - optional):
       RZ rotations to introduce quantum phase shifts.
    4. Measurement Layer:
       All 3 qubits measured into 3 classical bits.
    """
    num_qubits = 3
    qc = QuantumCircuit(num_qubits, num_qubits, name="AegisEdgeQ_Feature_Circuit")

    # Layer 1: Angle Encoding (RY gates)
    for i in range(num_qubits):
        angle = float(angles[i]) if i < len(angles) else 0.0
        qc.ry(angle, i)

    qc.barrier()

    # Layer 2: Entanglement
    ent_type = entanglement_type.lower()
    if ent_type == "cz":
        qc.cz(0, 1)
        qc.cz(1, 2)
        qc.cz(2, 0)
    else:  # Default CX (CNOT)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 0)

    qc.barrier()

    # Layer 3: Variational / Phase Modulation
    if add_variational_layer:
        if var_angles is None:
            var_angles = [np.pi / 4.0, np.pi / 3.0, np.pi / 6.0]
        for i in range(num_qubits):
            v_angle = float(var_angles[i % len(var_angles)])
            qc.rz(v_angle, i)
        qc.barrier()

    # Layer 4: Measurement
    qc.measure(range(num_qubits), range(num_qubits))

    return qc


def execute_quantum_simulation(
    circuit: QuantumCircuit,
    shots: int = 1024,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Execute a QuantumCircuit on the local Qiskit Aer simulator.

    Returns:
    - counts: raw bitstring frequency dictionary (e.g. {'000': 342, '111': 288, ...})
    - probabilities: normalized probabilities for each observed bitstring
    - expectation_z: expectation values <Z_i> for each qubit
    - shots: total shots simulated
    - depth: circuit depth
    - num_qubits: number of qubits
    - simulator_name: simulator backend name
    """
    try:
        simulator = AerSimulator(seed_simulator=random_seed)
        transpiled_qc = transpile(circuit, simulator)
        job = simulator.run(transpiled_qc, shots=shots)
        result = job.result()
        raw_counts = result.get_counts(circuit)
    except Exception as e:
        raise RuntimeError(f"Qiskit Aer simulation failed: {str(e)}") from e

    # Ensure counts are clean string keys with integer counts
    counts = {str(k): int(v) for k, v in raw_counts.items()}
    total_shots = sum(counts.values())

    # Compute normalized probabilities
    probabilities = {k: round(v / total_shots, 4) for k, v in counts.items()}

    # Compute per-qubit expectation values <Z> = (P(0) - P(1))
    # Bitstring in Qiskit is ordered q_N-1 ... q_1 q_0 (little-endian)
    num_qubits = circuit.num_qubits
    exp_z = []
    for qubit_idx in range(num_qubits):
        z_sum = 0
        for bitstring, count in counts.items():
            # bitstring[-(qubit_idx + 1)] corresponds to qubit_idx
            bit = bitstring[-(qubit_idx + 1)]
            val = 1 if bit == "0" else -1
            z_sum += val * count
        exp_z.append(round(z_sum / total_shots, 4))

    return {
        "counts": counts,
        "probabilities": probabilities,
        "expectation_z": exp_z,
        "shots": total_shots,
        "circuit_depth": circuit.depth(),
        "num_qubits": num_qubits,
        "gate_counts": dict(circuit.count_ops()),
        "simulator_name": simulator.name,
    }


def draw_circuit_ascii(circuit: QuantumCircuit) -> str:
    """
    Return the ASCII/Unicode text representation of the quantum circuit.
    """
    return circuit.draw(output="text").single_string()


def draw_circuit_matplotlib(
    circuit: QuantumCircuit, figsize: Tuple[int, int] = (9, 4)
) -> Optional[plt.Figure]:
    """
    Generate a Matplotlib Figure of the quantum circuit diagram.
    """
    try:
        fig = circuit.draw(output="mpl", style="iqp")
        return fig
    except Exception:
        # Fallback to standard matplotlib figure if specific style is missing
        try:
            fig = circuit.draw(output="mpl")
            return fig
        except Exception:
            return None
