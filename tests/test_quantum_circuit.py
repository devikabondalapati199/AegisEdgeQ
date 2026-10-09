"""
tests/test_quantum_circuit.py
Unit tests for the Qiskit Quantum Circuit Engine and Aer Simulator.
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from src.quantum_circuit import (
    normalize_features_for_angles,
    build_aegis_quantum_circuit,
    execute_quantum_simulation,
    draw_circuit_ascii,
)


def test_angle_normalization():
    """Verify classical features map correctly to [0, pi] rotation angles."""
    # Test min bound maps to 0.0, max bound maps to pi
    min_vitals = [18.0, 45.0, 85.0]
    angles_min = normalize_features_for_angles(min_vitals)
    assert np.allclose(angles_min, [0.0, 0.0, 0.0], atol=1e-5)

    max_vitals = [90.0, 145.0, 200.0]
    angles_max = normalize_features_for_angles(max_vitals)
    assert np.allclose(angles_max, [np.pi, np.pi, np.pi], atol=1e-5)

    # Test intermediate vitals
    mid_vitals = [54.0, 95.0, 142.5]
    angles_mid = normalize_features_for_angles(mid_vitals)
    for a in angles_mid:
        assert 0.0 <= a <= np.pi


def test_quantum_circuit_construction():
    """Verify circuit architecture, qubit/clbit counts, and gate composition."""
    angles = [np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    qc = build_aegis_quantum_circuit(angles, entanglement_type="cx")

    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == 3
    assert qc.num_clbits == 3

    ops = qc.count_ops()
    assert "ry" in ops
    assert ops["ry"] == 3
    assert "cx" in ops
    assert ops["cx"] == 3
    assert "measure" in ops
    assert ops["measure"] == 3


def test_cz_entanglement_circuit():
    """Verify circuit construction with CZ entanglement."""
    angles = [0.5, 1.0, 1.5]
    qc_cz = build_aegis_quantum_circuit(angles, entanglement_type="cz")
    ops = qc_cz.count_ops()
    assert "cz" in ops
    assert ops["cz"] == 3


def test_aer_quantum_simulator_execution():
    """Verify local execution on Qiskit Aer simulator produces valid measurements."""
    angles = [np.pi / 3, np.pi / 4, np.pi / 2]
    qc = build_aegis_quantum_circuit(angles)

    shots = 512
    sim_results = execute_quantum_simulation(qc, shots=shots, random_seed=42)

    assert "counts" in sim_results
    assert "probabilities" in sim_results
    assert "expectation_z" in sim_results
    assert "shots" in sim_results

    counts = sim_results["counts"]
    assert isinstance(counts, dict)
    assert len(counts) > 0

    # Total shots check
    assert sum(counts.values()) == shots

    # Bitstrings must be 3-bit binary strings
    for bitstring in counts.keys():
        assert len(bitstring) == 3
        assert set(bitstring).issubset({"0", "1"})

    # Probabilities must sum to ~1.0
    probs = sim_results["probabilities"]
    assert pytest.approx(sum(probs.values()), abs=0.02) == 1.0

    # Expectation values <Z_i> must be within [-1.0, 1.0]
    exp_z = sim_results["expectation_z"]
    assert len(exp_z) == 3
    for z in exp_z:
        assert -1.0 <= z <= 1.0


def test_circuit_ascii_drawing():
    """Verify ASCII circuit renderer returns non-empty representation."""
    angles = [0.2, 0.4, 0.6]
    qc = build_aegis_quantum_circuit(angles)
    ascii_art = draw_circuit_ascii(qc)

    assert isinstance(ascii_art, str)
    assert len(ascii_art) > 50
    assert "Ry" in ascii_art
    assert "q_" in ascii_art
