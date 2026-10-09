"""
tests/test_federated_demo.py
Unit tests for Federated Learning simulation, FedAvg aggregation, and Differential Privacy.
"""

import numpy as np
import pytest
from src.healthcare_ml import generate_synthetic_healthcare_data
from src.federated_demo import (
    CLIENT_METADATA,
    partition_healthcare_data,
    create_local_client_model,
    apply_laplace_differential_privacy,
    aggregate_fedavg,
    run_federated_learning_simulation,
)


def test_data_partitioning():
    """Verify data is partitioned across 3 institutional clients without sample loss."""
    df = generate_synthetic_healthcare_data(n_samples=600, random_state=42)
    partitions = partition_healthcare_data(df, random_state=42)

    assert len(partitions) == 3
    assert set(partitions.keys()) == {"client_alpha", "client_beta", "client_gamma"}

    # Verify total sample preservation
    total_samples = sum(len(part_df) for part_df in partitions.values())
    assert total_samples == 600

    # Verify each client receives a non-empty partition
    for cid, part_df in partitions.items():
        assert len(part_df) > 50


def test_fedavg_aggregation():
    """Verify FedAvg calculates exact weighted parameter averages."""
    # Mock client parameters
    # Client 1: 100 samples, w = [1.0, 2.0]
    # Client 2: 300 samples, w = [3.0, 4.0]
    client_params = [
        {
            "client_id": "c1",
            "n_samples": 100,
            "weights": np.array([[1.0, 2.0]]),
            "intercept": np.array([0.5]),
        },
        {
            "client_id": "c2",
            "n_samples": 300,
            "weights": np.array([[3.0, 4.0]]),
            "intercept": np.array([1.5]),
        },
    ]

    w_agg, b_agg = aggregate_fedavg(client_params, epsilon=None)

    # Expected: (100 * [1, 2] + 300 * [3, 4]) / 400 = (1/4)*[1,2] + (3/4)*[3,4] = [2.5, 3.5]
    expected_w = np.array([[2.5, 3.5]])
    expected_b = np.array([1.25])

    np.testing.assert_allclose(w_agg, expected_w, atol=1e-5)
    np.testing.assert_allclose(b_agg, expected_b, atol=1e-5)


def test_differential_privacy_noise():
    """Verify Laplace differential privacy mechanism and parameter shape preservation."""
    weights = np.array([[0.5, -0.3, 0.8, -0.2]])
    intercept = np.array([0.1])

    # No noise when epsilon is zero
    w_clean, b_clean = apply_laplace_differential_privacy(weights, intercept, epsilon=0.0)
    np.testing.assert_array_equal(w_clean, weights)
    np.testing.assert_array_equal(b_clean, intercept)

    # Noise injected when epsilon > 0
    rng = np.random.default_rng(123)
    w_noisy, b_noisy = apply_laplace_differential_privacy(
        weights, intercept, epsilon=0.5, rng=rng
    )

    assert w_noisy.shape == weights.shape
    assert b_noisy.shape == intercept.shape
    assert not np.array_equal(w_noisy, weights)


def test_federated_simulation_run():
    """Verify multi-round federated training convergence and evaluation."""
    results = run_federated_learning_simulation(
        n_samples=600, n_rounds=3, epsilon=None, random_state=42
    )

    assert "n_rounds" in results
    assert results["n_rounds"] == 3
    assert "round_history" in results
    assert len(results["round_history"]) == 3

    history = results["round_history"]
    for r_entry in history:
        assert "round" in r_entry
        assert "accuracy" in r_entry
        assert "f1_score" in r_entry
        assert 0.0 <= r_entry["accuracy"] <= 1.0

    # Final federated model should perform reasonably well (>0.60 accuracy)
    assert results["final_accuracy"] > 0.60
    assert len(results["client_stats"]) == 3
