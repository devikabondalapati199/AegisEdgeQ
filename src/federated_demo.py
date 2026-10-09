"""
src/federated_demo.py
Decentralized Federated Learning Simulation with FedAvg & Differential Privacy.

Implements:
- 3-Client Institutional Partitioning:
  * Hospital Alpha (Metropolitan Tertiary Care)
  * Regional Clinic Beta (Suburban Outpatient)
  * University Health Center Gamma (Academic Research)
- Local decentralized model training with strict data locality.
- Federated Averaging (FedAvg, McMahan et al. 2017) parameter aggregation:
    w_global = sum_{k=1}^K (n_k / N) * w_k
- Differential Privacy (Laplace Mechanism, Dwork et al. 2006) for privacy-utility tradeoff.
- Multi-round convergence simulation tracking global performance.

DISCLAIMER:
Federated Learning alone does NOT guarantee privacy against model inversion,
membership inference, or gradient reconstruction attacks. This module simulates
the decentralized workflow and demonstrates how Differential Privacy noise (epsilon)
affects the utility-privacy balance.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.healthcare_ml import FEATURE_COLUMNS, generate_synthetic_healthcare_data


CLIENT_METADATA = [
    {
        "id": "client_alpha",
        "name": "Hospital Alpha",
        "type": "Metropolitan Tertiary Care",
        "sample_weight": 0.45,
    },
    {
        "id": "client_beta",
        "name": "Regional Clinic Beta",
        "type": "Suburban Outpatient Facility",
        "sample_weight": 0.35,
    },
    {
        "id": "client_gamma",
        "name": "Health Center Gamma",
        "type": "University Academic Center",
        "sample_weight": 0.20,
    },
]


def partition_healthcare_data(
    df: pd.DataFrame,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Partition synthetic data across 3 clients using realistic clinical proportions.
    Each client's records remain private and locally siloed.
    """
    rng = np.random.default_rng(random_state)
    n_total = len(df)
    shuffled_indices = rng.permutation(n_total)

    idx_alpha = int(n_total * CLIENT_METADATA[0]["sample_weight"])
    idx_beta = idx_alpha + int(n_total * CLIENT_METADATA[1]["sample_weight"])

    partitions = {
        "client_alpha": df.iloc[shuffled_indices[:idx_alpha]].copy().reset_index(drop=True),
        "client_beta": df.iloc[shuffled_indices[idx_alpha:idx_beta]].copy().reset_index(drop=True),
        "client_gamma": df.iloc[shuffled_indices[idx_beta:]].copy().reset_index(drop=True),
    }

    return partitions


def create_local_client_model(random_state: int = 42) -> SGDClassifier:
    """
    Create an SGDClassifier configured with logistic loss (log_loss)
    to facilitate parameter-level weight extraction and FedAvg updates.
    """
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.001,
        max_iter=100,
        tol=1e-3,
        random_state=random_state,
        warm_start=True,
    )


def apply_laplace_differential_privacy(
    weights: np.ndarray,
    intercept: np.ndarray,
    epsilon: float,
    clip_norm: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Laplace Differential Privacy noise to the aggregated model parameters.
    Based on Dwork et al. (2006):
        Noise ~ Laplace(0, Delta / epsilon)
    where Delta is the L1 sensitivity bounded by clipping.
    """
    if epsilon <= 0.0 or np.isinf(epsilon):
        # No noise applied (epsilon -> infinity)
        return weights.copy(), intercept.copy()

    if rng is None:
        rng = np.random.default_rng()

    # Clip weights to enforce L1 sensitivity bound
    l1_w = np.sum(np.abs(weights))
    if l1_w > clip_norm:
        clipped_weights = weights * (clip_norm / l1_w)
    else:
        clipped_weights = weights.copy()

    scale = clip_norm / float(epsilon)
    w_noise = rng.laplace(0.0, scale, size=clipped_weights.shape)
    b_noise = rng.laplace(0.0, scale * 0.1, size=intercept.shape)

    noisy_weights = clipped_weights + w_noise
    noisy_intercept = intercept + b_noise

    return noisy_weights, noisy_intercept


def aggregate_fedavg(
    client_params: List[Dict[str, Any]],
    epsilon: Optional[float] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Execute Federated Averaging (FedAvg, McMahan et al. 2017) over local client updates:
        w_global = sum_{k=1}^K (n_k / N) * w_k
        b_global = sum_{k=1}^K (n_k / N) * b_k

    Optionally injects Laplace Differential Privacy noise into the aggregated vector.
    """
    total_samples = sum(cp["n_samples"] for cp in client_params)
    if total_samples == 0:
        raise ValueError("Total sample count cannot be zero during FedAvg aggregation.")

    # Weighted parameter sum
    w_shape = client_params[0]["weights"].shape
    b_shape = client_params[0]["intercept"].shape

    w_global = np.zeros(w_shape, dtype=float)
    b_global = np.zeros(b_shape, dtype=float)

    for cp in client_params:
        weight_ratio = cp["n_samples"] / total_samples
        w_global += weight_ratio * cp["weights"]
        b_global += weight_ratio * cp["intercept"]

    # Differential privacy injection
    if epsilon is not None and epsilon > 0.0:
        rng = np.random.default_rng(random_state)
        w_global, b_global = apply_laplace_differential_privacy(
            w_global, b_global, epsilon=epsilon, rng=rng
        )

    return w_global, b_global


def run_federated_learning_simulation(
    n_samples: int = 1500,
    n_rounds: int = 5,
    epsilon: Optional[float] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run an end-to-end multi-round Federated Learning simulation across 3 healthcare nodes.

    Process:
    1. Generate synthetic dataset and reserve held-out global test set.
    2. Fit standard scaler on baseline training schema.
    3. Partition decentralized client data among 3 institutions.
    4. Across Rounds 1 to n_rounds:
       - Broadcast global weights to local clients.
       - Train local SGD classifiers on private local data partitions.
       - Extract client weight vectors (no raw patient data transmitted).
       - Central coordinator performs FedAvg (+ optional DP noise).
       - Evaluate global model on the global held-out test set.
    """
    full_df = generate_synthetic_healthcare_data(
        n_samples=n_samples, random_state=random_state
    )

    # Global train-test split (20% global test set)
    test_cutoff = int(len(full_df) * 0.8)
    train_df = full_df.iloc[:test_cutoff].reset_index(drop=True)
    test_df = full_df.iloc[test_cutoff:].reset_index(drop=True)

    # Standardize features
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLUMNS])

    X_test = scaler.transform(test_df[FEATURE_COLUMNS])
    y_test = test_df["high_risk"].to_numpy()

    # Partition training data among 3 clients
    client_partitions = partition_healthcare_data(
        train_df, random_state=random_state
    )

    # Initialize global model
    global_model = create_local_client_model(random_state=random_state)
    # Warm start initialization with classes
    global_model.partial_fit(
        np.zeros((2, len(FEATURE_COLUMNS))), np.array([0, 1]), classes=np.array([0, 1])
    )
    current_weights = global_model.coef_.copy()
    current_intercept = global_model.intercept_.copy()

    round_history = []
    client_stats = []

    for meta in CLIENT_METADATA:
        cid = meta["id"]
        c_df = client_partitions[cid]
        client_stats.append(
            {
                "client_id": cid,
                "client_name": meta["name"],
                "facility_type": meta["type"],
                "sample_count": len(c_df),
                "risk_prevalence": float(c_df["high_risk"].mean()),
            }
        )

    # Run federation rounds
    for r in range(1, n_rounds + 1):
        local_updates = []

        for meta in CLIENT_METADATA:
            cid = meta["id"]
            c_df = client_partitions[cid]
            X_c = scaler.transform(c_df[FEATURE_COLUMNS])
            y_c = c_df["high_risk"].to_numpy()

            # Local training initialized with current global parameters
            local_model = create_local_client_model(
                random_state=random_state + r * 10
            )
            local_model.classes_ = np.array([0, 1])
            local_model.coef_ = current_weights.copy()
            local_model.intercept_ = current_intercept.copy()

            # Train on local private partition
            local_model.fit(X_c, y_c)

            local_updates.append(
                {
                    "client_id": cid,
                    "n_samples": len(c_df),
                    "weights": local_model.coef_.copy(),
                    "intercept": local_model.intercept_.copy(),
                }
            )

        # FedAvg Aggregation on coordinator
        agg_w, agg_b = aggregate_fedavg(
            local_updates,
            epsilon=epsilon,
            random_state=random_state + r,
        )

        current_weights = agg_w
        current_intercept = agg_b

        # Update global model with aggregated parameters
        global_model.coef_ = current_weights.copy()
        global_model.intercept_ = current_intercept.copy()

        # Evaluate global model on held-out test set
        y_pred = global_model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))

        round_history.append(
            {
                "round": r,
                "accuracy": acc,
                "f1_score": f1,
                "precision": prec,
                "recall": rec,
                "weights": current_weights.tolist()[0],
                "intercept": float(current_intercept[0]),
            }
        )

    return {
        "n_rounds": n_rounds,
        "epsilon": epsilon,
        "client_stats": client_stats,
        "round_history": round_history,
        "final_accuracy": round_history[-1]["accuracy"],
        "final_f1": round_history[-1]["f1_score"],
        "global_model": global_model,
        "scaler": scaler,
    }
