"""
tests/test_healthcare_ml.py
Unit tests for the Classical Healthcare Machine Learning module.
"""

import pytest
import numpy as np
import pandas as pd
from src.healthcare_ml import (
    FEATURE_COLUMNS,
    FEATURE_RANGES,
    generate_synthetic_healthcare_data,
    build_ml_pipeline,
    train_and_evaluate_model,
    validate_patient_input,
    predict_patient_risk,
)


def test_synthetic_data_generation():
    """Verify synthetic healthcare dataset generation properties."""
    df = generate_synthetic_healthcare_data(n_samples=500, random_state=42)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 500
    for col in FEATURE_COLUMNS + ["high_risk"]:
        assert col in df.columns

    # Verify binary target
    unique_labels = set(df["high_risk"].unique())
    assert unique_labels.issubset({0, 1})
    assert len(unique_labels) == 2  # Both classes present

    # Test reproducibility
    df2 = generate_synthetic_healthcare_data(n_samples=500, random_state=42)
    pd.testing.assert_frame_equal(df, df2)


def test_synthetic_data_missing_values():
    """Verify handling of synthetic missing values."""
    df_missing = generate_synthetic_healthcare_data(
        n_samples=300, random_state=42, introduce_missing=True, missing_ratio=0.1
    )
    assert df_missing[FEATURE_COLUMNS].isna().sum().sum() > 0

    # Test that the pipeline handles missing values seamlessly
    results = train_and_evaluate_model(df_missing, random_state=42)
    assert results["metrics"]["accuracy"] > 0.5


def test_model_training_and_metrics():
    """Verify training, held-out evaluation, and dynamic metric calculations."""
    df = generate_synthetic_healthcare_data(n_samples=800, random_state=42)
    results = train_and_evaluate_model(df, test_size=0.25, random_state=42)

    assert "pipeline" in results
    assert "metrics" in results
    metrics = results["metrics"]

    # Verify all required metrics are present and dynamically computed
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0

    # Metrics should be significantly better than random guessing on structured synthetic data
    assert metrics["accuracy"] > 0.70
    assert metrics["roc_auc"] > 0.70

    # Verify confusion matrix structure
    cm = np.array(metrics["confusion_matrix"])
    assert cm.shape == (2, 2)
    assert np.sum(cm) == metrics["n_test"]


def test_patient_input_validation():
    """Verify vital sign boundary checks and validation error detection."""
    valid_input = {
        "age": 45.0,
        "heart_rate": 72.0,
        "systolic_bp": 120.0,
        "body_temp": 36.6,
    }
    is_valid, msg = validate_patient_input(valid_input)
    assert is_valid is True
    assert msg is None

    # Missing field
    is_valid, msg = validate_patient_input({"age": 45.0})
    assert is_valid is False
    assert "Missing vital sign" in msg

    # Non-numeric value
    is_valid, msg = validate_patient_input(
        {"age": "forty-five", "heart_rate": 72.0, "systolic_bp": 120.0, "body_temp": 36.6}
    )
    assert is_valid is False

    # Out of physiological range (e.g., negative heart rate or 300 bpm)
    is_valid, msg = validate_patient_input(
        {"age": 45.0, "heart_rate": 300.0, "systolic_bp": 120.0, "body_temp": 36.6}
    )
    assert is_valid is False
    assert "outside plausible demonstration range" in msg


def test_predict_patient_risk():
    """Verify single-instance patient prediction inference."""
    df = generate_synthetic_healthcare_data(n_samples=500, random_state=42)
    results = train_and_evaluate_model(df, random_state=42)
    pipeline = results["pipeline"]

    nominal_vitals = {
        "age": 28.0,
        "heart_rate": 65.0,
        "systolic_bp": 110.0,
        "body_temp": 36.6,
    }
    prediction = predict_patient_risk(pipeline, nominal_vitals)

    assert "predicted_class" in prediction
    assert prediction["predicted_class"] in [0, 1]
    assert 0.0 <= prediction["risk_probability"] <= 1.0
    assert "risk_tier" in prediction
    assert "disclaimer" in prediction
    assert "SYNTHETIC DEMONSTRATION ONLY" in prediction["disclaimer"]

    # Invalid input raises ValueError
    with pytest.raises(ValueError):
        predict_patient_risk(pipeline, {"age": -5.0})
