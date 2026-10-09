"""
app.py
AegisEdge-Q: Privacy-Preserving Quantum Healthcare AI
Team Name: Innovista | Team ID: HT91

A Streamlit dashboard for decentralized healthcare risk analysis,
real Qiskit Aer quantum circuit simulation, and Federated Learning (FedAvg).
"""

import time
from typing import Dict, Any
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# AegisEdge-Q modules
from src.healthcare_ml import (
    FEATURE_COLUMNS,
    FEATURE_RANGES,
    generate_synthetic_healthcare_data,
    train_and_evaluate_model,
    validate_patient_input,
    predict_patient_risk,
)
from src.quantum_circuit import (
    normalize_features_for_angles,
    build_aegis_quantum_circuit,
    execute_quantum_simulation,
    draw_circuit_ascii,
    draw_circuit_matplotlib,
)
from src.federated_demo import (
    CLIENT_METADATA,
    run_federated_learning_simulation,
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AegisEdge-Q | Innovista",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 3px 12px;
        margin-right: 6px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-blue { background-color: #DBEAFE; color: #1E40AF; }
    .badge-green { background-color: #D1FAE5; color: #065F46; }
    .badge-purple { background-color: #EDE9FE; color: #5B21B6; }
    .badge-amber { background-color: #FEF3C7; color: #92400E; }
    .card-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 14px;
    }
    .stMetric {
        background: #F8FAFC;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Cached Model & Dataset Helpers
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_cached_dataset(n_samples: int = 1200, seed: int = 42) -> pd.DataFrame:
    return generate_synthetic_healthcare_data(n_samples=n_samples, random_state=seed)


@st.cache_resource(show_spinner=False)
def get_trained_baseline_model(n_samples: int = 1200, seed: int = 42) -> Dict[str, Any]:
    df = get_cached_dataset(n_samples=n_samples, seed=seed)
    return train_and_evaluate_model(df, random_state=seed)


# ---------------------------------------------------------
# Sidebar Navigation & Context
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.markdown("### **AegisEdge-Q**")
    st.markdown("**Team:** Innovista  \n**Team ID:** `HT91`")
    st.divider()

    menu_option = st.radio(
        "Navigation Menu",
        [
            "🏛️ Project Overview & Architecture",
            "🩺 Patient Risk Predictor (Classical ML)",
            "⚛️ Real Qiskit Aer Quantum Simulator",
            "📊 Model Evaluation & Metrics",
            "🌐 Federated Learning (FedAvg)",
            "🔒 Privacy, Security & Differential Privacy",
            "📚 Research References",
            "⚠️ Limitations & Technical Roadmap",
        ],
        index=0,
    )

    st.divider()
    st.markdown(
        """
        <div style="font-size:0.8rem; color:#6B7280; line-height:1.4;">
        <strong>Environment Status:</strong><br>
        • Python 3.13 / Qiskit Aer Simulator<br>
        • Local Simulation Only (No External APIs)<br>
        • Synthetic Data (Privacy-Safe)<br>
        • Zero PHI Stored or Transmitted
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# Top Header Banner
# ---------------------------------------------------------
st.markdown('<p class="main-header">🛡️ AegisEdge-Q: Privacy-Preserving Quantum Healthcare AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Decentralized Healthcare Intelligence via Local Qiskit Aer Quantum Circuits & Federated Learning</p>', unsafe_allow_html=True)

st.markdown(
    """
    <span class="badge-pill badge-blue">Team: Innovista</span>
    <span class="badge-pill badge-green">Team ID: HT91</span>
    <span class="badge-pill badge-purple">Qiskit Aer 0.17 Local Simulator</span>
    <span class="badge-pill badge-amber">100% Synthetic Data Demo</span>
    """,
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Section 1: Overview & Architecture
# ---------------------------------------------------------
if menu_option == "🏛️ Project Overview & Architecture":
    st.header("Project Overview & Edge Architecture")

    st.markdown(
        """
        **AegisEdge-Q** solves a fundamental bottleneck in modern digital health:
        *How can institutions collaboratively extract high-precision predictive signals from vital signs
        without centralizing patient data, violating HIPAA/GDPR, or relying on untrusted external cloud APIs?*

        Our multi-tier framework marries **local edge intelligence**, **reproducible classical machine learning**,
        **real local Qiskit quantum circuit encoding**, and **Federated Averaging (FedAvg)** with **Differential Privacy**.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="card-box">
                <h4>🩺 Edge Clinical Tier</h4>
                <p>Private hospital nodes record patient vital signs locally. Raw patient records are <strong>never</strong> shared or pooled into a central repository.</p>
                <ul>
                    <li>Age, Heart Rate, BP, Temp</li>
                    <li>Leak-free scikit-learn pipeline</li>
                    <li>Strict local privacy preservation</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="card-box">
                <h4>⚛️ Quantum Feature Tier</h4>
                <p>Parameterized 3-qubit circuits encode continuous vitals into quantum state Hilbert spaces using local Qiskit Aer.</p>
                <ul>
                    <li>$R_y(\theta)$ rotation angle encoding</li>
                    <li>$CX$ / $CZ$ entanglement topology</li>
                    <li>Zero paid hardware credentials needed</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="card-box">
                <h4>🌐 Federated Governance Tier</h4>
                <p>Multi-institutional consensus across 3 clinical sites using weighted FedAvg with Laplace Differential Privacy.</p>
                <ul>
                    <li>Decentralized SGD updates</li>
                    <li>Calibrated $\\epsilon$ Laplace perturbation</li>
                    <li>Resilience against gradient inversion</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Synthetic Demonstration Dataset Preview")
    st.caption("AegisEdge-Q generates a reproducible synthetic healthcare dataset for training and verification.")
    df_sample = get_cached_dataset(n_samples=8)
    st.dataframe(df_sample, use_container_width=True)

    with st.expander("ℹ️ Data Generation Methodology & Feature Specifications", expanded=False):
        st.markdown(
            """
            - **`age`**: Patient chronological age (18 - 85 years, uniformly generated).
            - **`heart_rate`**: Resting heart rate in beats per minute (normal mean 76 bpm, clipped 48-145 bpm).
            - **`systolic_bp`**: Systolic blood pressure in mmHg (normal mean 124 mmHg, clipped 85-205 mmHg).
            - **`body_temp`**: Core body temperature in Celsius (normal mean 36.8 °C, clipped 35.5-40.2 °C).
            - **`high_risk`**: Ground truth binary risk label (0 = Nominal, 1 = Elevated Risk Indicator) computed via a documented logistic combination of hypertension, tachycardia, fever, and age weighting with stochastic perturbation.
            - **Ethical Safeguard**: Synthetic demonstration data only. Contains no protected health information (PHI).
            """
        )

# ---------------------------------------------------------
# Section 2: Patient Risk Predictor (Classical ML)
# ---------------------------------------------------------
elif menu_option == "🩺 Patient Risk Predictor (Classical ML)":
    st.header("Patient Vital Signs Risk Estimator")
    st.info("Input patient vital signs to execute inference using the trained classical machine learning pipeline. Input boundary checks are strictly enforced.")

    trained_bundle = get_trained_baseline_model()
    pipeline = trained_bundle["pipeline"]

    st.subheader("Patient Vital Signs Input")

    # Preset quick buttons
    preset_cols = st.columns(3)
    preset_choice = None
    with preset_cols[0]:
        if st.button("Preset: Nominal Baseline Patient", use_container_width=True):
            st.session_state["p_age"] = 32.0
            st.session_state["p_hr"] = 68.0
            st.session_state["p_bp"] = 118.0
            st.session_state["p_temp"] = 36.6
    with preset_cols[1]:
        if st.button("Preset: Hypertensive / High Risk", use_container_width=True):
            st.session_state["p_age"] = 72.0
            st.session_state["p_hr"] = 104.0
            st.session_state["p_bp"] = 168.0
            st.session_state["p_temp"] = 38.6
    with preset_cols[2]:
        if st.button("Preset: Borderline Intermediate", use_container_width=True):
            st.session_state["p_age"] = 55.0
            st.session_state["p_hr"] = 86.0
            st.session_state["p_bp"] = 138.0
            st.session_state["p_temp"] = 37.3

    p_age = st.session_state.get("p_age", 45.0)
    p_hr = st.session_state.get("p_hr", 74.0)
    p_bp = st.session_state.get("p_bp", 122.0)
    p_temp = st.session_state.get("p_temp", 36.8)

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        in_age = st.slider("Patient Age (Years)", 18.0, 95.0, float(p_age), step=1.0)
        in_hr = st.slider("Resting Heart Rate (BPM)", 40.0, 160.0, float(p_hr), step=1.0)
    with col_v2:
        in_bp = st.slider("Systolic Blood Pressure (mmHg)", 80.0, 210.0, float(p_bp), step=1.0)
        in_temp = st.slider("Body Temperature (°C)", 35.0, 41.0, float(p_temp), step=0.1)

    patient_payload = {
        "age": in_age,
        "heart_rate": in_hr,
        "systolic_bp": in_bp,
        "body_temp": in_temp,
    }

    try:
        prediction = predict_patient_risk(pipeline, patient_payload)
        prob = prediction["risk_probability"]
        tier = prediction["risk_tier"]

        st.divider()
        st.subheader("Inference Results & Risk Stratification")

        res_c1, res_c2, res_c3 = st.columns(3)
        with res_c1:
            st.metric("Risk Class Output", "Class 1 (Elevated Risk)" if prediction["predicted_class"] == 1 else "Class 0 (Nominal)")
        with res_c2:
            st.metric("Estimated Risk Probability", f"{prob * 100:.1f} %")
        with res_c3:
            st.metric("Model Confidence", f"{prediction['confidence'] * 100:.1f} %")

        if prediction["predicted_class"] == 1:
            st.error(f"⚠️ **{tier}**: {prediction['recommendation']}")
        else:
            st.success(f"✅ **{tier}**: {prediction['recommendation']}")

        # Learned Coefficients breakdown
        st.subheader("Learned Feature Importance (Trained Baseline Weights)")
        coefs = trained_bundle["coefficients"]
        fig, ax = plt.subplots(figsize=(7, 2.5))
        ax.barh(list(coefs.keys()), list(coefs.values()), color="#2563EB")
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Logistic Regression Coefficient Weight")
        ax.set_title("Standardized Impact on Risk Propensity")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error(f"Validation Error: {str(e)}")

    st.caption("⚠️ **Notice:** Synthetic algorithmic demonstration. Not intended for clinical diagnostic use.")

# ---------------------------------------------------------
# Section 3: Real Qiskit Aer Quantum Simulator
# ---------------------------------------------------------
elif menu_option == "⚛️ Real Qiskit Aer Quantum Simulator":
    st.header("Real Qiskit Quantum Circuit Simulation")
    st.markdown(
        """
        This section uses **Qiskit** and **Qiskit Aer** (`AerSimulator`) running directly on your local system.
        Classical vital signs are normalized into rotation angles $\\theta_i \\in [0, \\pi]$ for $R_y$ single-qubit gates,
        followed by entangling operations ($CX$ / $CZ$) and projective measurements.
        """
    )

    st.warning(
        "🔬 **Scientific Integrity Notice:** A quantum circuit executing on a simulator is an exploratory state-encoding "
        "and Hilbert space demonstration. It is NOT automatically a trained quantum diagnostic tool. "
        "AegisEdge-Q explicitly distinguishes real quantum circuit execution from clinical risk predictions."
    )

    qc_c1, qc_c2 = st.columns([1, 1])
    with qc_c1:
        st.subheader("Circuit Configuration Parameters")
        q_age = st.slider("Input Age (Feature 0 -> Qubit 0)", 18.0, 90.0, 52.0)
        q_hr = st.slider("Input Heart Rate (Feature 1 -> Qubit 1)", 45.0, 145.0, 85.0)
        q_bp = st.slider("Input Systolic BP (Feature 2 -> Qubit 2)", 85.0, 200.0, 135.0)

        ent_type = st.selectbox("Entanglement Topology", ["CX (Controlled-NOT Ring)", "CZ (Controlled-Z Ring)"])
        ent_key = "cx" if "CX" in ent_type else "cz"

        use_var = st.checkbox("Include Phase Rotation Layer (RZ Modulation)", value=True)
        shots = st.select_slider("Simulation Shot Count", options=[256, 512, 1024, 2048, 4096], value=1024)

    angles = normalize_features_for_angles([q_age, q_hr, q_bp])

    with qc_c2:
        st.subheader("Feature-to-Angle Mapping")
        st.markdown(
            f"""
            - **Qubit 0 ($q_0$)**: $\\theta_0 = {angles[0]:.3f}\\text{{ rad}}$ ({q_age} yrs normalized)
            - **Qubit 1 ($q_1$)**: $\\theta_1 = {angles[1]:.3f}\\text{{ rad}}$ ({q_hr} bpm normalized)
            - **Qubit 2 ($q_2$)**: $\\theta_2 = {angles[2]:.3f}\\text{{ rad}}$ ({q_bp} mmHg normalized)
            """
        )

        qc = build_aegis_quantum_circuit(
            angles=angles.tolist(),
            entanglement_type=ent_key,
            add_variational_layer=use_var,
        )

        st.markdown(f"**Circuit Depth:** `{qc.depth()}` | **Qubits:** `{qc.num_qubits}` | **Classical Bits:** `{qc.num_clbits}`")
        st.markdown(f"**Gate Inventory:** `{dict(qc.count_ops())}`")

    st.divider()
    run_sim = st.button("🚀 Execute Circuit on Local Qiskit Aer Simulator", type="primary")

    if run_sim or "sim_result" in st.session_state:
        if run_sim:
            t0 = time.time()
            sim_res = execute_quantum_simulation(qc, shots=shots, random_seed=42)
            elapsed = time.time() - t0
            sim_res["elapsed_sec"] = elapsed
            st.session_state["sim_result"] = sim_res
            st.session_state["current_qc"] = qc

        res = st.session_state["sim_result"]
        current_qc = st.session_state.get("current_qc", qc)

        st.success(
            f"✅ **Execution Complete** on backend `{res['simulator_name']}` in `{res.get('elapsed_sec', 0.05):.3f}s` with `{res['shots']}` shots."
        )

        # Plot Measurement Probabilities
        st.subheader("Observed Bitstring Measurement Distribution")
        probs = res["probabilities"]
        sorted_keys = sorted(probs.keys())
        sorted_vals = [probs[k] for k in sorted_keys]

        fig_dist, ax_dist = plt.subplots(figsize=(8, 3.2))
        bars = ax_dist.bar(sorted_keys, sorted_vals, color="#7C3AED", edgecolor="#4C1D95", alpha=0.85)
        ax_dist.set_ylabel("Probability")
        ax_dist.set_xlabel("3-Qubit Measured Computational State (|q2 q1 q0⟩)")
        ax_dist.set_title("Projective Measurement Probabilities on Qiskit Aer")
        ax_dist.set_ylim(0, max(sorted_vals) * 1.25 if sorted_vals else 1.0)
        for bar in bars:
            h = bar.get_height()
            ax_dist.text(bar.get_x() + bar.get_width() / 2.0, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_dist)
        plt.close(fig_dist)

        # Per-qubit Pauli-Z expectations
        st.subheader("Per-Qubit Pauli-Z Expectation Values ⟨Z_i⟩")
        exp_z = res["expectation_z"]
        ez_c1, ez_c2, ez_c3 = st.columns(3)
        with ez_c1:
            st.metric("Qubit 0 ⟨Z₀⟩ (Age)", f"{exp_z[0]:.3f}")
        with ez_c2:
            st.metric("Qubit 1 ⟨Z₁⟩ (Heart Rate)", f"{exp_z[1]:.3f}")
        with ez_c3:
            st.metric("Qubit 2 ⟨Z₂⟩ (Blood Pressure)", f"{exp_z[2]:.3f}")

        # Render Circuit Diagram
        st.subheader("Quantum Circuit Diagram")
        fig_qc = draw_circuit_matplotlib(current_qc)
        if fig_qc is not None:
            st.pyplot(fig_qc)
            plt.close(fig_qc)
        else:
            st.text(draw_circuit_ascii(current_qc))

        with st.expander("Show ASCII Circuit Text & Raw Counts JSON", expanded=False):
            st.code(draw_circuit_ascii(current_qc))
            st.json(res["counts"])

# ---------------------------------------------------------
# Section 4: Model Evaluation & Metrics
# ---------------------------------------------------------
elif menu_option == "📊 Model Evaluation & Metrics":
    st.header("Classical Model Evaluation & Metrics")
    st.markdown(
        """
        All metrics shown in this section are **calculated dynamically from actual held-out test predictions**
        on the synthesized healthcare test split. AegisEdge-Q never hardcodes evaluation numbers.
        """
    )

    trained_bundle = get_trained_baseline_model()
    metrics = trained_bundle["metrics"]

    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f} %")
    with m_col2:
        st.metric("Precision", f"{metrics['precision'] * 100:.1f} %")
    with m_col3:
        st.metric("Recall", f"{metrics['recall'] * 100:.1f} %")
    with m_col4:
        st.metric("F1-Score", f"{metrics['f1']:.3f}")
    with m_col5:
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    st.caption(f"Evaluated on N_train = {metrics['n_train']} samples | N_test = {metrics['n_test']} held-out samples.")

    plot_c1, plot_c2 = st.columns(2)
    with plot_c1:
        st.subheader("Confusion Matrix (Held-out Test Set)")
        cm = np.array(metrics["confusion_matrix"])
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        cax = ax_cm.matshow(cm, cmap="Blues")
        fig_cm.colorbar(cax)
        for i in range(2):
            for j in range(2):
                ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < cm.max() / 2 else "white", fontsize=14, fontweight="bold")
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["Pred 0 (Normal)", "Pred 1 (Risk)"])
        ax_cm.set_yticklabels(["True 0 (Normal)", "True 1 (Risk)"])
        ax_cm.set_xlabel("Predicted Label")
        ax_cm.set_ylabel("True Ground Truth")
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with plot_c2:
        st.subheader("ROC Curve (Receiver Operating Characteristic)")
        fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
        ax_roc.plot(metrics["roc_fpr"], metrics["roc_tpr"], color="#2563EB", lw=2, label=f"ROC (AUC = {metrics['roc_auc']:.3f})")
        ax_roc.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel("False Positive Rate (FPR)")
        ax_roc.set_ylabel("True Positive Rate (TPR)")
        ax_roc.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig_roc)
        plt.close(fig_roc)

# ---------------------------------------------------------
# Section 5: Federated Learning Simulation
# ---------------------------------------------------------
elif menu_option == "🌐 Federated Learning (FedAvg)":
    st.header("Decentralized Federated Learning Simulation")
    st.markdown(
        """
        Federated Learning enables multiple healthcare centers to jointly train a predictive model
        without sharing raw patient vitals. Each hospital node computes parameter updates locally,
        and only the parameter weights ($w_k, b_k$) are sent to the coordinator for **Federated Averaging (FedAvg)**.
        """
    )

    fl_c1, fl_c2 = st.columns([1, 1])
    with fl_c1:
        st.subheader("Simulation Parameters")
        n_rounds = st.slider("Aggregation Rounds", 1, 10, 5)
        n_samples = st.slider("Total Synthetic Training Pool", 600, 3000, 1500, step=300)
        enable_dp = st.checkbox("Inject Laplace Differential Privacy Noise (ε)", value=False)
        epsilon = None
        if enable_dp:
            epsilon = st.slider("Privacy Budget (ε - Epsilon)", 0.1, 5.0, 1.0, step=0.1, help="Lower ε = stronger privacy with more noise; Higher ε = higher accuracy.")

    with fl_c2:
        st.subheader("Participating Institutional Clients")
        for m in CLIENT_METADATA:
            st.markdown(
                f"""
                - **{m['name']}** ({m['type']}): Target sample allocation ~`{int(m['sample_weight'] * 100)}%`
                """
            )

    run_fl = st.button("🚀 Run Federated Learning (FedAvg) Rounds", type="primary")

    if run_fl or "fl_result" in st.session_state:
        if run_fl:
            with st.spinner("Executing decentralized training rounds across 3 simulated nodes..."):
                fl_res = run_federated_learning_simulation(
                    n_samples=n_samples,
                    n_rounds=n_rounds,
                    epsilon=epsilon,
                    random_state=42,
                )
                st.session_state["fl_result"] = fl_res

        res = st.session_state["fl_result"]

        st.subheader("Federated Aggregation Convergence")
        r_history = res["round_history"]
        rounds = [item["round"] for item in r_history]
        accuracies = [item["accuracy"] * 100 for item in r_history]
        f1s = [item["f1_score"] * 100 for item in r_history]

        col_met1, col_met2, col_met3 = st.columns(3)
        with col_met1:
            st.metric("Final Global Test Accuracy", f"{res['final_accuracy'] * 100:.1f} %")
        with col_met2:
            st.metric("Final Global Test F1", f"{res['final_f1'] * 100:.1f} %")
        with col_met3:
            st.metric("Privacy Guarantee", f"Laplace DP (ε = {res['epsilon']})" if res["epsilon"] else "Standard FedAvg (No Noise)")

        fig_fl, ax_fl = plt.subplots(figsize=(8, 3.2))
        ax_fl.plot(rounds, accuracies, marker="o", color="#2563EB", label="Global Accuracy (%)")
        ax_fl.plot(rounds, f1s, marker="s", color="#059669", label="Global F1-Score (%)")
        ax_fl.set_xlabel("Federated Communication Round")
        ax_fl.set_ylabel("Metric Value (%)")
        ax_fl.set_title("Global Model Convergence Across Communication Rounds")
        ax_fl.set_ylim(40, 100)
        ax_fl.grid(True, linestyle="--", alpha=0.5)
        ax_fl.legend()
        plt.tight_layout()
        st.pyplot(fig_fl)
        plt.close(fig_fl)

        st.subheader("Client Node Local Partitions")
        c_stats_df = pd.DataFrame(res["client_stats"])
        c_stats_df.columns = ["Client ID", "Institution Name", "Facility Type", "Local Records", "Risk Prevalence"]
        st.dataframe(c_stats_df, use_container_width=True)

# ---------------------------------------------------------
# Section 6: Privacy, Security & Differential Privacy
# ---------------------------------------------------------
elif menu_option == "🔒 Privacy, Security & Differential Privacy":
    st.header("Privacy, Security & Threat Modeling")

    st.markdown(
        """
        ### Why Federated Learning Alone Is Not Enough
        A common misconception in decentralized healthcare AI is that keeping raw records local
        guarantees absolute privacy. In reality:
        - **Gradient / Model Inversion Attacks**: An adversary observing parameter updates $\\Delta w_k$ can reconstruct patient training samples under certain conditions.
        - **Membership Inference**: Model overconfidence can reveal whether a specific patient record was used in training.
        """
    )

    col_dp1, col_dp2 = st.columns(2)
    with col_dp1:
        st.markdown(
            """
            <div class="card-box">
                <h4>🛡️ Differential Privacy (DP) Defense</h4>
                <p>Based on Dwork et al. (2006), Differential Privacy bounds the maximum information leakage of any single patient by adding calibrated noise:</p>
                <p>$$\\text{Noise} \\sim \\text{Laplace}\\left(0, \\frac{\\Delta_1}{\\epsilon}\\right)$$</p>
                <p>Where $\\Delta_1$ is the $L_1$ sensitivity bound and $\\epsilon$ is the privacy budget.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_dp2:
        st.markdown(
            """
            <div class="card-box">
                <h4>⚖️ The Privacy-Utility Tradeoff</h4>
                <p>Choosing $\\epsilon$ represents a delicate balance:</p>
                <ul>
                    <li><strong>Small $\\epsilon$ (e.g. 0.2):</strong> Strict cryptographic privacy; high noise; lower predictive utility.</li>
                    <li><strong>Moderate $\\epsilon$ (e.g. 1.0 - 2.0):</strong> Strong empirical privacy with minor utility degradation.</li>
                    <li><strong>Large $\\epsilon$ or $\\infty$:</strong> Standard machine learning without differential privacy bounds.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("AegisEdge-Q Privacy Checklist & Safeguards")
    st.markdown(
        """
        - ✅ **Synthetic Data Sandboxing:** All vital signs demonstrated are synthetically generated.
        - ✅ **Zero PHI Collection:** No real patient identifiers, names, medical record numbers, or timestamps are processed.
        - ✅ **Local Execution Sandbox:** Computation runs completely on the user's local hardware without transmitting data to external APIs.
        - ✅ **Isolated Model Aggregation:** Client partitions are strictly isolated; only parameter weights are aggregated.
        - ⚠️ **Demonstration Limitation:** Real clinical federated deployments require hardware-backed secure enclaves (SGX/Nitro) and homomorphic encryption in addition to DP.
        """
    )

# ---------------------------------------------------------
# Section 7: Research References
# ---------------------------------------------------------
elif menu_option == "📚 Research References":
    st.header("Foundational Scientific Literature")
    st.markdown(
        """
        AegisEdge-Q is built upon established peer-reviewed literature spanning quantum feature spaces,
        federated optimization, and differential privacy theory.
        """
    )

    papers = [
        {
            "title": "Supervised learning with quantum-enhanced feature spaces",
            "authors": "Havlíček, V., Córcoles, A. D., Temme, K., Harrow, A. W., Kandala, A., Chow, J. M., & Gambetta, J. M.",
            "year": "2019",
            "journal": "Nature, 567(7747), 209–212",
            "doi": "https://doi.org/10.1038/s41586-019-0980-2",
            "relevance": "Demonstrates the theoretical mapping of classical features into non-linear quantum Hilbert spaces using parameterized rotation gates and entangling unitary circuits, providing the architectural foundation for our Qiskit feature-encoding stage.",
        },
        {
            "title": "Communication-Efficient Learning of Deep Networks from Decentralized Data",
            "authors": "McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A.",
            "year": "2017",
            "journal": "Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)",
            "doi": "https://proceedings.mlr.press/v54/mcmahan17a.html",
            "relevance": "Introduces the Federated Averaging (FedAvg) algorithm, allowing distributed nodes to perform local stochastic gradient descent iterations and aggregate parameter updates without centralizing raw training datasets.",
        },
        {
            "title": "Calibrating Noise to Sensitivity in Private Data Analysis",
            "authors": "Dwork, C., McSherry, F., Nissim, K., & Smith, A.",
            "year": "2006",
            "journal": "Theory of Cryptography Conference (TCC), 265–284",
            "doi": "https://doi.org/10.1007/11681878_14",
            "relevance": "Establishes the mathematical formulation of Differential Privacy and the Laplace mechanism, which bounds information leakage by injecting noise proportional to L1 sensitivity and the privacy budget epsilon.",
        },
        {
            "title": "A systematic review of quantum machine learning for digital health",
            "authors": "Gupta, R., et al.",
            "year": "2025",
            "journal": "Digital Health / PubMed",
            "doi": "https://pubmed.ncbi.nlm.nih.gov/40316703/",
            "relevance": "Reviews current experimental developments, limitations, and realistic translational hurdles in applying quantum machine learning algorithms to healthcare diagnostics, underscoring the necessity of honest benchmarking against classical baselines.",
        },
    ]

    for p in papers:
        st.markdown(
            f"""
            <div class="card-box">
                <h4>📄 {p['title']} ({p['year']})</h4>
                <p><em>{p['authors']}</em><br>
                <strong>Publication:</strong> {p['journal']}<br>
                <strong>DOI Link:</strong> <a href="{p['doi']}" target="_blank">{p['doi']}</a></p>
                <p><strong>Relevance to AegisEdge-Q:</strong> {p['relevance']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------
# Section 8: Limitations & Roadmap
# ---------------------------------------------------------
elif menu_option == "⚠️ Limitations & Technical Roadmap":
    st.header("Limitations & Technical Roadmap")

    st.markdown(
        """
        ### Honest Technical Disclosures
        In compliance with scientific rigor and competition standards, we document the precise boundaries
        of this prototype:
        """
    )

    lim_c1, lim_c2 = st.columns(2)
    with lim_c1:
        st.subheader("Current Prototype Boundaries")
        st.markdown(
            """
            1. **No Real Quantum Advantage Claim:**
               Our 3-qubit circuit runs on a classical simulator (Qiskit Aer). Simulating 3 qubits on classical hardware does not offer computational speedup over classical logistic regression.
            2. **Exploratory State Preparation vs. Clinical Diagnosis:**
               Unsupervised projective measurement counts reflect quantum state tomography, not an independently trained diagnostic classifier.
            3. **Simulated Institutional Nodes:**
               Federated clients operate in a local multi-process simulation rather than across separate physical servers.
            4. **Synthetic Data Sandbox:**
               Vital signs are mathematically generated for demonstration purposes and are not real hospital patient records.
            """
        )
    with lim_c2:
        st.subheader("Future Engineering Roadmap")
        st.markdown(
            """
            1. **Trainable Variational Quantum Classifiers (VQC):**
               Integrate Parameterized Quantum Circuits (PQC) trained with Parameter Shift Rule or SPSA gradient optimization.
            2. **Quantum Key Distribution (BB84 Protocol):**
               Implement simulated photon polarization state exchange for tamper-evident cryptographic key distribution between nodes.
            3. **Secure Enclave Integration:**
               Deploy federated aggregation within Confidential Computing hardware enclaves (Intel SGX / AWS Nitro).
            4. **Real Clinical Dataset Benchmarking:**
               Benchmark on anonymized clinical datasets (such as PhysioNet MIMIC-IV) with formal IRB approvals.
            """
        )

# Footer
st.divider()
st.caption("AegisEdge-Q Prototype | Team Innovista (HT91) | Built with Streamlit, Qiskit, Qiskit Aer, and Scikit-learn.")
