# 🛡️ AegisEdge-Q: Privacy-Preserving Quantum Healthcare AI

**Team Name:** Innovista  
**Team ID:** `HT91`  
**License:** MIT  
**Prototype Version:** 1.0.0

---

## 📌 Executive Summary

**AegisEdge-Q** is an edge-first, privacy-preserving healthcare analytics prototype designed to resolve a core dilemma in modern digital medicine: **How can medical institutions train and evaluate predictive AI models without exposing sensitive Patient Health Information (PHI) or depending on centralized external cloud APIs?**

AegisEdge-Q solves this challenge through a multi-tier decentralized framework combining:
1. **Classical Healthcare ML Pipeline:** Reproducible synthetic vital signs generation, leak-free scikit-learn preprocessing, and dynamic evaluation metrics (accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix).
2. **Real Qiskit Aer Quantum Circuit Simulation:** Real local quantum circuit execution using `qiskit` and `qiskit-aer` (`AerSimulator`), featuring 3-qubit parameterized state preparation ($R_y$ angle encoding), entangling topologies ($CX$ / $CZ$), projective measurements, bitstring probability distributions, and per-qubit Pauli-Z expectation values $\langle Z_i \rangle$.
3. **Decentralized Federated Learning (FedAvg):** 3-institution decentralized training simulation (Hospital Alpha, Clinic Beta, Health Center Gamma) aggregating only model weight vectors with strict data locality.
4. **Differential Privacy (DP) Engine:** Calibrated Laplace noise injection ($\epsilon$) demonstrating the mathematical privacy-utility tradeoff against gradient inversion and membership inference attacks.
5. **Interactive Streamlit Dashboard:** Professional clinical UI with real-time inference, simulator visualizer, metrics heatmaps, and zero required initial file uploads.

> [!IMPORTANT]
> **Ethical & Scientific Disclaimer:**  
> All vital signs and patient records used within this prototype are **100% synthetic demonstration data**. This project is an algorithmic and computational prototype. It is **not** clinically validated, **not** hospital-ready, and **must not** be used for medical diagnosis or clinical decision-making.

---

## 🏗️ System Architecture

```
                    ┌────────────────────────────────────────────────────────┐
                    │               AEGISEDGE-Q ARCHITECTURE                 │
                    └────────────────────────────────────────────────────────┘

  [ INSTITUTIONAL EDGE SILOS ]
  ┌───────────────────────┐   ┌───────────────────────┐   ┌───────────────────────┐
  │    Hospital Alpha     │   │ Regional Clinic Beta  │   │  Health Center Gamma  │
  │   (Tertiary Care)     │   │      (Outpatient)     │   │  (Academic Research)  │
  │  - Local Data Silo    │   │  - Local Data Silo    │   │  - Local Data Silo    │
  │  - Private Training   │   │  - Private Training   │   │  - Private Training   │
  └──────────┬────────────┘   └──────────┬────────────┘   └──────────┬────────────┘
             │ w_alpha                   │ w_beta                    │ w_gamma
             │ (Local Weights)           │ (Local Weights)           │ (Local Weights)
             └─────────────────────┬─────┴───────────────────────────┘
                                   │  (RAW PATIENT DATA NEVER SENT)
                                   ▼
  [ FEDERATED AGGREGATION & DIFFERENTIAL PRIVACY COORDINATOR ]
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  1. Federated Averaging (FedAvg):                                       │
  │     w_global = sum_{k=1}^K (n_k / N) * w_k                              │
  │                                                                         │
  │  2. Differential Privacy Perturbation (Laplace Mechanism):              │
  │     w_global_DP = w_global + Laplace(0, Delta / epsilon)                │
  └────────────────────────────────┬────────────────────────────────────────┘
                                   │
                                   ▼
  [ QUANTUM FEATURE TIER - LOCAL QISKIT AER ]
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  - Parameterized 3-Qubit Circuit: |psi(theta)>                          │
  │  - Angle Encoding: theta_i = pi * x_norm_i via R_y(theta)               │
  │  - Entanglement Ring: CX(q0, q1), CX(q1, q2), CX(q2, q0)                │
  │  - Simulation: Local AerSimulator (No paid APIs or cloud keys)          │
  │  - Output: Measurement counts {000, 001, ..., 111}, Expectation <Z_i>   │
  └────────────────────────────────┬────────────────────────────────────────┘
                                   │
                                   ▼
  [ STREAMLIT CLINICAL DASHBOARD ]
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  - Patient Risk Estimator (Classical Pipeline)                          │
  │  - Real Qiskit Circuit Diagram & Measurement Distributions              │
  │  - Dynamic Metrics, Confusion Matrix Heatmap, ROC Curve                 │
  │  - Multi-Round FedAvg Convergence Visualizer                            │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
AegisEdgeQ/
├── app.py                      # Interactive Streamlit dashboard application
├── requirements.txt            # Python dependencies (tested with Python 3.11 / 3.13)
├── README.md                   # Comprehensive documentation and launch guide
├── .gitignore                  # Git exclusions for Python caches and artifacts
├── src/
│   ├── __init__.py             # Package declaration
│   ├── healthcare_ml.py        # Synthetic dataset generator, ML pipeline, inference & metrics
│   ├── quantum_circuit.py      # Qiskit circuit builder, angle encoder, Aer simulator execution
│   └── federated_demo.py       # 3-client partitioning, FedAvg aggregator, Laplace DP noise
├── tests/
│   ├── __init__.py             # Test package declaration
│   ├── test_healthcare_ml.py   # Pytest suite for ML generator, pipeline, and metrics
│   ├── test_quantum_circuit.py # Pytest suite for Qiskit circuit and Aer simulation
│   └── test_federated_demo.py  # Pytest suite for FedAvg aggregation and DP noise
└── data/
    └── .gitkeep                # Local data placeholder
```

---

## ⚡ Quick Start & Installation

### 1. Prerequisites
- **Python:** 3.11 or 3.13 installed (64-bit).
- **Git** (optional).
- Windows, macOS, or Linux.
- **No quantum hardware accounts, credit cards, or external API keys are required.**

### 2. Install Dependencies
Open your PowerShell, Command Prompt, or terminal in the project directory:

```bash
python -m pip install -r requirements.txt
```

### 3. Run Automated Tests
Execute the automated `pytest` suite to verify that the quantum simulator, machine learning pipeline, and federated learning modules operate correctly:

```bash
python -m pytest -v
```

### 4. Launch the Streamlit Dashboard
Start the interactive application:

```bash
python -m streamlit run app.py
```

The application will open automatically in your default web browser at `http://localhost:8501`.

---

## 🔬 Core Capabilities

### 1. Healthcare Machine Learning (`src/healthcare_ml.py`)
- **Reproducible Synthetic Generation:** Simulates four physiological vitals:
  - `age` (18–85 years)
  - `heart_rate` (50–130 bpm)
  - `systolic_bp` (90–190 mmHg)
  - `body_temp` (36.0–39.5 °C)
- **Ground-Truth Labeling:** Deterministic logistic risk formula modeling multi-vital instability without using real patient records.
- **Strict Leak-Free Pipeline:** `SimpleImputer` $\rightarrow$ `StandardScaler` $\rightarrow$ `LogisticRegression`. Preprocessors are fitted strictly on the training partition.
- **Dynamic Metrics (Never Hardcoded):** Evaluates held-out test predictions dynamically:
  - Accuracy, Precision, Recall, F1-score
  - 2x2 Confusion Matrix heatmap
  - ROC-AUC and empirical ROC curve
- **Sanitized Patient Inference:** Validates user inputs against physiological thresholds before executing inference.

### 2. Real Qiskit Aer Quantum Circuit Simulator (`src/quantum_circuit.py`)
- **Real Qiskit Execution:** Imported from `qiskit` and `qiskit_aer.AerSimulator`. Modern Qiskit 1.x/2.x APIs are used without deprecated calls.
- **State Preparation ($R_y$ Angle Encoding):** Normalizes continuous classical features into rotation angles $\theta_i \in [0, \pi]$:
  $$\theta_i = \pi \cdot \left(\frac{x_i - \min(x_i)}{\max(x_i) - \min(x_i)}\right)$$
- **Entanglement Layer:** Configurable $CX$ (CNOT) or $CZ$ topologies entangling the 3 qubits into non-separable quantum states.
- **Phase Modulation:** Optional $R_z$ variational rotation layer.
- **Measurement & Tomography:** Projective measurements onto classical registers returning exact counts and bitstring probabilities.
- **Expectation Values:** Calculates per-qubit Pauli-Z expectations:
  $$\langle Z_i \rangle = \frac{N(|0\rangle) - N(|1\rangle)}{N_{shots}} \in [-1, 1]$$
- **Visual Circuit Rendering:** Displays Matplotlib circuit schematics and ASCII/Unicode diagrams.
- **Scientific Demarcation:** Clearly documents that circuit measurement is a quantum state preparation simulation, not an unsupervised clinical diagnostic.

### 3. Decentralized Federated Learning (`src/federated_demo.py`)
- **3 Simulated Institutional Nodes:**
  - `Hospital Alpha`: Metropolitan Tertiary Care Center (~45% sample allocation)
  - `Regional Clinic Beta`: Suburban Outpatient Facility (~35% sample allocation)
  - `Health Center Gamma`: University Academic Center (~20% sample allocation)
- **Data Locality:** Client datasets remain isolated in their respective local environments.
- **Federated Averaging (FedAvg):** The coordinator aggregates local model weight vectors:
  $$w_{\text{global}} = \sum_{k=1}^K \frac{n_k}{N} w_k, \quad b_{\text{global}} = \sum_{k=1}^K \frac{n_k}{N} b_k$$
- **Differential Privacy ($\epsilon$):** Injects Laplace noise calibrated to $L_1$ sensitivity into the aggregated weights:
  $$\text{Noise} \sim \text{Laplace}\left(0, \frac{\Delta_1}{\epsilon}\right)$$
  Enables interactive investigation of the privacy-utility tradeoff (lower $\epsilon$ = higher privacy with lower test accuracy).

---

## 📚 Foundational Research References

1. **Havlíček et al. (2019)**  
   *Supervised learning with quantum-enhanced feature spaces.*  
   *Nature*, 567(7747), 209–212.  
   DOI: [https://doi.org/10.1038/s41586-019-0980-2](https://doi.org/10.1038/s41586-019-0980-2)  
   *Relevance:* Demonstrates mapping classical inputs into quantum Hilbert spaces via parameterized unitary circuits and entangling gates, forming the foundation of our Qiskit feature-encoding stage.

2. **McMahan et al. (2017)**  
   *Communication-Efficient Learning of Deep Networks from Decentralized Data.*  
   *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)*.  
   Paper: [https://proceedings.mlr.press/v54/mcmahan17a.html](https://proceedings.mlr.press/v54/mcmahan17a.html)  
   *Relevance:* Origin of the Federated Averaging (FedAvg) algorithm, enabling decentralized training across clinical institutions without pooling raw patient records.

3. **Dwork et al. (2006)**  
   *Calibrating Noise to Sensitivity in Private Data Analysis.*  
   *Theory of Cryptography Conference (TCC)*, 265–284.  
   DOI: [https://doi.org/10.1007/11681878_14](https://doi.org/10.1007/11681878_14)  
   *Relevance:* Establishes the mathematical formulation of Differential Privacy and the Laplace mechanism utilized in our privacy coordinator.

4. **Gupta et al. (2025)**  
   *A systematic review of quantum machine learning for digital health.*  
   *PubMed / Digital Health*.  
   URL: [https://pubmed.ncbi.nlm.nih.gov/40316703/](https://pubmed.ncbi.nlm.nih.gov/40316703/)  
   *Relevance:* Reviews the current state of QML in healthcare, emphasizing rigorous classical benchmarking, NISQ constraints, and translational requirements.

---

## 🛠️ Troubleshooting & FAQ

### 1. `qiskit-aer` on Windows
- If `pip install qiskit-aer` encounters issues, ensure you are using standard 64-bit Python (Python 3.11 or 3.13). Pre-built binary wheels (`cp313-win_amd64` / `cp311-win_amd64`) are distributed on PyPI and do not require Microsoft Visual C++ Build Tools.

### 2. Streamlit Port Conflicts
- If port `8501` is already in use by another application:
  ```bash
  python -m streamlit run app.py --server.port 8502
  ```

### 3. Headless / Server Environments
- To run Streamlit in a non-GUI or automated container:
  ```bash
  python -m streamlit run app.py --server.headless=true
  ```

---

## 🏆 Final Acceptance Verification

- [x] Project files created and organized directly in workspace root.
- [x] Dependencies tested and installed via `requirements.txt`.
- [x] Real Qiskit Aer circuit builder and execution verified (`AerSimulator`).
- [x] Scikit-learn classical ML pipeline with dynamic metric computation verified.
- [x] 3-client Federated Learning (FedAvg) and Laplace Differential Privacy verified.
- [x] Complete automated unit test suite (`pytest`) passing 100%.
- [x] Streamlit web application verified and runnable out-of-the-box.
