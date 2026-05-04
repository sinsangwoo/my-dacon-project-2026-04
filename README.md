# 🚀 DACON Warehouse Delay Prediction: Advanced ML Pipeline

## 🎯 Overview
This repository contains a highly engineered, production-ready Machine Learning pipeline designed for the **DACON Warehouse Delay Prediction** challenge. The primary objective is to predict the `avg_delay_minutes_next_30m` for warehouse robots operating in a highly dynamic, non-stationary environment.

The system is built to combat **variance collapse, temporal leakage, and domain shift (drift)**, which are the primary causes of CV-LB (Cross-Validation vs Leaderboard) divergence in complex time-series scenarios.

---

## 🏗️ Core Architecture & Innovations

### 1. 🛡️ Single Source of Truth (SSOT) Gateway
All machine learning operations are strictly routed through a module-scoped gateway (`utils.SAFE_FIT`, `utils.SAFE_PREDICT`). This enforces:
* **Zero Silent Failures**: Strict type validation (`np.float32`) and shape constraints before reaching `LightGBM`.
* **Complete Isolation**: Prevents namespace shadowing and guarantees deterministic execution across the entire repository.

### 2. 🌀 2-Stage Structural Architecture (Power-Damped Asymmetric Execution)
To handle extreme variance in tail delays (the top 5-10% of values), the pipeline utilizes a two-stage approach:
1. **Tail Classifier**: A binary classifier trained to detect the highest risk (top 10%) scenarios.
2. **Dual Regressors**: Separate, specialized regressors for tail and non-tail behaviors.
3. **Sigmoid Gap Blending**: A dynamic blending mechanism that calculates the prediction gap between the two regressors and smoothly interpolates the final prediction, ensuring tail signals are not suppressed by global means.

### 3. 🔪 Collective Drift Suppression
The `CollectiveDriftPruner` and `DomainShiftAudit` modules form a robust defense against environmental distribution shifts. 
* Performs iterative, adversarial pruning relative to the **actual test distribution**.
* Actively neutralizes features that cause the model to memorize the training manifold, forcing the model to rely only on generalized, drift-stable signals.

### 4. 🔍 Anti-CV-Illusion & Forensic Intelligence
To ensure the validation score is an honest proxy for leaderboard performance:
* **Time-Aware CV**: Expanding-window chunking (`NFOLDS + 2`) completely isolates past and future events.
* **Forensic Logger**: Tracks generalization gaps, scenario-level dynamics, feature blindness, and resource bottlenecks, generating a comprehensive `intelligence_summary.txt` after every run.

---

## 🛤️ Pipeline Phases (`pipeline.sh`)

The system is orchestrated by a 9-phase master script that guarantees artifact isolation and reproducibility via a strict `RUN_ID` contract.

* **Phase 1: `1_reconstruct_schema`** - Base feature tracking and layout synchronization.
* **Phase 2: `2_build_base`** - Core feature engineering and data downcasting.
* **Phase 2.5: `2.5_drift_audit`** - Adversarial Domain Shift Audit and generation of the `stability_manifest`.
* **Phase 3: `3_train_raw`** - Baseline model evaluation.
* **Phase 5: `5_train_leakage_free`** - **[CORE]** Leakage-free training loop with dynamic feature pruning, latent embeddings (PCA), and time-aware splitting.
* **Phase 6: `6_calibrate`** - Post-training variance recovery and scale recalibration.
* **Phase 7: `7_inference`** - Leakage-free test inference using fold-specific statistics.
* **Phase 8: `8_submission`** - Fingerprint-verified submission generation.
* **Phase 9: `9_intelligence`** - Forensic diagnosis and top-10 risk factor reporting.

---

## 🛠️ Execution & Usage

The pipeline is strictly governed by `pipeline.sh`. **Do not run `main.py` directly without a `RUN_ID`.**

### Full Pipeline Execution
```bash
./pipeline.sh
```

### Resume from a Specific Phase
```bash
./pipeline.sh --start-phase 5_train_leakage_free
```

### Smoke Test (Rapid Verification)
Executes the entire pipeline on a tiny subset of data to verify architectural integrity.
```bash
./pipeline.sh --smoke-test
```

### Debug Mode
```bash
./pipeline.sh --debug
```

---

## 📂 Repository Structure & Data Management
Based on the `.gitignore` rules, the repository strictly separates tracked source code from volatile artifacts and data.

```text
├── data/               # [IGNORED] Raw datasets (train.csv, test.csv, layout_info.csv)
├── src/                # [TRACKED] Core pipeline logic and ML modules
├── logs/               # [IGNORED] Phase-specific execution logs, forensics, and intelligence reports
├── outputs/            # [IGNORED] Processed data, saved models, and predictions
├── .done/              # [IGNORED] State-tracking markers for pipeline.sh phase resumption
├── research/           # [IGNORED] Experimental notebooks (*.ipynb) and scratchpads
├── pipeline.sh         # [TRACKED] Master orchestration script
├── main.py             # [TRACKED] Python entrypoint
└── README.md           # [TRACKED] You are here
```

### Artifact Isolation (`RUN_ID`)
Every execution automatically generates a unique `RUN_ID` (e.g., `run_20260504_120000`). All outputs in the ignored directories are strictly isolated to prevent cross-contamination:
* **Logs**: `./logs/{RUN_ID}/`
* **Models**: `./outputs/{RUN_ID}/models/`
* **Predictions**: `./outputs/{RUN_ID}/predictions/`
* **Submissions**: `./outputs/{RUN_ID}/submission.csv`

---

## 📝 Golden Rules for Future Development
1. **Never perform local function-level imports** of `SAFE_` methods. Always use `utils.SAFE_FIT`.
2. **Never modify splitting logic in isolation**. Changes to `trainer.py` must be mirrored in `cv_reliability.py`.
3. **No naked `.fit()` or `.predict()` calls** allowed on model objects. The repository is hardened to fail if this contract is violated.
