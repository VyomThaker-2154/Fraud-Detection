# Proactive Fraud Detection

**Repository:** Fraud Detection Analysis
**Notebook:** `fraud_detect.ipynb`
**Author (notebook):** Vyom

---

## Project summary

This project contains an end‑to‑end exploratory analysis and machine learning pipeline for detecting fraudulent transactions using the provided `Fraud.csv` dataset. The notebook walks through data loading, cleaning, feature engineering, imbalance handling, model training and hyperparameter tuning, model interpretation (feature importance + SHAP), and business‑facing insights and recommendations.

The goal is to provide a production‑ready analysis that data science, ML engineering, and risk teams can use to move toward a real‑time scoring pipeline.

---

## Table of contents

1. Project overview
2. Dataset
3. Key notebook sections
4. Prerequisites / environment
5. How to run
6. Expected outputs and artifacts
7. Modeling details
8. Evaluation & interpretation
9. Productionization checklist
10. Limitations & future work
11. Contact

---

## 1) Project overview

* **Type:** Binary classification (fraud vs non‑fraud)
* **Primary model:** XGBoost (tuned via `GridSearchCV`) — identified as the best performing model in the notebook
* **Imbalance strategy:** SMOTE (oversampling of minority class)
* **Interpretability:** Feature importance (XGBoost) + SHAP values for local/global explanations
* **Preprocessing highlights:** IQR outlier clipping, log transforms of monetary fields, engineered balance deltas & ratios, time features (hour/day), one‑hot encoding for transaction type

---

## 2) Dataset

* **File expected:** `Fraud.csv` (place in the same folder as the notebook when running)
* **Primary fields used:** `step` (time), `type` (transaction type), `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, `isFraud` (target) and other system/flag fields present in the CSV
* **Note:** The notebook derives additional columns such as `hour`, `day`, log‑transformed monetary features, `origin_balance_delta`, `dest_balance_delta`, and ratio features.

---

## 3) Key notebook sections (high level)

* **1. Imports & environment setup** — packages and helper functions.
* **2. Data loading & basic EDA** — shape, missing values, distribution of target by merchant/transaction type.
* **3. Outlier handling** — IQR clipping (implemented via `handle_outliers`) and before/after visuals.
* **4. Multicollinearity check** — VIF calculation and correlation heatmap for monetary features.
* **5. Feature engineering** — log transforms, deltas, ratios, time features, and encoding transaction type.
* **6. Train/test split & scaling** — `StandardScaler` used for numeric features.
* **7. Class imbalance** — SMOTE oversampling on training data.
* **8. Model training & tuning** — baseline models and GridSearchCV to tune XGBoost hyperparameters.
* **9. Model evaluation** — accuracy, precision, recall, F1, ROC AUC and confusion matrix analysis.
* **10. Interpretability** — feature importance plot and SHAP summary/force plots.
* **11. Business insights & recommendations** — actionable items for monitoring and model deployment.

---

## 4) Prerequisites / environment

Recommended Python environment (one option using `pip`):

```bash
python >= 3.8
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn shap statsmodels jupyterlab
```

Alternatively use `conda` to create an isolated environment.

> Note: SHAP requires a C++ build toolchain for speed, but it installs on most platforms via pip as shown above.

---

## 5) How to run

1. Place `Fraud.csv` in the repository root (same folder as `fraud_detect.ipynb`).
2. Start JupyterLab / Notebook:

```bash
jupyter lab
# or
jupyter notebook
```

3. Open `fraud_detect.ipynb` and run cells sequentially. The notebook is organized to run top‑to‑bottom.

**Optional**: convert the notebook to a script for automated runs:

```bash
jupyter nbconvert --to script fraud_detect.ipynb
python fraud_detect.py
```

(If converting to a script, ensure path variables and interactive plotting calls are handled appropriately.)

---

## 6) Expected outputs and deliverables

* Exploratory plots (distributions, boxplots, correlation heatmap, time‑based histograms)
* Cleaned dataset preview and final feature matrix (`X`) and labels (`y`)
* Model training logs and best hyperparameters (GridSearchCV results)
* Evaluation metrics (accuracy, precision, recall, F1, ROC AUC) printed to the notebook
* Feature importance chart (XGBoost) and SHAP summary plots
* Business insights printed in the final notebook cell describing monitoring and next steps

**Artifacts to save (recommended):**

* `best_model_xgb.joblib` or `best_model_xgb.pkl` after training
* `feature_list.json` recording the final feature order used by the model
* Evaluation report (CSV / JSON) with metrics and confusion matrix

---

## 7) Modeling details (concise)

* **Feature preprocessing:** IQR clipping for outliers, log(1+x) transforms for monetary fields, engineered deltas and ratios, one‑hot encoding for categorical transaction type, scaling with `StandardScaler`.
* **Imbalance handling:** SMOTE applied only to training data after train/test split.
* **Models explored:** Logistic Regression, Random Forest, XGBoost (notebook chooses tuned XGBoost as final candidate).
* **Hyperparameter tuning:** `GridSearchCV` on XGBoost hyperparameters such as `max_depth`, `n_estimators`, `learning_rate`, etc.
* **Interpretability:** Tree‑based feature importance and SHAP (global and per‑example) used to explain model decisions.

---

## 8) Evaluation & interpretation

* Evaluate with multiple metrics: precision and recall are critical for fraud detection (emphasize recall and precision tradeoff depending on business cost of false positive vs false negative).
* Use ROC AUC for model ranking; however threshold tuning on predicted probabilities is necessary for production (choose threshold by maximizing business utility or via Precision‑Recall curve).
* SHAP plots provide both global feature importance and local explanations for flagged transactions — extremely useful for audit trails and compliance.

---

## 9) Productionization checklist (next steps to convert notebook into production)

* Convert notebook into modular Python scripts or a package with clear interfaces (preprocessing, predict, explain).
* Persist preprocessing pipeline (scaler, feature list, encoders) and model with `joblib` or `pickle`.
* Expose a prediction API (FastAPI or Flask) that accepts raw transaction payloads and returns prediction + top SHAP contributors.
* Implement streaming or micro‑batch scoring (Kafka/Flask + Redis) for near‑real‑time detection.
* Monitoring: model drift, data drift (via PSI), alerting on spike in predicted fraud rates and false positives.
* Logging and audit trail: store input features, model scores, SHAP values and final decision for each flagged transaction.
* Retraining strategy: scheduled retrain (weekly/monthly) or trigger‑based retrain when drift exceeds thresholds.

---

## 10) Limitations & suggested improvements

* **Data limitations:** training distribution may not represent future fraud patterns. Maintain continuous labeling and feedback loop.
* **Feature scope:** consider adding entity enrichment (customer profiles, device fingerprinting, IP geolocation, transaction velocity features).
* **Imbalance & cost sensitivity:** SMOTE helps, but consider ensemble strategies, cost‑sensitive learning or anomaly detection models for rare fraud types.
* **Explainability at scale:** SHAP is powerful but computationally heavy; investigate faster approximations or limit SHAP to flagged transactions only.

---

## 11) Contact / Author

Notebook author: Vyom (as included in the notebook header)

For changes, questions or production support requests, leave a short issue or reach out to the data science team owning this notebook.

---

*End of README*
