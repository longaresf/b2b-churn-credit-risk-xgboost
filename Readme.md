# B2B Customer Churn & Invoice Default Risk Engine (XGBoost)
[![CI Pipeline](https://github.com/longaresf/xgboost-advanced-classification-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/longaresf/xgboost-advanced-classification-pipeline/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)

This repository contains a high-performance **Machine Learning** ecosystem developed in **Python**, focused on using **XGBoost (Extreme Gradient Boosting)** to solve binary classification problems, analyze customer churn, and assess credit risk.

## Business Impact & Operational Use Case

### What business problem does this architecture solve?
In B2B distribution, SaaS, and recurring-service companies, **losing an existing client or suffering late invoice payments silently erodes cash flow**. Most teams only react *after* a customer stops ordering or an invoice becomes 60+ days overdue.

This pipeline transforms historical billing, order frequency, and payment behavior into **actionable risk probabilities**, allowing finance and commercial teams to intervene proactively.

### Key Outcomes & Capabilities
* **Early Warning Scoring:** Ranks accounts by probability of churn or payment delinquency before revenue is lost.
* **Business-Driven Threshold Optimization:** Tunes classification thresholds based on financial cost asymmetry (cost of false alarm vs. cost of lost account/unpaid invoice), rather than relying on naive 0.50 cutoffs.
* **Executive Explainability (SHAP):** Translates model predictions into human-readable risk drivers (e.g., *"Order frequency dropped 40% in the last 60 days"*), avoiding "black box" distrust from management.
* **Production-Ready MLOps:** Modular `scikit-learn` + `XGBoost` pipelines preventing data leakage, backed by automated unit tests (`pytest`) and CI/CD.

---

## Architecture Flow

```mermaid
flowchart LR
    A[ERP / Billing & Order History] --> B[Feature Engineering & Leakage-Free Split]
    B --> C[XGBoost Classifier + Hyperparameter Tuning]
    C --> D[Cost-Based Threshold Optimization]
    D --> E[Risk Scoring Table & SHAP Drivers]
    E --> F[Commercial Retention / Collections Action List]

## 🚀 Technical Capabilities and Feature Engineering

* **Cutting-Edge XGBoost Modeling:** Implementation of the `XGBClassifier` algorithm, leveraging the power of gradient transfer learning, internal regularization ($L_1$ and $L_2$), and pruning strategies to maximize predictive performance.
* **Secure Transformation Pipelines:** Integration of `ColumnTransformer` to automate isolated preprocessing workflows: missing-value imputation (`SimpleImputer`), feature scaling (`StandardScaler`), and categorical encoding (`OneHotEncoder`).
* **Ultra-Robust Validation Strategy:** Implementation of advanced resampling techniques such as **`RepeatedStratifiedKFold`** and **`RepeatedKFold`**, mitigating class imbalance and reducing the variance of performance estimates.
* **Dual and Multidimensional Evaluation:** Construction of a metric environment capable of evaluating performance in both classification scenarios (*Accuracy*, *Precision*, *Recall*, $F_1$-Score, *ROC-AUC*) and continuous regression scenarios (*MAE*, *RMSE*, $R^2$).

## 🛠️ Technology Stack

* **Core Language:** Python 3.x
* **Data Structures:** `Pandas`, `NumPy`
* **Statistical Visualization:** `Seaborn`, `Matplotlib`
* **Modeling and Preprocessing:** `Scikit-Learn` (Sklearn)
* **Core Algorithm:** `XGBoost` (eXtreme Gradient Boosting)

## ⚙️ Pipeline Architecture and Error Mitigation

The framework was designed to resolve critical challenges in deploying predictive models:

1. **Data Leakage Immunity:** By tightly packaging preprocessing (`SimpleImputer`, `StandardScaler`) together with the estimator inside a `Pipeline`, all transformation parameters are learned exclusively from the training folds.
2. **Efficient Hyperparameter Space Optimization:** Coordinated use of exhaustive searches (`GridSearchCV`) and stochastic searches (`RandomizedSearchCV`) to tune critical hyperparameters such as learning rate, tree depth, and regularization coefficients.
3. **Discriminative Capacity Analysis:** Visual evaluation of the trade-off between the true-positive and false-positive rates using ROC curves, ensuring that the model can distinguish effectively between positive and negative classes.

## 📌 System Data Flow

$$\text{Structured Data} \longrightarrow \text{Pipeline (Imputation + Encoders + Scaling)} \longrightarrow \text{Repeated Stratified K-Fold} \longrightarrow \text{Fine-Tuning (Grid/Random Search)} \longrightarrow \text{Optimized XGBoost Model}$$

## 🔧 Local Configuration and Installation

To reproduce the experiments, audit the confusion matrices, or run the model locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/longaresf/b2b-churn-credit-risk-xgboost.git
   ```
2. **Enter the working directory:**
   ```bash
   cd b2b-churn-credit-risk-xgboost
   ```

3. **Set up the virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the pipeline:**
   Open your Jupyter Notebook environment or run the main script to visualize the XGBoost learning and performance curves:

   ```bash
   python main.py  # Replace with the name of your main file if different
   ```

## ✒️ Author
    Francisco Longares - Data Scientist & AI Solutions Developer - longaresf
