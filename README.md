# 🏦 Credit Default Prediction – Logistic Regression

A project to predict the probability of customer default based on demographic and financial data. The dataset is provided by the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) competition on Kaggle.

## 🧠 Project Goal

Build a logistic regression model that:
- predicts the probability of a customer defaulting,
- handles anomalies, outliers, and missing values,
- is robust and generalizes well to new data.

---

## 📊 Dataset Overview

### Main dataset:
- 150,000 rows, 11 features
- Target: `target` (1 = default within 2 years, 0 = no default)

### Key characteristics:
- **Missing values**:
  - `monthly_income` ~20%
  - `dependents` ~2.6%
- **Anomalies**:
  - `age` = 0 → invalid
  - `revolving_utilization`, `debt_ratio` > 50,000
  - `past_due_*` = 98 → likely coded as "98 or more"
- **Imbalanced target**: only ~6.7% of customers defaulted

---

## 🧼 Data Preprocessing

- Removed entries with `age < 18` and `open_credit_lines > 30`
- Capped extreme values using quantile clipping
- Missing values:
  - `monthly_income` → median
  - `dependents` → 0 (interpreted as "no dependents")
- Applied `StandardScaler` to numerical features
- Used a `Pipeline` to ensure consistent preprocessing on test data

---

## 🧪 Model

**Logistic Regression**, parameters:
- `max_iter=3000`
- `class_weight='balanced'` to handle class imbalance

### Evaluation Metric:
- **ROC-AUC = 0.84** — strong performance
- The model reliably distinguishes between default and non-default clients

---

## 📈 ROC Curve & Threshold Selection

A ROC curve was built to analyze the trade-off between sensitivity (recall) and specificity.  

---


📚 Libraries Used
pandas, numpy

scikit-learn

seaborn, matplotlib

phik (for advanced correlation analysis)

✅ Summary
Logistic Regression achieved strong, stable results

Data preprocessing handled imbalance, missing values, and outliers

A reproducible pipeline was implemented for both training and inference

🚀 Future Improvements
Try more complex models (e.g., Random Forest, XGBoost)

Use GridSearchCV to tune hyperparameters

Automate threshold selection based on business KPIs