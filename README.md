# 💼 Credit Default Prediction & ECL Modeling

## 📌 Project Objective

This project aims to build a robust and interpretable machine learning pipeline to predict the **Probability of Default (PD)** and calculate the **Expected Credit Loss (ECL)** for loan applicants. It follows a simplified **IFRS 9 framework** using publicly available data.

> 💡 **ECL Formula**:  
**ECL = PD × LGD × EAD**

Where:
- **PD** — Probability of Default (learned via ML)
- **LGD** — Loss Given Default (assumed as 0.45)
- **EAD** — Exposure at Default (estimated as `debt_ratio × monthly_income`)

---

## 📂 Dataset

- Source: [Kaggle — Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)
- Rows: 150,000
- Target: `SeriousDlqin2yrs` — binary indicator of default within 2 years
- No real LGD or EAD columns → assumptions were made for ECL estimation

---

## 🛠️ Technologies Used

- **Python 3.x**
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn` (pipelines, preprocessing, GridSearchCV)
- `lightgbm` for gradient boosting
- `shap` for explainability

---

## 🧪 Workflow Overview

### 1. **Data Cleaning**
- Removed duplicates
- Clipped extreme outliers
- Filled missing values:
  - `monthly_income` → median
  - `dependents` → 0
- Clipped values like `past_due_*` to max 10 (e.g., coded 98 = 98+ days)

### 2. **Model Training**
Trained 3 models using full pipeline and `GridSearchCV`:

| Model                | CV ROC-AUC | Validation ROC-AUC |
|---------------------|------------|---------------------|
| Logistic Regression | 0.8557     | 0.8554              |
| Random Forest       | 0.8644     | 0.8629              |
| ✅ LightGBM (Best)   | 0.8655     | 0.8644              |

> All models included consistent preprocessing: imputation, scaling, clipping, and class balancing.

### 3. **Model Explainability**
- Used **SHAP** to analyze the impact of each feature
- Top features:
  - `revolving_utilization` (credit usage)
  - `past_due_*` variables (delinquencies)
  - `age` (younger → more risky)
- Weak features:
  - `dependents`, `real_estate_loans` (flat SHAP)

### 4. **ECL Estimation**
- Set **LGD = 0.45** (industry proxy)
- Estimated **EAD = debt_ratio × monthly_income**
- Calculated ECL for both train and test datasets
- Exported test results as: `ECL_results.csv`

### 5. **ECL Insights**
- Most clients have low ECL → healthy portfolio
- Small segment of clients contributes to large portion of credit risk
- Top 10 clients with highest ECL identified for risk-based actions

---

## 📊 Key Visuals

- ROC Curve and AUC scores
- SHAP Summary and Dependence Plots
- Distribution of ECL across the portfolio
- Table of top clients by ECL

---

## 📝 Conclusion

✅ We built a production-ready pipeline to calculate ECL using public data.  
⚙️ All data processing and modeling is done within scikit-learn pipelines.  
📈 Model performance is strong and interpretable (LightGBM ROC-AUC ≈ 0.864).  
📦 Final predictions (PD, LGD, EAD, ECL) saved to `ECL_results.csv`.

> 📁 The project is ready to be extended to real-world systems with actual LGD/EAD inputs or connected to internal credit databases.

---

## 📁 Files in Repository

- `give_me_some_credit_v5.ipynb` — full notebook with code, results, and visuals
- `ECL_results.csv` — final table of predictions for test data
- `README.md` — project overview
