# CreditWise-Loan-System
End-to-end supervised ML pipeline using KNN, Logistic Regression and Naive Bayes to predict loan approval. Binary classification with EDA, feature engineering and model evaluation (Precision, Recall, F1).


![Python](https://img.shields.io/badge/Python-3.x-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange) ![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626) ![Status](https://img.shields.io/badge/Task-Binary%20Classification-green)

---

## 📌 Overview

**CreditWise Loan System** is a machine learning mini project that predicts whether a loan application will be **Approved (Yes)** or **Rejected (No)** based on the applicant's financial and personal profile. The project covers the full ML workflow: data cleaning, missing-value imputation, EDA, encoding, correlation analysis, feature scaling, feature engineering, model training and metric-based model selection.

## 🎯 Problem Statement

Banks receive a large number of loan applications and need a fast, consistent way to screen them. This project frames it as a **supervised binary classification** problem: given applicant details, predict `Loan_Approved` (1 = Yes, 0 = No).

## 📊 Dataset

- **File:** `loan_approval_data.csv`
- **Size:** 1,000 records × 20 columns
- **Target:** `Loan_Approved` (≈ 70% rejected / 30% approved, an imbalanced dataset)
- **Features:** Applicant & co-applicant income, employment status, age, marital status, dependents, credit score, existing loans, DTI ratio, savings, collateral value, loan amount, loan term, loan purpose, property area, education level, gender, employer category
- **Data quality:** 50 missing values (5%) in every column

## 🔄 Project Workflow

1. **Data loading & inspection**: shape, `info()`, `describe()`, null counts
2. **Missing-value handling**: `SimpleImputer` with mean (numeric) and most-frequent (categorical)
3. **Exploratory Data Analysis (EDA)**: target pie chart, categorical bar charts, income histograms, boxplots by loan status, credit-score distribution by approval
4. **Preprocessing**: dropped `Applicant_ID`; Label Encoding (education, target); One-Hot Encoding with `drop='first'` for the other categorical columns
5. **Correlation analysis**: heatmap + correlation of every feature with the target
6. **Train-test split (80/20) & feature scaling** with `StandardScaler` (fit on train only, so no data leakage)
7. **Baseline model training**: Logistic Regression, KNN (k=5), Gaussian Naive Bayes
8. **Model evaluation**: Accuracy, Precision, Recall, F1-score, Confusion Matrix
9. **Feature engineering**: squared `Credit_Score` and `DTI_Ratio`, then retraining and re-evaluation
10. **Final model selection** based on the evaluation metrics

## 🔍 Key Insights from EDA

- The dataset is **imbalanced** (~70% / 30%), so Precision, Recall and F1 matter more than accuracy alone.
- **`Credit_Score` (corr ≈ +0.45)** and **`DTI_Ratio` (corr ≈ −0.44)** are by far the strongest predictors of approval.
- No applicant with a credit score below ~650 was approved, so the score behaves like a near cut-off.
- Income, savings and age show almost no separation between approved and rejected applicants.

## 🤖 Models Used

| Model | Type | Notes |
|---|---|---|
| Logistic Regression | Linear classifier | Best final model |
| K-Nearest Neighbors (k=5) | Distance-based | Weakest performer |
| Gaussian Naive Bayes | Probabilistic | Best baseline precision |

## 📈 Results

**Baseline (before feature engineering)**

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 0.865 | 0.783 | 0.770 | 0.777 |
| KNN | 0.760 | 0.627 | 0.525 | 0.571 |
| Naive Bayes | 0.865 | 0.804 | 0.738 | 0.769 |

**After feature engineering**

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| **Logistic Regression** | **0.875** | **0.790** | **0.803** | **0.797** |
| KNN | 0.755 | 0.620 | 0.508 | 0.559 |
| Naive Bayes | 0.865 | 0.783 | 0.770 | 0.777 |

✅ **Logistic Regression** is the final model: after feature engineering it achieved the best Accuracy (87.5%), Precision, Recall and F1-score of the three models.

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `scikit-learn` · `Jupyter Notebook`

## 📁 Repository Structure

```
CreditWise-Loan-System/
│── CreditWise_Loan_System.ipynb   # Complete project notebook
│── loan_approval_data.csv         # Dataset (add this file)
│── README.md
```

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/GadgetWise-Loan-System.git
cd CreditWise-Loan-System

# 2. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# 3. Launch the notebook
jupyter notebook CreditWise_Loan_System.ipynb
```

> Make sure `loan_approval_data.csv` is in the same folder as the notebook.

## 🚀 Future Improvements

- Hyperparameter tuning with `GridSearchCV`
- Handle class imbalance (SMOTE / `class_weight='balanced'`)
- Cross-validation for more reliable metrics
- Try tree-based models (Random Forest, XGBoost)
- Deploy as a web app (Streamlit / Flask)

## 🏷️ Keywords

`machine-learning` `supervised-learning` `binary-classification` `loan-approval-prediction` `logistic-regression` `knn` `naive-bayes` `eda` `feature-engineering` `data-preprocessing` `model-evaluation` `precision-recall-f1` `confusion-matrix` `scikit-learn` `python`

## 👤 Author

**Alhin Arabia** · [GitHub](https://github.com/alhinarabia)

⭐ If you found this project useful, consider giving it a star!
