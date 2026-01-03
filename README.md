<<<<<<< HEAD
# Hypertension-Risk-Prediction
Kaggle hypertension research project. Explained 

# Hypertension risk prediction
---
Hi there! This is **my second Kaggle project**. Welcome to **HYPERTENSION RISK PREDICTION** project. This project uses synthetic yet realistic healthcare dataset to explore the factors contributing to hypertension and build predictive machine learning models.

## Project Objectives
- Perform exploratory data analysis (EDA)
- Build binary classification models (e.g., Logistic Regression, Random Forest, XGBoost)
- Analyze feature importance
- Interpret predictions using SHAP values and feature importances.

## 📁 Dataset Information
📄 **Name:** Hypertension Risk Prediction Dataset  
🔗 **Source:** [🩺 Hypertension Risk Prediction Dataset](https://www.kaggle.com/datasets/miadul/hypertension-risk-prediction-dataset)  
👤 **Author:** [Ejtolf](https://github.com/Ejtolf)  
📝 **Task Type:** Binary Classification (Target: `Has_Hypertension` - Yes/No

## 💻 Tools & Technologies
- **Python** (Pandas, NumPy, Matplotlib, Seaborn) to code
- **Scikit-Learn** to build binary ML models (Logistic Regression, Random Forest) 
- **XGBoost** to build gradient boost model
- **SHAP** to check for features importance 
- **Jupyter Notebook** to research.

--- 

# Project Report: Hypertension Prediction
## 1. Project Goal

Build and compare machine learning models to classify hypertension presence based on patient features. Evaluate model performance and analyze feature importance.

## 2. Models and Performance Metrics
| Metric   | Value  |
| -------- | ------ |
| Accuracy | \~0.72 |
| F1-score | \~0.73 |
| ROC AUC  | \~0.81 |

Baseline model with reasonable interpretability but lower performance compared to more complex models.

## 3. Random Forest
| Metric   | Value  |
| -------- | ------ |
| Accuracy | \~0.95 |
| F1-score | \~0.95 |
| ROC AUC  | \~0.99 |

Significant improvement in performance. Model effectively captures complex patterns. Feature bp_history was most important.

## 4. XGBoost
| Metric   | Value   |
| -------- | ------- |
| Accuracy | \~0.98  |
| F1-score | \~0.98  |
| ROC AUC  | \~0.998 |

Best-performing model overall, showing excellent generalization capability.

## 5. Feature Importance Analysis

- Both Random Forest and XGBoost identified bp_history (history of high blood pressure) as the most important feature.
- Followed by age, stress score, BMI, and salt intake.
- Smoking status and exercise level had lesser influence.

![SHAP-Plot: feature importances](/reports//hypertension_risk_shap.png)

# 🩺 Feature-level interpretation

---

### 🩺 `bp_history_0`, `bp_history_2` - **the most influential features**

- Previous blood pressure issues are the strongest predictors

- Clinically intuitive and robust signal

### 👪 `family_history_1`

- Family history present = higher risk

- Absent = lower risk

- Genetic predisposition plays a meaningful role

### 🎂 `age`
- Older age means higher predicted risk

- Classic age-related hypertension risk captured well

### 🚬 `smoking_status_1`
- Smoking substantially increases risk

- Non-smoking reduces it

### 😖 `stress_score`
- Higher stress increases risk (and lower decreases)

### 🧂 `salt_intake`
- High salt intake is the reason of risk growth

### 😴  `sleep_duration`
- Short sleep => increased risk

- Longer sleep => reduced risk

### ⚖️ `bmi`
- Higher BMI makes a risk higher, lower more protective

### 🏃 `exercise_level_1`, `excercise_level_2`
- Physical activity may act indirectly (via BMI or stress)

## 6. Conclusions and Recommendations

- Random Forest and XGBoost models significantly outperform Logistic Regression for hypertension classification.

- XGBoost is recommended as the primary model due to highest accuracy and robustness.

- Feature importance and SHAP analysis enhance model interpretability, valuable for clinical insights.
=======
# hypertension-risk-prediction
>>>>>>> 4a44302 (Explained & Deployed)