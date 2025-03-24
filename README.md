# CA04_Ensemble_Models

## Predicting Income Based on Census Data

### Assignment Overview

This project focuses on using **Ensemble Machine Learning Models** to classify whether a person earns over $50K per year based on census demographic data. The dataset includes binned versions of demographic features like age, education, hours worked, and more.

We apply and evaluate four key ensemble classifiers:
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost

---

## Dataset

- **Source**: U.S. Census Bureau  
- **Size**: 48,842 instances  
- **Target**: Binary classification — `<=50K` (0) or `>50K` (1)  
- **Split**: Pre-labeled `train` and `test` sets (column: `flag`)

🔗 [Dataset Link](https://github.com/ArinB/MSBA-CA-03-Decision-Trees/blob/master/census_data.csv?raw=true)

---

## Preprocessing Summary

- Stripped whitespace from categorical features
- Applied:
  - `Ordinal Encoding` to ordered features
  - `One-Hot Encoding` to nominal features
- Aligned training and test feature columns after encoding

---

## Section Summaries

### Section 2 – Hyperparameter Tuning: `n_estimators`

For each model, we tested multiple values of `n_estimators` and plotted:
- **Accuracy vs. n_estimators**
- **AUC Score vs. n_estimators**

This helped identify optimal tree counts for performance and efficiency.

---

### Section 3 – Random Forest

- Accuracy forms a shallow U-shape with lowest performance around 250–350 estimators.
- Optimal performance observed at `n_estimators = 50`, but performance differences across all values were minor.

---

### Section 4 – AdaBoost, Gradient Boosting, XGBoost

Each model was evaluated in the same way as Random Forest:

#### AdaBoost
- **Best Accuracy**: At `n_estimators = 100`
- **Best AUC**: Plateaued at `n_estimators = 100`
- Minor decline after peak — indicating diminishing returns.

#### Gradient Boosting
- **Best Accuracy**: At `n_estimators = 100`
- **Best AUC**: Slightly higher at `n_estimators = 150`
- Accuracy and AUC plateau beyond that.

#### XGBoost
- **Best Accuracy & AUC**: At `n_estimators = 50`
- Performance consistently declined with higher estimators, likely due to overfitting.

---

## Section 5 – Performance Comparison Table

| Model            | Best Accuracy | Accuracy @ n_estimators | Best AUC  | AUC @ n_estimators |
|------------------|----------------|--------------------------|-----------|---------------------|
| Random Forest     | 0.838X         | 50                       | 0.90XX    | 250                 |
| AdaBoost          | 0.8461         | 100                      | 0.8974    | 100                 |
| Gradient Boosting | 0.8482         | 100                      | 0.8993    | 150                 |
| XGBoost           | 0.8450         | 50                       | 0.8966    | 50                  |

> *Gradient Boosting showed the best overall AUC and accuracy in this assignment.*


