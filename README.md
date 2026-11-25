# CIE 2 – Activity 2: Binary Classification with Logistic Regression

> **A diabetes screening model: probabilities, decision thresholds and the cost of being wrong**

| | |
|---|---|
| **Institution** | Zeal College of Engineering and Research, Narhe, Pune – 41 |
| **Department / Class** | AI & DS, T.Y. B.Tech, Semester V (Academic Year 2026–27) |
| **Course** | Regression and Classification Algorithms |
| **Activity** | CIE 2, Activity 2 (Unit II – Classification & Logistic Regression), 5 marks |
| **Student** | Manthan Raut |
| **Roll No. / Division** | AD3314 / C |
| **Submission GitHub ID** | [https://github.com/complexiota](https://github.com/complexiota) |

---

## 1. Objective

Take the role of a data scientist at a community health-screening programme. Build a Logistic Regression model for a two-class problem, convert predicted probabilities into class labels, evaluate it with the confusion matrix, accuracy, precision, recall, specificity, F1-score and ROC-AUC, compare it with a majority-class baseline, study the effect of the decision threshold, and **recommend a threshold with a justification**.

## 2. Problem statement

> Can routine measurements (glucose, BMI, age, etc.) flag people who should be referred for a confirmatory diabetes test?

| Item | Definition |
|------|-----------|
| **Positive class (1)** | Person has diabetes |
| **Negative class (0)** | Person does not have diabetes |
| **Who acts on the output** | Nurses and clinicians, who refer flagged people for a full lab test |
| **False negative (FN)** | A diabetic person is cleared, so the case is missed and untreated (**most serious error**) |
| **False positive (FP)** | A healthy person is flagged, so an extra confirmatory test is done (cost and worry, less serious) |
| **Primary metric** | Recall (sensitivity) at an acceptable precision, supported by F1 and ROC-AUC |
| **Baseline** | Always predict "no diabetes" (majority class) |

## 3. Dataset

- **Name:** Pima Indians Diabetes Database
- **Original source:** Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C. and Johannes, R.S. (1988). *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.* National Institute of Diabetes and Digestive and Kidney Diseases. https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- **CSV mirror used in the notebook:** https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv (no header row; column names assigned in code)
- **Size:** 768 rows × 9 columns (8 features + 1 target)
- **Class balance:** 500 negative (65.1%) and 268 positive (34.9%), moderately imbalanced
- **Data ethics:** Anonymised public data; no identifier column is used. It represents one specific population (women of Pima Indian heritage, aged 21 and above), so the model is **not** valid for diagnosing individuals or for other populations.

### Data dictionary

| Column | Meaning | Role |
|--------|---------|------|
| `Pregnancies` | Number of times pregnant | Feature |
| `Glucose` | Plasma glucose concentration (2-hour oral glucose tolerance test) | Feature |
| `BloodPressure` | Diastolic blood pressure (mm Hg) | Feature |
| `SkinThickness` | Triceps skin-fold thickness (mm) | Feature |
| `Insulin` | 2-hour serum insulin (mu U/ml) | Feature |
| `BMI` | Body mass index (kg/m²) | Feature |
| `DiabetesPedigreeFunction` | Diabetes family-history score | Feature |
| `Age` | Age in years | Feature |
| `Outcome` | 1 = diabetes, 0 = no diabetes | **Target** |

### Data quality findings

- No duplicate rows; labels contain only 0 and 1; no NaN values as loaded.
- **Impossible zeros** (a person cannot have a glucose, blood pressure, skin thickness, insulin or BMI of 0) were found:

| Column | Zero values (treated as missing) |
|--------|----------------------------------|
| Glucose | 5 |
| BloodPressure | 35 |
| SkinThickness | 227 |
| Insulin | 374 |
| BMI | 11 |

`Pregnancies = 0` is valid and was left unchanged.

## 4. Repository structure

```
CIE2-Activity2-LogisticRegression-<RollNo>/
├── README.md                              # this file
├── RCA_Activity_2.ipynb                   # executed notebook (all outputs visible)
├── requirements.txt                       # library versions used
└── figures/                               # saved plots (optional)
```

## 5. Work completed

| Step | What was done |
|------|---------------|
| 1. Problem framing | Defined positive class, who acts, cost of each error, primary metric |
| 2. Data inspection | Types, summary statistics, missing values, duplicates, label check; class distribution bar chart |
| 3. Data preparation | Impossible zeros converted to missing and **median-imputed**; features standardised; no categorical encoding needed (all numeric) |
| 4. Split | **Stratified** 80:20 split (seed 42); imputer and scaler fit on training data only, inside a pipeline |
| 5. Theory | Why Linear Regression is unsuitable for class labels; sigmoid and decision threshold; sigmoid plot |
| 6. Training | Logistic Regression (`C=1.0`, L2 penalty, `lbfgs` solver, `max_iter=1000`); intercept, coefficients and odds ratios interpreted |
| 7. Predictions | Test probabilities and labels; effect of thresholds 0.3, 0.5 and 0.7 on positive counts |
| 8. Evaluation | Confusion matrix, accuracy, precision, recall, specificity, F1, classification report, majority baseline, ROC-AUC, precision-recall curve, threshold table, train vs test comparison |
| 9. Threshold selection | Threshold chosen with 5-fold cross-validation on the **training** set, then reported once on the test set |
| 10. Decision behaviour | Predicted probability vs `Glucose` with threshold lines |
| 11. New example | One new record classified through the same pipeline |

**Environment:** Python 3.13, numpy 2.4.6, pandas 2.2.3, scikit-learn 1.6.1, scipy 1.16.3. Random seed: `42`.

## 6. Key results

**Split (stratified, 80:20, `random_state=42`):**

| Set | Records | Class 0 | Class 1 | Positive share |
|-----|---------|---------|---------|----------------|
| Train | 614 | 400 | 214 | 34.9% |
| Test | 154 | 100 | 54 | 35.1% |

Both sets keep the original class proportions.

### Model coefficients (standardised features)

Intercept: **−0.8722**

| Feature | Coefficient (per 1 SD) | Odds ratio (e^coef) |
|---------|------------------------|---------------------|
| Glucose | +1.1826 | 3.263 |
| BMI | +0.6887 | 1.991 |
| Pregnancies | +0.3774 | 1.459 |
| DiabetesPedigreeFunction | +0.2333 | 1.263 |
| Age | +0.1478 | 1.159 |
| Insulin | −0.0661 | 0.936 |
| BloodPressure | −0.0441 | 0.957 |
| SkinThickness | +0.0283 | 1.029 |

**Interpretation:** A positive coefficient raises the probability of diabetes, and a negative one lowers it, holding the other features constant. Because features are standardised, each coefficient is per 1 standard-deviation increase. **Glucose** is by far the strongest driver (one SD higher multiplies the odds by about 3.3), followed by **BMI** (about 2.0) and **Pregnancies** (about 1.5). `Insulin`, `BloodPressure` and `SkinThickness` have coefficients close to zero, so they add little once the others are included; the small negative sign on `Insulin` is not a reliable finding (the column is 49% imputed). These show association, not causation.

### Effect of the threshold on the number of positives (test set)

Actual positives in the test set: 54.

| Threshold | Predicted positive | Share of test set |
|-----------|--------------------|-------------------|
| 0.3 | 72 | 46.8% |
| 0.5 | 45 | 29.2% |
| 0.7 | 28 | 18.2% |

### Confusion matrix (test set, threshold 0.5)

|  | Predicted 0 | Predicted 1 |
|--|-------------|-------------|
| **Actual 0** | TN = 82 | FP = 18 |
| **Actual 1** | FN = 27 | TP = 27 |

- **TP (27):** diabetic people correctly flagged for referral.
- **TN (82):** healthy people correctly cleared.
- **FP (18):** healthy people flagged, so an unnecessary confirmatory test.
- **FN (27):** diabetic people cleared, so missed cases (the most serious error).

### Classification evaluation results (test set, threshold 0.5)

| Metric | Formula | Result |
|--------|---------|--------|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) | 0.708 |
| Precision | TP / (TP + FP) | 0.600 |
| Recall (sensitivity) | TP / (TP + FN) | 0.500 |
| Specificity | TN / (TN + FP) | 0.820 |
| F1-score | 2 × (P × R) / (P + R) | 0.545 |
| ROC-AUC | Area under ROC curve | 0.813 |

**Interpretation:** At the default 0.5 threshold, the model finds only half of the people who truly have diabetes (recall 0.50), so it misses 27 of 54. When it does flag someone, it is right 60% of the time (precision 0.60), and it correctly clears 82% of healthy people. The ranking ability is good (AUC 0.813), but the default threshold is too cautious for a screening task.

**Classification report (test set):**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0: No diabetes | 0.75 | 0.82 | 0.78 | 100 |
| 1: Diabetes | 0.60 | 0.50 | 0.55 | 54 |
| Accuracy | | | 0.71 | 154 |
| Macro avg | 0.68 | 0.66 | 0.67 | 154 |
| Weighted avg | 0.70 | 0.71 | 0.70 | 154 |

### Comparison with the majority-class baseline

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Always predict "No diabetes" | 0.649 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression (0.5) | 0.708 | 0.600 | 0.500 | 0.545 | 0.813 |

**Interpretation:** The baseline already reaches 64.9% accuracy by never predicting diabetes, yet it finds **no** diabetic person. This is why accuracy alone misleads on imbalanced data. The logistic model gains only about 6 accuracy points, but it adds real detection (recall 0.50) and good ranking (AUC 0.813 vs 0.5).

### Threshold comparison (test set)

| Threshold | Predicted positive | Accuracy | Precision | Recall | F1-score | FN (missed) | FP (false alarms) |
|-----------|--------------------|----------|-----------|--------|----------|-------------|-------------------|
| 0.3 | 72 | 0.740 | 0.597 | 0.796 | 0.683 | 11 | 29 |
| 0.5 | 45 | 0.708 | 0.600 | 0.500 | 0.545 | 27 | 18 |
| 0.7 | 28 | 0.727 | 0.714 | 0.370 | 0.488 | 34 | 8 |

**Trade-off:** Raising the threshold flags fewer people. Precision rises (0.597 → 0.714) because flagged people are more likely truly diabetic, but recall collapses (0.796 → 0.370), so missed cases go from 11 to 34. A lower threshold does the opposite: far fewer missed cases, but more false alarms (8 → 29). Accuracy barely changes across thresholds, which again shows it is a poor guide here.

### Recommended threshold

| Item | Value |
|------|-------|
| Goal | Recall of at least 80% |
| Selection method | 5-fold cross-validated probabilities on the **training** set (no test leakage) |
| **Recommended threshold** | **0.28** |
| Training-CV recall / precision at 0.28 | 0.804 / 0.581 |

**Test-set result at 0.28 (reported once):**

|  | Predicted 0 | Predicted 1 |
|--|-------------|-------------|
| **Actual 0** | TN = 67 | FP = 33 |
| **Actual 1** | FN = 9 | TP = 45 |

| Metric | At 0.5 | At 0.28 (recommended) |
|--------|--------|-----------------------|
| Accuracy | 0.708 | 0.727 |
| Precision | 0.600 | 0.577 |
| Recall | 0.500 | **0.833** |
| F1-score | 0.545 | 0.682 |
| Missed cases (FN) | 27 | **9** |
| False alarms (FP) | 18 | 33 |

**Justification:** A false negative (a diabetic person told they are fine) is more costly than a false positive (an extra confirmatory test). Lowering the threshold from 0.5 to 0.28 cuts missed cases from 27 to 9 (recall 0.50 → 0.83) and also raises F1 and accuracy. The price is 15 more false alarms (18 → 33) and slightly lower precision (0.600 → 0.577). The cross-validated recall target of 0.80 was met on training data and held up on the test set (0.833). The final threshold should be agreed with clinicians who know the real cost of each error.

### Training vs test (over/underfitting check, threshold 0.5)

| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 0.796 | 0.708 |
| Precision | 0.770 | 0.600 |
| Recall | 0.593 | 0.500 |
| F1-score | 0.670 | 0.545 |
| ROC-AUC | 0.850 | 0.813 |

**Interpretation:** Test scores are lower than training scores by a few points (AUC 0.850 → 0.813), which is expected and shows only mild overfitting; with just 154 test records, some of the gap is sampling noise. Both sets are modest on recall at 0.5, so the main limitation is the default threshold and the simple linear boundary, not memorisation.

### Decision behaviour (predicted probability vs Glucose)

With all other features held at their training medians, the predicted probability of diabetes rises smoothly in an S-shape as glucose increases. It crosses the default 0.5 line at a glucose of roughly **150** and crosses the recommended 0.28 line at roughly **125**. So the recommended threshold flags people from a lower glucose level, which is how it catches more true cases. Test records with diabetes (red) cluster at higher glucose and healthy records (blue) at lower glucose, but the groups overlap, so no threshold separates them perfectly.

### New prediction

| Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI | DPF | Age | P(diabetes) | Class at 0.5 | Class at 0.28 |
|-------------|---------|---------------|---------------|---------|-----|-----|-----|-------------|--------------|---------------|
| 2 | 150 | 80 | 30 | 120 | 33.0 | 0.5 | 45 | **0.561** | 1 | 1 |

All values are inside the training range. This is a probability-based estimate, not a guaranteed outcome.

## 7. Decision log (summary)

| # | Decision | Alternatives | Reason |
|---|----------|--------------|--------|
| 1 | Used Pima Indians Diabetes | Telco churn, breast cancer | Health screening where a missed case is clearly costly; moderate imbalance |
| 2 | Positive class = diabetes | Reverse coding | It is the case we most want to catch |
| 3 | Zeros → missing → median imputation | Drop rows, mean imputation, keep zeros | 374 Insulin and 227 SkinThickness zeros are too many to drop; median is robust to skew |
| 4 | Standardised features | No scaling | Helps the optimiser; comparable coefficients |
| 5 | Stratified 80:20 split, seed 42 | Random split, 70:30 | Keeps about 35% positives in both sets |
| 6 | Pipeline (imputer + scaler + model) | Manual steps | Guarantees training-only fitting |
| 7 | Primary metric = recall (with precision, F1, AUC) | Accuracy | A false negative is the costly error; accuracy misleads with imbalance |
| 8 | Threshold = 0.28 | 0.3, 0.5, 0.7 | Highest threshold with training-CV recall of at least 80%; test recall 0.833 |
| 9 | Model must not decide alone | None | Screening aid only; needs clinical confirmation |

## 8. Conclusion

The Pima Indians Diabetes data (768 records, 8 numeric features, 34.9% positive) was prepared by treating impossible zeros as missing, median-imputing and standardising inside a pipeline, and splitting 80:20 with stratification. Logistic Regression ranks people well (**test ROC-AUC 0.813**), with Glucose, BMI and Pregnancies as the strongest drivers. At the default 0.5 threshold, however, recall is only 0.50 (27 of 54 diabetic people missed), even though accuracy (0.708) beats the majority baseline (0.649, recall 0). Because a missed diagnosis costs more than an extra test, I recommend a threshold of **0.28**, chosen by cross-validation on the training set; on the test set it raises recall to **0.833** (missed cases fall from 27 to 9) at the cost of 33 false alarms and precision of 0.577. The model is a screening aid, not a diagnosis.

## 9. Limitations and responsible use

- **Screening aid only.** The model must not diagnose or exclude diabetes on its own; every positive and every borderline negative needs clinical confirmation.
- Data comes from one population and is small (768 records; 154 in the test set), so results may not transfer to other groups and the metrics carry noticeable uncertainty.
- Many `Insulin` (374) and `SkinThickness` (227) values were missing and imputed with the median, which adds uncertainty.
- A linear decision boundary may miss interactions between features.
- At the recommended threshold, about one in three flagged people does not have diabetes (precision 0.577), which means extra tests and anxiety for them; at 0.5, about half of the diabetic people are missed.
- A prediction is a **probability-based estimate, not a guaranteed outcome**.

## 10. How to run

1. Open `RCA_Activity_2.ipynb` in [Google Colab](https://colab.research.google.com) (or Jupyter).
2. Run **Runtime → Run all**. The dataset is downloaded automatically from the CSV link above.
3. For a local run: `pip install -r requirements.txt`, then open the notebook.

**Libraries:** pandas, numpy, matplotlib, seaborn, scipy, scikit-learn (exact versions in `requirements.txt`).

## 11. References

1. Smith, J.W. et al. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. *Proc. Symposium on Computer Applications and Medical Care*, 261–265.
2. Pima Indians Diabetes Database (Kaggle mirror of the NIDDK data): https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
3. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR* 12, 2825–2830.

---

*Submitted for CIE 2 evaluation. Faculty submission GitHub ID: [complexiota](https://github.com/complexiota). All work was completed independently; decisions and explanations are in my own words.*
