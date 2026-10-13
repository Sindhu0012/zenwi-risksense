# Zenwi RiskSense

**An ML prototype for identifying at-risk students, built as an R&D feature concept for [Zenwi](https://github.com/Sindhu0012/zenwi), my study-tracker app.**

> **Honest framing:** This uses a public Kaggle dataset, not real Zenwi user data (Zenwi has too few users for meaningful ML). The engineering is real; the data source is public and clearly labeled as such.

## The problem

Can we predict which students are at risk of dropping out from everyday habits (study time, sleep, screen time, attendance) rather than waiting until grades show it?

## Dataset

Enhanced Student Habits & Performance Dataset (Kaggle): 80,000 students, 31 columns (29 features plus `student_id` and the `dropout_risk` Yes/No label). No missing values.

## What I found during EDA

![Dropout risk balance](balance.png)

The dataset is heavily imbalanced: only 1,582 of 80,000 students (about 2%) are labeled "Yes". A model can score 98% accuracy by predicting "No" every time while being useless, so I used `class_weight='balanced'` and evaluated with precision, recall, ROC-AUC and PR-AUC instead of accuracy.

![Correlation heatmap](heatmap.png)

## A data leak I caught

With all features, Random Forest scored a perfect 1.0 on every metric. That's a red flag, so I investigated. A depth-3 decision tree exposed this rule:

> `dropout_risk = Yes` exactly when `stress_level > 8` AND `motivation_level <= 3` AND `exam_anxiety_score == 10`

I verified it directly: the rule predicts all 1,582 dropouts with **0 mismatches**. The label is a hard-coded synthetic formula, so the models were rediscovering the dataset's generator, not learning anything about students.

I then removed those three columns and retrained. All conclusions below come from this leak-free setup.

## Models trained

- Logistic Regression (`class_weight='balanced'`)
- Random Forest (`class_weight='balanced_subsample'`, 200 trees)

## Results

| Metric (Yes class) | Logistic Regression | Random Forest |
|---|---|---|
| Precision | 0.03 | 0.00 |
| Recall | 0.56 | 0.00 |
| ROC-AUC | 0.688 | 0.676 |
| PR-AUC | 0.049 | 0.056 |

For reference: a no-skill model has PR-AUC of about 0.02 (the dropout rate) and ROC-AUC of 0.5. Both models are only slightly better than random. Random Forest reached 98% accuracy while catching 0 of 316 dropouts, which is why accuracy is the wrong metric here.

For comparison, with the leaky columns included: Logistic Regression had recall 1.00 and PR-AUC 0.816, and Random Forest scored 1.0 on everything.

## Feature importance (leak-free model)

![Feature importance](feature_importance.png)

| Rank | Feature | Importance |
|---|---|---|
| 1 | previous_gpa | 0.097 |
| 2 | mental_health_rating | 0.087 |
| 3 | exam_score | 0.077 |
| 4 | attendance_percentage | 0.063 |
| 5 | study_hours_per_day | 0.060 |

The importances are spread almost evenly (0.05 to 0.10) and the model barely beats random, so I don't read much into this ranking. It mostly shows that once the generating columns are removed, no remaining feature carries a strong signal.

## Takeaways

- Perfect scores usually mean leakage or synthetic labels: investigate before celebrating.
- Use recall, precision and PR-AUC on imbalanced data, not accuracy.
- Shallow decision trees and feature importance are quick ways to expose hidden rules.
- This dataset is not suitable for a real dropout predictor. A real version needs real, noisy student data.

## Why this matters for Zenwi

Zenwi already tracks study habits and time management, so a risk feature is a plausible future direction. This project shows the pipeline and, more importantly, the validation discipline such a feature would need before anyone trusts it.

## Tech stack

Python, pandas, scikit-learn, matplotlib, seaborn

## Project structure

```
zenwi-risksense/
├── 1_eda.py                  # Shape, types, missing values, class balance
├── 2_leak_check.py           # Finds and verifies the hidden label rule
├── 3_train_model.py          # Trains Logistic Regression and Random Forest
├── 4_feature_importance.py   # Feature importance on the leak-free data
├── balance.png               # Class balance chart
├── heatmap.png               # Correlation heatmap
├── feature_importance.png    # Feature importance chart
├── data/students.csv         # Kaggle dataset (not included, download it yourself)
└── README.md
```
**[Zenwi RiskSense](https://github.com/Sindhu0012/zenwi-risksense)** - ML experiment on a public student dataset. Models hit ~100% until I traced the label to a hard-coded synthetic rule, then retrained honestly (PR-AUC ~0.05). Python, scikit-learn, pandas.
