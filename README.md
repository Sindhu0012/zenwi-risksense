cat > README.md << 'EOF'
# Zenwi RiskSense

**An early-warning ML prototype for identifying at-risk students — built as an R&D feature concept for [Zenwi](https://github.com/Sindhu0012/zenwi), my study-tracker app.**

> **Honest framing:** This uses a public Kaggle dataset, not real Zenwi user data (Zenwi currently has too few users for any meaningful ML). This project prototypes the ML approach Zenwi could eventually use if it scales — the engineering here is real, the data source is public and clearly labeled as such.

## The problem

Can we predict which students are at risk of dropping out *before* it happens, using everyday habits — study time, sleep, screen time, attendance — rather than waiting until grades already show it?

## Dataset

[Enhanced Student Habits & Performance Dataset](https://www.kaggle.com) — 80,000 students, 30 features including study hours, sleep, social media/Netflix use, attendance, stress, motivation, and a `dropout_risk` (Yes/No) label.

## What I found during EDA

![Dropout risk balance](balance.png)

The dataset is heavily imbalanced — the vast majority of students are labeled "not at risk," with only a small fraction flagged "Yes." This matters a lot: a model can score 95%+ accuracy just by predicting "No" every single time, while being completely useless. This is why I used `class_weight='balanced'` during training and evaluated with precision/recall/ROC-AUC instead of relying on plain accuracy.

![Correlation heatmap](heatmap.png)

## A data-leak I caught and removed

Three columns — `stress_level`, `motivation_level`, and `exam_anxiety_score` — turned out to correlate suspiciously strongly with the `dropout_risk` label. I tested this directly with a simple rule (`stress > 8 AND motivation ≤ 3 AND anxiety = 10`), and it matched the real labels almost perfectly on its own — a strong sign these columns were used to *construct* the synthetic label in the first place, rather than being independent predictors.

**I trained and report two versions**: one including these columns (inflated, unrealistic performance) and one honestly excluding them (the real, trustworthy result). All final conclusions below are from the **honest, leak-free model**.

## Models trained

- Logistic Regression (`class_weight='balanced'`)
- Random Forest (`class_weight='balanced_subsample'`, 200 trees)

## Results (honest model — leaky columns removed)

<!-- Fill these in from your train.py output, the "(WITHOUT stress, motivation, anxiety)" section -->
| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Precision (Yes) | _fill in_ | _fill in_ |
| Recall (Yes) | _fill in_ | _fill in_ |
| ROC-AUC | _fill in_ | _fill in_ |
| PR-AUC | _fill in_ | _fill in_ |

## What actually predicts risk

![Feature importance](feature_importance.png)

Top drivers of dropout risk (honest model, no leaked columns):

1. **Previous GPA** (0.097) — past academic performance is the single strongest signal
2. **Mental health rating** (0.087)
3. **Exam score** (0.077)
4. **Attendance percentage** (0.063)
5. **Study hours per day** (0.060)
6. Time management score, screen time, sleep hours, social media hours, Netflix hours — all moderately contribute

## Why this matters for Zenwi

If Zenwi scaled to real users, a feature like this could flag at-risk students early — using exactly the kind of data Zenwi already tracks (study habits, time management) — well before a drop in grades becomes visible.

## Tech stack

Python, pandas, scikit-learn, matplotlib, seaborn

## Project structure
