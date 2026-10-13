import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data/students.csv')
y = (df['dropout_risk'] == 'Yes').astype(int)

X = df.drop(columns=['student_id', 'dropout_risk',
                      'stress_level', 'motivation_level', 'exam_anxiety_score'])
X = pd.get_dummies(X, drop_first=True)

rf = RandomForestClassifier(n_estimators=200, class_weight='balanced_subsample',
                             min_samples_leaf=5, n_jobs=-1, random_state=42)
rf.fit(X, y)

imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
imp.sort_values().plot(kind='barh', color='#2b6cb0')
plt.title('Top 10 features predicting dropout risk\n(honest model — no leaky columns)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Saved feature_importance.png")
print("\nTop 10 features:\n", imp.round(4))