import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data/students.csv')
y = (df['dropout_risk'] == 'Yes').astype(int)
X = df.drop(columns=['student_id', 'dropout_risk'])
X = pd.get_dummies(X, drop_first=True)

rf = RandomForestClassifier(n_estimators=200, class_weight='balanced_subsample',
                            min_samples_leaf=5, n_jobs=-1, random_state=42)
rf.fit(X, y)
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top 10 feature importances:\n")
print(imp.head(10).round(4))

tree = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
tree.fit(X, y)
print("\nDecision tree rules:\n")
print(export_text(tree, feature_names=list(X.columns)))

# 3) Min/max of the key columns for each class
key = ['stress_level', 'motivation_level', 'exam_anxiety_score', 'previous_gpa', 'mental_health_rating']
print("\nRanges by dropout risk:\n")
print(df.groupby('dropout_risk')[key].agg(['min', 'max']).T)