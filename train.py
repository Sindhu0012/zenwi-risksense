import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, roc_auc_score

df = pd.read_csv('data/students.csv')

y = (df['dropout_risk'] == 'Yes').astype(int)
X = df.drop(columns=['student_id', 'dropout_risk'])

num_cols = X.select_dtypes(include='number').columns
cat_cols = X.select_dtypes(exclude='number').columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Train dropout rate:", round(y_train.mean() * 100, 2), "%")
print("Test dropout rate:", round(y_test.mean() * 100, 2), "%")


def make_pipeline(model):
    pre = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ])
    return Pipeline([('pre', pre), ('model', model)])


models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, class_weight='balanced_subsample',
        min_samples_leaf=5, n_jobs=-1, random_state=42
    ),
}

for name, model in models.items():
    pipe = make_pipeline(model)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=['No', 'Yes']))
    print("ROC-AUC:", round(roc_auc_score(y_test, proba), 4))
    print("PR-AUC :", round(average_precision_score(y_test, proba), 4))