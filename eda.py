import pandas as pd

df = pd.read_csv('data/students.csv')

print("Shape:", df.shape)
print("\nColumn types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nDropout risk balance:\n", df['dropout_risk'].value_counts())
print("\nExam score summary:\n", df['exam_score'].describe())