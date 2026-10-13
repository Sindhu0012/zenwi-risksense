import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/students.csv')

df['dropout_num'] = (df['dropout_risk'] == 'Yes').astype(int)

num_cols = df.select_dtypes(include='number').columns.drop(['student_id', 'dropout_num'])
print("Average of each feature, by dropout risk:\n")
print(df.groupby('dropout_risk')[num_cols].mean().T.round(2))

corr = df[list(num_cols) + ['dropout_num']].corr()['dropout_num'].drop('dropout_num')
print("\nCorrelation with dropout (sorted):\n")
print(corr.sort_values())

cat_cols = df.select_dtypes(include='object').columns.drop('dropout_risk')
for c in cat_cols:
    print(f"\nDropout rate by {c}:")
    print((df.groupby(c)['dropout_num'].mean() * 100).round(2))

plt.figure(figsize=(14, 12))
sns.heatmap(df[list(num_cols) + ['dropout_num']].corr(), cmap='coolwarm', center=0)
plt.title('Correlation heatmap')
plt.tight_layout()
plt.savefig('heatmap.png')

plt.figure(figsize=(6, 4))
df['dropout_risk'].value_counts().plot(kind='bar')
plt.title('Dropout risk balance')
plt.tight_layout()
plt.savefig('balance.png')

print("\nSaved heatmap.png and balance.png")