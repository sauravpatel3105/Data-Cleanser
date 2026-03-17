# 1. Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from scipy import stats
from scipy.stats.mstats import winsorize

# Create output folder
os.makedirs("outputs", exist_ok=True)

# 2. Load Dataset
df = pd.read_csv("data/patient_health_data.csv")

print("Dataset Loaded Successfully")
print("Dataset Shape:", df.shape)

df.head()

# 3. Basic Dataset Information
print("\nDataset Info")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

# 4. Missing Value Analysis
missing_values = df.isnull().sum()

missing_percent = (df.isnull().sum()/len(df))*100

missing_report = pd.DataFrame({
    "Missing Values": missing_values,
    "Percentage": missing_percent
})

print("\nMissing Value Summary")
print(missing_report)

# 5. Visualize Missing Data
plt.figure(figsize=(8,5))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.savefig("outputs/missing_values_heatmap.png")
plt.show()

# 6. Separate Column Types
numerical_cols = ["age","bmi","blood_pressure","cholesterol","glucose"]

categorical_cols = ["gender","region"]

# 7. Simple Imputer (Numerical)
num_imputer = SimpleImputer(strategy="mean")

df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

print("\nNumerical Missing Values Filled Using Mean")

# 8. Simple Imputer (Categorical)
cat_imputer = SimpleImputer(strategy="most_frequent")

df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

print("Categorical Missing Values Filled Using Most Frequent")

# 9. Missing Indicator + Random Sampling
df["bmi_missing"] = df["bmi"].isnull().astype(int)

print("Missing Indicator Column Created")

# 10. KNN Imputer
knn = KNNImputer(n_neighbors=5)

df[numerical_cols] = knn.fit_transform(df[numerical_cols])

print("KNN Imputation Applied")

# 11. MICE (Iterative Imputer)
mice = IterativeImputer(max_iter=10, random_state=42)

df[numerical_cols] = mice.fit_transform(df[numerical_cols])

print("MICE Imputation Applied")

# 12. Outlier Detection (Z-Score)
z_scores = np.abs(stats.zscore(df[numerical_cols]))

df_z = df[(z_scores < 3).all(axis=1)]

print("\nDataset Shape After Z-Score Outlier Removal:", df_z.shape)

# 13. Outlier Detection (IQR Method)
Q1 = df["bmi"].quantile(0.25)
Q3 = df["bmi"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_iqr = df[(df["bmi"] >= lower_bound) & (df["bmi"] <= upper_bound)]

print("Dataset Shape After IQR:", df_iqr.shape)

# 14. Percentile Method
lower = df["glucose"].quantile(0.01)
upper = df["glucose"].quantile(0.99)

df_percentile = df[(df["glucose"] >= lower) & (df["glucose"] <= upper)]

print("Dataset Shape After Percentile Method:", df_percentile.shape)

# 15. Winsorization Technique
df["cholesterol"] = winsorize(df["cholesterol"], limits=[0.05,0.05])

print("Winsorization Applied to Cholesterol")

# 16. Visualization Before vs After
plt.figure(figsize=(6,4))
sns.boxplot(x=df["bmi"])
plt.title("BMI Boxplot After Cleaning")
plt.savefig("outputs/bmi_boxplot.png")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df["glucose"], bins=30)
plt.title("Glucose Distribution")
plt.savefig("outputs/glucose_distribution.png")
plt.show()

# 17. Compare Before vs After
print("\nFinal Dataset Shape:", df.shape)

print("\nFinal Dataset Summary")
print(df.describe())

# 18. Save Final Clean Dataset
df.to_csv("outputs/final_clean_dataset.csv", index=False)

print("\nFinal Clean Dataset Saved")

# 19. Correlation Analysis
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("outputs/correlation_heatmap.png")
plt.show()

print("\nProject Completed Successfully")