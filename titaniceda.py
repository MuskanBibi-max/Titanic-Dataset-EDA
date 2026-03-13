# ==========================================================
# Advanced Titanic Dataset EDA
# Internship Data Analytics Project
# ==========================================================

# -----------------------------
# Install libraries if missing
# -----------------------------
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import pandas as pd
except ImportError:
    install("pandas")
    import pandas as pd

try:
    import seaborn as sns
except ImportError:
    install("seaborn")
    import seaborn as sns

try:
    import matplotlib.pyplot as plt
except ImportError:
    install("matplotlib")
    import matplotlib.pyplot as plt

import numpy as np
import os

# -----------------------------
# Create output folder
# -----------------------------
output_dir = "outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# -----------------------------
# Load Titanic Dataset
# -----------------------------
print("Loading Titanic dataset...")
df = sns.load_dataset("titanic")
print(df.head())
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())

# -----------------------------
# Feature Engineering
# -----------------------------
df['family_size'] = df['sibsp'] + df['parch'] + 1
df['age_group'] = pd.cut(df['age'], bins=[0,12,18,35,60,100],
                         labels=["Child","Teen","Young Adult","Adult","Senior"])

# -----------------------------
# 1. Survival Count
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="survived", data=df)
plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Number of Passengers")
plt.savefig(f"{output_dir}/survival_count.png")
plt.show()

# -----------------------------
# 2. Survival by Sex
# -----------------------------
plt.figure(figsize=(6,4))
sns.barplot(x="sex", y="survived", data=df)
plt.title("Survival Rate by Gender")
plt.savefig(f"{output_dir}/survival_by_sex.png")
plt.show()

# -----------------------------
# 3. Survival by Passenger Class
# -----------------------------
plt.figure(figsize=(6,4))
sns.barplot(x="pclass", y="survived", data=df)
plt.title("Survival Rate by Passenger Class")
plt.savefig(f"{output_dir}/survival_by_class.png")
plt.show()

# -----------------------------
# 4. Age Distribution (Histogram + KDE)
# -----------------------------
plt.figure(figsize=(6,4))
sns.histplot(df["age"], kde=True, bins=30, color="skyblue")
plt.title("Age Distribution")
plt.savefig(f"{output_dir}/age_distribution.png")
plt.show()

# -----------------------------
# 5. Age vs Survival (Violin Plot)
# -----------------------------
plt.figure(figsize=(6,4))
sns.violinplot(x="survived", y="age", data=df, inner="quartile", palette="Pastel1")
plt.title("Age Distribution by Survival")
plt.savefig(f"{output_dir}/age_violin.png")
plt.show()

# -----------------------------
# 6. Fare vs Survival (Boxplot)
# -----------------------------
plt.figure(figsize=(6,4))
sns.boxplot(x="survived", y="fare", data=df)
plt.title("Fare Distribution by Survival")
plt.savefig(f"{output_dir}/fare_boxplot.png")
plt.show()

# -----------------------------
# 7. Family Size vs Survival
# -----------------------------
plt.figure(figsize=(8,4))
sns.barplot(x="family_size", y="survived", data=df)
plt.title("Survival Rate by Family Size")
plt.savefig(f"{output_dir}/family_size.png")
plt.show()

# -----------------------------
# 8. Embarked Port vs Survival
# -----------------------------
plt.figure(figsize=(6,4))
sns.barplot(x="embarked", y="survived", data=df)
plt.title("Survival Rate by Embarkation Port")
plt.savefig(f"{output_dir}/embarked_survival.png")
plt.show()

# -----------------------------
# 9. Correlation Heatmap
# -----------------------------
numeric_features = df.select_dtypes(include=['float64','int64'])
corr = numeric_features.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(8,6))
sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig(f"{output_dir}/correlation_heatmap.png")
plt.show()

# -----------------------------
# 10. Pairplot for Key Numeric Features
# -----------------------------
sns.pairplot(df[['age','fare','sibsp','parch','survived']], hue='survived')
plt.savefig(f"{output_dir}/pairplot.png")
plt.show()

# -----------------------------
# 11. Age Group vs Survival
# -----------------------------
plt.figure(figsize=(8,4))
sns.barplot(x="age_group", y="survived", data=df)
plt.title("Survival Rate by Age Group")
plt.savefig(f"{output_dir}/age_group_survival.png")
plt.show()

# -----------------------------
# Key Insights
# -----------------------------
print("\nKey Insights from Advanced Titanic EDA:")
print("• Female passengers had higher survival rates than male passengers.")
print("• First-class passengers survived at higher rates than lower classes.")
print("• Children had better survival probability than adults.")
print("• Passengers with small families had higher survival rates than those alone or in large families.")
print("• Higher fare passengers generally survived more.")
print("• Embarkation port correlates with survival due to passenger class distribution.")
print("\nAll charts saved in the 'outputs' folder.")
