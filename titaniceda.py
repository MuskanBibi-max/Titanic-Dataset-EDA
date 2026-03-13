# ==========================================================
# Titanic Dataset Exploratory Data Analysis (EDA)
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


# -----------------------------
# Load Titanic Dataset
# -----------------------------
print("\nLoading Titanic dataset...")

df = sns.load_dataset("titanic")

print("\nFirst 5 rows:")
print(df.head())


# -----------------------------
# Dataset Information
# -----------------------------
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# -----------------------------
# Survival Rate by Gender
# -----------------------------
plt.figure(figsize=(6,4))

sns.barplot(x="sex", y="survived", data=df)

plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")

plt.savefig("survival_by_gender.png")
plt.show()


# -----------------------------
# Survival Rate by Passenger Class
# -----------------------------
plt.figure(figsize=(6,4))

sns.barplot(x="pclass", y="survived", data=df)

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")

plt.savefig("survival_by_class.png")
plt.show()


# -----------------------------
# Age Distribution (Boxplot)
# -----------------------------
plt.figure(figsize=(6,4))

sns.boxplot(x="survived", y="age", data=df)

plt.title("Age Distribution of Survivors")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Age")

plt.savefig("age_boxplot.png")
plt.show()


# -----------------------------
# Create Age Groups
# -----------------------------
df["age_group"] = pd.cut(
    df["age"],
    bins=[0,12,18,35,60,100],
    labels=["Child","Teen","Young Adult","Adult","Senior"]
)

plt.figure(figsize=(8,4))

sns.barplot(x="age_group", y="survived", data=df)

plt.title("Survival Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Survival Rate")

plt.xticks(rotation=30)

plt.savefig("survival_by_age_group.png")
plt.show()


# -----------------------------
# Survival Count Visualization
# -----------------------------
plt.figure(figsize=(6,4))

sns.countplot(x="survived", data=df)

plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Number of Passengers")

plt.savefig("survival_count.png")
plt.show()


# -----------------------------
# Insight Report
# -----------------------------
print("\nKey Insights from Titanic Dataset Analysis:\n")

print("• Female passengers had significantly higher survival rates compared to male passengers.")
print("• First-class passengers had the highest survival probability.")
print("• Children had better survival chances compared to adults.")
print("• Passenger class strongly influenced survival chances.")
print("• Older passengers generally had lower survival probability.")

print("\nEDA completed successfully. Charts exported as PNG files.")