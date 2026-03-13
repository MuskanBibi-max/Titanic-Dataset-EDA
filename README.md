# 📊 Advanced Titanic Dataset EDA

## Project Overview
This project performs **Exploratory Data Analysis (EDA)** on the Titanic dataset to uncover patterns and insights related to passenger survival. The analysis explores survival rates based on **gender, passenger class, age, fare, family size, and embarkation port**, using multiple visualizations.

This project demonstrates skills in **data cleaning, feature engineering, visualization, and insight generation**, which are key for data analytics internships.

---

## 🛠 Technologies Used
- **Python** – Programming language  
- **Pandas** – Data manipulation and analysis  
- **NumPy** – Numerical operations  
- **Seaborn** – Statistical data visualization  
- **Matplotlib** – Plotting and chart export  

---

## 📁 Dataset
We use the **Titanic dataset** from Seaborn:

- **Survived** – 0 = No, 1 = Yes  
- **Pclass** – Passenger class (1, 2, 3)  
- **Sex** – Male / Female  
- **Age** – Passenger age  
- **Fare** – Ticket fare  
- **SibSp** – Number of siblings/spouses aboard  
- **Parch** – Number of parents/children aboard  
- **Embarked** – Port of embarkation (C, Q, S)  

Additional features were created:

- **family_size** = `sibsp + parch + 1`  
- **age_group** = Bucketed age ranges (Child, Teen, Young Adult, Adult, Senior)  

---

## 📊 Visualizations
The analysis produces **10+ professional charts**:

| Chart | Description | File |
|-------|-------------|------|
| Survival Count | Overall survival numbers | survival_count.png |
| Survival by Sex | Survival rate for male vs female | survival_by_sex.png |
| Survival by Class | Survival rate by passenger class | survival_by_class.png |
| Age Distribution | Age histogram with KDE | age_distribution.png |
| Age vs Survival | Violin plot of age for survivors vs non-survivors | age_violin.png |
| Fare vs Survival | Boxplot of fare vs survival | fare_boxplot.png |
| Family Size vs Survival | Survival by family size | family_size.png |
| Embarked Port vs Survival | Survival by port of embarkation | embarked_survival.png |
| Correlation Heatmap | Heatmap of numeric feature correlations | correlation_heatmap.png |
| Pairplot | Pairwise scatterplots of numeric features | pairplot.png |
| Age Group vs Survival | Survival rate by age group | age_group_survival.png |

All charts are saved in the **`outputs` folder**.

---

## 🔍 Key Insights
- Female passengers had significantly higher survival rates than males.  
- First-class passengers survived at higher rates compared to second and third classes.  
- Children had better survival probability than adults.  
- Passengers with small families (2–4 people) survived more than those alone or with very large families.  
- Passengers who paid higher fares (usually first class) had higher survival.  
- Embarkation port shows some correlation with survival due to passenger class distribution.  

---

## 🚀 How to Run
1. Clone or download the repository.
2. Install required Python libraries:

```bash
pip install pandas numpy seaborn matplotlib

Run the Python script:

python titanic_eda_advanced.py

All visualizations will be saved in the outputs folder. Insights will be printed in the terminal.
