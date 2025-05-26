# Titanic Dataset - Data Cleaning & Preprocessing

This repository contains the code and documentation for the initial data cleaning and preprocessing steps performed on the Titanic dataset. This task is part of my machine learning internship, focusing on preparing raw data for model training.

## Project Objective

The primary objective of this phase is to learn and apply fundamental data cleaning and preprocessing techniques to the Titanic dataset. This involves:

1.  Importing and exploring basic dataset information.
2.  Handling missing values effectively.
3.  Converting categorical features into numerical representations.
4.  Normalizing/standardizing numerical features.
5.  Visualizing and identifying potential outliers.

## Dataset

The dataset used is `Titanic-Dataset.csv`, containing information about passengers on the Titanic, including whether they survived.

## Code Implementation (Google Colab)

The following Python code was executed in a Google Colab environment to perform the data cleaning and preprocessing steps.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Import the dataset and explore basic info
# Assuming the file 'Titanic-Dataset.csv' is uploaded to your Google Colab environment
df = pd.read_csv('Titanic-Dataset.csv')

print("--- Initial Data Info ---")
df.info()
print("\n--- Missing Values Before Handling ---")
print(df.isnull().sum())
print("\n--- First 5 Rows ---")
print(df.head())

# 2. Handle missing values
# 'Age': Impute missing ages with the median, as age can be skewed.
df['Age'].fillna(df['Age'].median(), inplace=True)

# 'Embarked': Impute missing embarked with the mode, as it's a categorical variable.
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# 'Cabin': This column has too many missing values and might not be directly useful without complex feature engineering.
# For simplicity in this task, we will drop the 'Cabin' column.
df.drop('Cabin', axis=1, inplace=True)

print("\n--- Missing Values After Handling ---")
print(df.isnull().sum())

# 3. Convert categorical features into numerical using encoding
# 'Sex': Use Label Encoding as it's a binary categorical variable.
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# 'Embarked': Use One-Hot Encoding as there are more than two categories ('S', 'C', 'Q').
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True) # drop_first avoids multicollinearity

print("\n--- Data Types After Encoding ---")
print(df.info())
print("\n--- First 5 Rows After Encoding ---")
print(df.head())

# 4. Normalize/Standardize the numerical features
# Features to scale: 'Age', 'Fare', 'SibSp', 'Parch'
# 'PassengerId' and 'Survived' are identifiers/target and 'Pclass' is already ordinal.
# Ticket and Name are not numerical features.
numerical_features = ['Age', 'Fare', 'SibSp', 'Parch']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

print("\n--- First 5 Rows After Scaling Numerical Features ---")
print(df.head())

# 5. Visualize outliers using boxplots and optionally remove them
# We'll visualize for 'Age' and 'Fare' as they are continuous and more prone to outliers.
# Outlier removal is highly context-dependent and should be done carefully.
# For demonstration, we'll show how to detect and a basic way to cap/remove (commented out actual removal).

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.boxplot(y=df['Age'])
plt.title('Boxplot of Age')

plt.subplot(1, 2, 2)
sns.boxplot(y=df['Fare'])
plt.title('Boxplot of Fare')

plt.tight_layout()
plt.show()

# Example of outlier detection using IQR method (for Fare)
Q1_fare = df['Fare'].quantile(0.25)
Q3_fare = df['Fare'].quantile(0.75)
IQR_fare = Q3_fare - Q1_fare
lower_bound_fare = Q1_fare - 1.5 * IQR_fare
upper_bound_fare = Q3_fare + 1.5 * IQR_fare

print(f"\n--- Outlier Detection for Fare ---")
print(f"Lower Bound (Fare): {lower_bound_fare}")
print(f"Upper Bound (Fare): {upper_bound_fare}")
print(f"Number of outliers in Fare: {df[(df['Fare'] < lower_bound_fare) | (df['Fare'] > upper_bound_fare)].shape[0]}")

# # To remove outliers (use with caution, uncomment if you decide to remove)
df_cleaned = df[(df['Fare'] >= lower_bound_fare) & (df['Fare'] <= upper_bound_fare)]
print(f"Shape after removing Fare outliers: {df_cleaned.shape}")

# Also consider dropping 'Name' and 'Ticket' as they are not numerical features for direct modeling.
df.drop(['Name', 'Ticket'], axis=1, inplace=True)

print("\n--- Final Cleaned Data Info ---")
df.info()
print("\n--- First 5 Rows of Final Cleaned Data ---")
print(df.head())
