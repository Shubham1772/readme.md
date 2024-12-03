# readme.md
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset (replace 'data.csv' with your actual data file)
data = pd.read_csv('data.csv')

# Display the first few rows of the dataset
print(data.head())

# Data Cleaning: Check for missing values
print(data.isnull().sum())

# Fill missing values or drop them as necessary
data.fillna(method='ffill', inplace=True)

# Exploratory Data Analysis: Visualizing the impact of discounts on sales
plt.figure(figsize=(10, 6))
sns.barplot(x='Discount', y='Units_Sold', data=data)
plt.title('Impact of Discounts on Units Sold')
plt.xlabel('Discount (%)')
plt.ylabel('Units Sold')
plt.show()

# Preparing data for modeling
X = data[['Discount']]
y = data['Units_Sold']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
predictions = model.predict(X_test)

# Evaluating the model's performance
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
