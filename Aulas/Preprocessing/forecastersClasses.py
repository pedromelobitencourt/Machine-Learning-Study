import pandas as pd # Load csv files
import numpy as np
import seaborn as sns # View graphs
import matplotlib.pyplot as plt # View graphs
import plotly.express as px # Create a dynamic graph

base_credit = pd.read_csv("credit_data.csv")

# Replace the inconsistent values with the mean of the column
mean = base_credit['age'][base_credit['age'] > 0].mean() # Removing the inconsistent values and get the mean
base_credit.loc[base_credit['age'] < 0, 'age'] = mean # Changing the inconsistent ages

base_credit['age'].fillna(base_credit['age'].mean(), inplace=True) # Change the 'base_credit': True and the columns that had 'age' == null to the mean

# Predictors or ForeCasters
X_credit = base_credit.iloc[:, 1:4].values # Getting all rows and 'income', 'age', 'loan' columns and put each row in an array

print(base_credit.head(8), '\n')
print(X_credit)

# Classes
Y_credit = base_credit.iloc[:, 4].values # Getting the fourth column (It includes 4, cuz it's'not a range)