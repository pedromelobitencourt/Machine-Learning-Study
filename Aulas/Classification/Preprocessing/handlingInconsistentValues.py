import pandas as pd # Load csv files
import numpy as np
import seaborn as sns # View graphs
import matplotlib.pyplot as plt # View graphs
import plotly.express as px # Create a dynamic graph

base_credit = pd.read_csv("credit_data.csv")

print(base_credit.loc[base_credit['age'] < 0]) # It prints all lines that have age < 0
print()
print(base_credit[base_credit['age'] < 0])

print()

## There are some ways to handle it:
    # Delete the column that has the problem
base_credit2 = base_credit.drop('age', axis=1)
print(base_credit2.head(11), '\n')
    # Delete only the rows that has inconsistent values
base_credit3 = base_credit.drop(base_credit[base_credit['age'] < 0].index)
print(base_credit3.shape[0]) # How many rows
print(base_credit3.shape[1], '\n') # How many columns
    # Replace the inconsistent values with the mean of the column
# base_credit.mean() # Get the mean values of every column
mean = base_credit['age'][base_credit['age'] > 0].mean() # Removing the inconsistent values and get the mean
base_credit.loc[base_credit['age'] < 0, 'age'] = mean # Changing the inconsistent ages
print(base_credit[base_credit['age'] < 0])
print('\n', base_credit.head(28))
