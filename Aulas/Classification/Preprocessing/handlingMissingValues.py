import pandas as pd # Load csv files
import numpy as np
import seaborn as sns # View graphs
import matplotlib.pyplot as plt # View graphs
import plotly.express as px # Create a dynamic graph

base_credit = pd.read_csv("credit_data.csv")

# Replace the inconsistent values with the mean of the column
mean = base_credit['age'][base_credit['age'] > 0].mean() # Removing the inconsistent values and get the mean
base_credit.loc[base_credit['age'] < 0, 'age'] = mean # Changing the inconsistent ages

print(base_credit.isnull()) # It prints for every row and for every column if this cell is null

print('\n', base_credit.isnull().sum()) # How many times 'True' appears for each column

print('\n', base_credit.loc[pd.isnull(base_credit['age'])], '\n') # Show the lines that have null 'age'

base_credit['age'].fillna(base_credit['age'].mean(), inplace=True) # Change the 'base_credit': True and the columns that had 'age' == null to the mean
print('\n', base_credit.head(31), '\n')

print(base_credit.loc[(base_credit['clientid'] == 29) | (base_credit['clientid'] == 31) | (base_credit['clientid'] == 32)], '\n')
print(base_credit.loc[base_credit['clientid'].isin([29, 31, 32])])


