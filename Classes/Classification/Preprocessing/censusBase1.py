import pandas as pd # Load csv files
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns # View graphs
import matplotlib.pyplot as plt # View graphs
import plotly.express as px # Create a dynamic graph

base_census = pd.read_csv('census.csv')
# Age: Discrete numeric variable
# Workclass: ordinal categorical variable (There is no sort)
# Final_weight: Contiguos numeric variable (There are variable that are so different from others)
# Education: Ordinal categorical variable
# Education-num: Discrete numeric variable
# Marital-status, Occupation, Relationship, Race, Sex, Native-country: Nominal categoric variable
# Income: Ordinal categoric variable

# ======================================================================================================================================

# print(base_census.describe())
# print(base_census.isnull().sum())

# ======================================================================================================================================

# print(np.unique(base_census['age'], return_counts=True)) # Divide the income variable in groups and return the number of each group
# sns.countplot(x = base_census['age']) # Create a graph with the number of each group; unbalanced data, there are more people with income <= 50k
# plt.show()

# ======================================================================================================================================

# plt.hist(x = base_census['age']) # Create a graph with the frequency of each age
# plt.show()

# ======================================================================================================================================

# q: What is the difference between hist and countplot?
# a: hist is used to show the frequency of a variable (between an interval), while countplot is used to show the frequency of each group of a variable

# graph = px.treemap(base_census, path=['workclass', 'age']) # It shows the frequency of each group of 'workclass' and 'age'
# graph.show()

# ======================================================================================================================================

# graph = px.parallel_categories(base_census, dimensions=['occupation', 'relationship'])
# graph.show()

# ======================================================================================================================================

X_census = base_census.iloc[:, 0:14].values # Getting all rows and all columns except the last one
# the first parameter of iloc is the rows that will be selected
# the second parameter is the columns that will be selected

Y_census = base_census.iloc[:, 14].values # Getting all rows and the last column