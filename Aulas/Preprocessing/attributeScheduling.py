import pandas as pd # Load csv files
from sklearn.preprocessing import StandardScaler
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

# Classes
Y_credit = base_credit.iloc[:, 4].values # Getting the fourth column (It includes 4, cuz it's'not a range)


print(X_credit[:, 0]) # Prints the income of every row
print(X_credit[:, 0].max())

# If the difference between maximum or minimum values of distinct attributes are so high, there could be a problem with some machine learning algorithms
# If so, we should apply an equation to standardize the values
# Also, for instance, if the values of an attribute are much greater than the values of another attribute, the algorithm could prioritize an attribute

# Standardization: When there are some outliers results (Values that are ) -> x = (x - mean()) / 
scaler_credit = StandardScaler()
X_credit = scaler_credit.fit_transform(X_credit)
print(X_credit[:, 0].max(), X_credit[:, 1].max(), X_credit[:, 2].max()) # Now the values are not so different like before (They are on the same scale)
# Now the algorithm will not consider an algorithm more important than another one