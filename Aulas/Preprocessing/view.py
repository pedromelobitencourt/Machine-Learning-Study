import pandas as pd # Load csv files
import numpy as np
import seaborn as sns # View graphs
import matplotlib.pyplot as plt # View graphs
import plotly.express as px # Create a dynamic graph

base_credit = pd.read_csv("credit_data.csv")

print(np.unique(base_credit['default'])) # Show the unique values of the column
print(np.unique(base_credit['default'], return_counts=True)) # Show the number of each value
# (array([0, 1]), array([1717,  283])) # Unbalanced data


# sns.countplot(x = base_credit['default']) # Graph
# plt.show() # Show the graph

# plt.hist(x=base_credit['age']) # Dividing into groups defined by 'age' (A = 18 to 28, B = 29 to 37....) and showing the frequency of each age
# plt.show()

# graph = px.scatter_matrix(base_credit, dimensions=['age']) # Comparing age to age
# graph.show()

# graph = px.scatter_matrix(base_credit, dimensions=['age', 'income']) # All combination of axes (age, income; income, age; age, age)
# graph.show()

graph = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color='default') # See points colored based on 'default' attribute
graph.show()