# Combine 'income' and 'age'
import pandas as pd
import plotly.express as px

credit = pd.read_csv('../../Databases/credit_data.csv')

graph = px.scatter(x=credit['income'], y=credit['age'])
graph.show()

# Combine 'income' and 'loan'

graph = px.scatter(x=credit['income'], y=credit['loan'])
graph.show()

# Combine 'age' and 'loan'

graph = px.scatter(x=credit['age'], y=credit['loan'])
graph.show()


census = pd.read_csv('../../Databases/census.csv')

# Combine 'age' and 'final-weight'

graph = px.scatter(x=census['age'], y=census['final-weight'])
graph.show()