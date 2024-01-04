import pandas as pd
import plotly.express as px

census = pd.read_csv('../../Databases/census.csv')
credit = pd.read_csv('../../Databases/credit_data.csv')

print(credit.isnull().sum(), '\n==========================================')
credit.dropna(inplace=True)
print(credit.isnull().sum(), '\n==========================================')

# Age Outliers
graph = px.box(credit, y='age')
graph.show() # q1: 25% of data; q3: 75% of data; median: 50% of data

age_outliers = credit[credit['age'] < 0]
print('Age outliers:', age_outliers, '\n==========================================')

# Loan Outliers
graph = px.box(credit, y='loan')
graph.show()

loan_outliers = credit[credit['loan'] > 13300]
print('Loan outliers:', loan_outliers)