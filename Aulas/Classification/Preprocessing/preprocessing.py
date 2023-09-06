import pandas as pd # Load csv files
import numpy as np
import seaborn as sns # View graphs
import matplotlib.pyplot as plt # View graphs
import plotly.express as px # Create a dynamic graph

base_credit = pd.read_csv("credit_data.csv")

print(base_credit.head(15))
print()
print(base_credit.describe(), '\n') # Some info, like: mean: media, std: devio padrao
print(base_credit[base_credit['income'] >= 69995.68]) # Get the line that the income is greater or equal to 69995.68