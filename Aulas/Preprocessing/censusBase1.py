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

print(base_census.describe())
print(base_census.isnull().sum())