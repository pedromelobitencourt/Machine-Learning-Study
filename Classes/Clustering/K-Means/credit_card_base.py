import plotly.express as px
import plotly.graph_objects as go # concatenate graphs
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd

credit_card_base = pd.read_csv('../../../Databases/credit_card_clients.csv', header=1) # header=1: separate headers from data
print(credit_card_base)