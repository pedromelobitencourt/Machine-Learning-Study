import plotly.express as px
import plotly.graph_objects as go # concatenate graphs
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import pandas as pd
from sklearn.decomposition import PCA

credit_card_base = pd.read_csv('../../../Databases/credit_card_clients.csv', header=1) # header=1: separate headers from data
credit_card_base['BILL_TOTAL'] = 0


for i in range(1, 7):
    credit_card_base['BILL_TOTAL'] += credit_card_base[f'BILL_AMT{i}']

x_card = credit_card_base.iloc[:,[ 1, 25 ] ].values
card_scaler = StandardScaler()
x_card = card_scaler.fit_transform(x_card)

dbscan = DBSCAN(eps=0.37, min_samples=5)
labels = dbscan.fit_predict(x_card)

print(np.unique(labels, return_counts=True))

graph = px.scatter(x=x_card[:, 0], y=x_card[:, 1], color=labels)
graph.show()