import plotly.express as px
import plotly.graph_objects as go # concatenate graphs
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd

credit_card_base = pd.read_csv('../../../Databases/credit_card_clients.csv', header=1) # header=1: separate headers from data
credit_card_base['BILL_TOTAL'] = 0


for i in range(1, 7):
    credit_card_base['BILL_TOTAL'] += credit_card_base[f'BILL_AMT{i}']

x_card = credit_card_base.iloc[:,[ 1, 25 ] ].values
card_scaler = StandardScaler()
x_card = card_scaler.fit_transform(x_card)

wcss = []

for i in range(1, 11):
    card_kmeans = KMeans(n_clusters=i, random_state=0)
    card_kmeans.fit(x_card)
    wcss.append(card_kmeans.inertia_)

print(wcss) # the wcss shouldn't have a big fall

graph = px.line(x=range(1, 11), y=wcss)

card_kmeans = KMeans(n_clusters=4, random_state=0)
labels = card_kmeans.fit_predict(x_card)

graph = px.scatter(x=x_card[:, 0], y=x_card[:, 1], color=labels)
graph.show()

client_list = np.column_stack((credit_card_base, labels))
client_list = client_list[client_list[:, 26].argsort()]
print(client_list)