import plotly.express as px
import plotly.graph_objects as go # concatenate graphs
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage
import plotly.figure_factory as ff
import plotly.subplots as sp
from sklearn.cluster import AgglomerativeClustering

credit_card_base = pd.read_csv('../../../Databases/credit_card_clients.csv', header=1) # header=1: separate headers from data
credit_card_base['BILL_TOTAL'] = 0


for i in range(1, 7):
    credit_card_base['BILL_TOTAL'] += credit_card_base[f'BILL_AMT{i}']

x_card = credit_card_base.iloc[:,[ 1, 25 ] ].values
card_scaler = StandardScaler()
x_card = card_scaler.fit_transform(x_card)

print(x_card)

# Perform hierarchical clustering
linked = linkage(x_card, method='ward')

# Create dendrogram using Plotly
dendrogram_fig = ff.create_dendrogram(linked, orientation='bottom')

# Create subplots
combined_fig = sp.make_subplots(rows=1, cols=2, subplot_titles=('Dendrogram', 'Scatter Plot'))

# Add dendrogram to the first subplot
for trace in dendrogram_fig['data']:
    combined_fig.add_trace(trace, row=1, col=1)

# Update layout
combined_fig.update_layout(height=400, width=800, title_text="Dendrogram and Scatter Plot")

# Show the combined figure
combined_fig.show()

hc = AgglomerativeClustering(n_clusters=3, linkage='ward', affinity='euclidean')
labels = hc.fit_predict(x_card)

print('Labels', labels)