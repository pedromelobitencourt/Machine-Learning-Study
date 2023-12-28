import plotly.express as px
import plotly.graph_objects as go # concatenate graphs
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs # generate random data

x_random, y_random = make_blobs(n_samples=200, centers=5)

graph = px.scatter(x=x_random[:, 0], y=x_random[:, 1])

blobs_kmeans = KMeans(n_clusters=5)
blobs_kmeans.fit(x_random)

labels = blobs_kmeans.predict(x_random)
centroids = blobs_kmeans.cluster_centers_

graph1 = px.scatter(x=x_random[:, 0], y=x_random[:, 1], color=labels)
graph2 = px.scatter(x=centroids[:, 0], y=centroids[:, 1], size=[5, 5, 5, 5, 5]) # five positions because there is 5 clusters and the value 5 is the size of the centroid in the graph
graph3 = go.Figure(data=graph1.data + graph2.data)
graph3.show()