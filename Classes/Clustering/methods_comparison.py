import plotly.express as px
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

x_random, y_random = datasets.make_moons(n_samples=1500, noise=0.09)
graph = px.scatter(x=x_random[:, 0], y=x_random[:, 1])
graph.show()

kmeans = KMeans(n_clusters=2)
labels = kmeans_labels = kmeans.fit_predict(x_random)
kmeans_graph = px.scatter(x=x_random[:, 0], y=x_random[:, 1], color=labels)
kmeans_graph.show()

hc = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
labels = hc.fit_predict(x_random)

hc_graph = px.scatter(x=x_random[:, 0], y=x_random[:, 1], color=labels)
hc_graph.show()

dbscan = DBSCAN(eps=0.1)
labels = dbscan.fit_predict(x=x_random)
dbscan_graph = px.scatter(x=x_random[:, 0], y=x_random[:, 1], color=labels)
dbscan_graph.show()