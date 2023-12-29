import plotly.express as px
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

x_random, y_random = datasets.make_moons(n_samples=1500, noise=0.09)
graph = px.scatter(x=x_random[:, 0], y=x_random[:, 1])
graph.show()

kmeans = KMeans(n_clusters=2)
labels = kmeans_labels = kmeans.fit_predict(x_random)
kmeans_graph = px.scatter(x=x_random[:, 0], y=x_random[:, 1], color=labels)
kmeans_graph.show()

hc = AgglomerativeClustering(n_clusters=2, affinity='euclidean')