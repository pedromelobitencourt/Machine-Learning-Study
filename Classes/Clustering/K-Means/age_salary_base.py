import plotly.express as px
import plotly.graph_objects as go # concatenate graphs
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

x = [ 20, 27, 21, 37, 46, 53, 55, 47, 52, 32, 39, 41, 39, 48, 48 ]
y = [ 1000, 1200, 2900, 1850, 900, 950, 2000, 2100, 3000, 5900, 4100, 5100, 7000, 5000, 6500 ]

graph = px.scatter(x = x, y = y)
# graph.show()

new_list = []

for i in range(len(x)):
    new_list.append([x[i], y[i]])

salary_base = np.array(new_list)

salary_scaler = StandardScaler()
salary_base = salary_scaler.fit_transform(salary_base)

print(salary_base)

salary_kmeans = KMeans(n_clusters=3)
salary_kmeans.fit(salary_base)

centroids = salary_kmeans.cluster_centers_
print(centroids)

print(salary_scaler.inverse_transform(salary_kmeans.cluster_centers_)) # the centroid's coordinate

labels = salary_kmeans.labels_
print(labels) # An array showing each element group

graph1 = px.scatter(x=salary_base[:,0], y=salary_base[:, 1], color=labels)
graph2 = px.scatter(x=centroids[:, 0], y=centroids[:, 1], size=[12, 12, 12])
graph3 = go.Figure(data=graph1.data + graph2.data)
graph3.show()