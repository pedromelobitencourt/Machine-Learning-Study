import plotly.express as px
import plotly.graph_objects as go # concatenate graphs
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

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

dbscan = DBSCAN(eps=0.95, min_samples=2) # min_samples: how many elements should be in the area to be considered a cluster
dbscan.fit(salary_base)

labels = dbscan.labels_
print(labels)