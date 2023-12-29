import plotly.figure_factory as ff
import plotly.subplots as sp
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering

x = [20, 27, 21, 37, 46, 53, 55, 47, 52, 32, 39, 41, 39, 48, 48]
y = [1000, 1200, 2900, 1850, 900, 950, 2000, 2100, 3000, 5900, 4100, 5100, 7000, 5000, 6500]

new_list = []

for i in range(len(x)):
    new_list.append([x[i], y[i]])

salary_base = np.array(new_list)

salary_scaler = StandardScaler()
salary_base = salary_scaler.fit_transform(salary_base)

print(salary_base)

# Perform hierarchical clustering
linked = linkage(salary_base, method='ward')

# Create dendrogram using Plotly
dendrogram_fig = ff.create_dendrogram(linked, orientation='bottom')

# Create scatter plot
scatter_fig = px.scatter(x=x, y=y)

# Create subplots
combined_fig = sp.make_subplots(rows=1, cols=2, subplot_titles=('Dendrogram', 'Scatter Plot'))

# Add dendrogram to the first subplot
for trace in dendrogram_fig['data']:
    combined_fig.add_trace(trace, row=1, col=1)

# Add scatter plot to the second subplot
combined_fig.add_trace(scatter_fig['data'][0], row=1, col=2)

# Update layout
combined_fig.update_layout(height=400, width=800, title_text="Dendrogram and Scatter Plot")

# Show the combined figure
combined_fig.show()

hc = AgglomerativeClustering(n_clusters=3, linkage='ward', affinity='euclidean')
labels = hc.fit_predict(salary_base)

print('Labels', labels)