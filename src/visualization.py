import umap.umap_ as umap
import plotly.express as px
import numpy as np
import pandas as pd

#Load PCA reduced data and Kmeans cluster labels
X_pca = np.load("results/X_pca.npy")
labels_full = np.load("results/kmeans_labels.npy")

#Loading a random sample from the full dataset
sample_size = 10000
sample_idx = np.random.choice(len(X_pca), size=sample_size, replace=False)

X_sample = X_pca[sample_idx]        #Sampled PCA features
labels = labels_full[sample_idx]    #Corresponding cluster labels

#PCA --> UMAP 2D 
umap_2d = umap.UMAP(
    n_neighbors = 30,
    min_dist = 0.1,
    n_components = 2,
    random_state=42
).fit_transform(X_sample)

np.save("results/umap_2d.npy",umap_2d)

#Plot 2D UMAP
fig = px.scatter(
    x = umap_2d[:,0],
    y = umap_2d[:,1], 
    color = labels.astype(str),
    title = "UMAP Clusters",
    opacity = 0.7
)

fig.show()

#Creating cluster summaries
df = pd.read_csv("results/pca_with_clusters.csv")
cluster_summary = df.groupby("cluster").mean(numeric_only=True)
cluster_summary.to_csv("results/cluster_summary.csv")
