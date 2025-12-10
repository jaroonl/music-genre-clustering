import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

#Loading the PCA data
X_pca = np.load("results/X_pca.npy")

#Elbow Method to Estimate K

#SSE (sum of squared distances between each data point and the centroid of the cluster)
inertias = [] 
K_range = range(2,15)

for k in K_range:
    km = KMeans(n_clusters=k, n_init = 'auto', random_state=42)
    km.fit(X_pca)
    inertias.append(km.inertia_) #store the SSE for this k

#Plot the elbow curve
plt.plot(K_range, inertias, marker='o')
plt.xlabel("K (number of clusters)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.savefig("results/elbowMethod.png") #Elbow is around 8-9
plt.show()


#Silhouette score 
# measures how similiar an object is to its own cluster compared to other clusters
sil_scores = []

#Sample 10000 points
sample_idx = np.random.choice(len(X_pca), size=10000, replace=False)
X_sample = X_pca[sample_idx]

for k in K_range:
    km = KMeans(n_clusters = k, n_init='auto', random_state=42)
    labels = km.fit_predict(X_sample)
    sil_scores.append(silhouette_score(X_sample,labels))

#Plot the silhouette scores
plt.plot(K_range, sil_scores, marker = 'o')
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores (sample)")
plt.savefig("results/silhouetteScore.png")
plt.show() #Close the plot

#Combined graph
#Plot Inertia (left axis)
fig,ax1 = plt.subplots(figsize=(10,5))
color = "tab:blue"
ax1.set_xlabel("K")
ax1.set_ylabel("Inertia", color=color)
ax1.plot(K_range, inertias, marker='o', color=color, label="Inertia(Elbow)")
ax1.tick_params(axis='y', labelcolor=color)

#Plot silhouette score (right axis)
ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel("Silhouette Score", color=color)
ax2.plot(K_range, sil_scores,marker='s',color=color, label="Silhouette Score")
ax2.tick_params(axis='y',labelcolor=color)

plt.title("Elbow Method and Silhouette Score")
fig.tight_layout()
plt.savefig("results/combinedGraph.png")
plt.show()

#Fit KMeans

best_k = 9  #has a higher silhouette score than 8
kmeans = KMeans(n_clusters=best_k, n_init="auto",random_state=42)
labels = kmeans.fit_predict(X_pca)

#Save cluster assignments for visualization
np.save("results/kmeans_labels.npy", labels)

#Append cluster labels to DataFrame
df = pd.read_csv("results/pca_ready.csv")
df["cluster"] = labels
df.to_csv("results/pca_with_clusters.csv",index = False) #Save to use for visualization