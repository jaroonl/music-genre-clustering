import pandas as pd
import numpy as np
from sklearn.decomposition import PCA #contains algorithm for dimensionality reduction (high dimen)

INPUT_X = "results/X_scaled.npy" #normalized numeric matrix for PCA
INPUT_DF = "results/cleaned_with_index.csv" #metadata and features matched to rows in PCA
OUTPUT_PCA = "results/X_pca.npy" #PCA transformed feature matrix for clustering
OUTPUT_DF = "results/pca_ready.csv" #original dataset aligned with PCA (interpretation)

#save PCA results
df = pd.read_csv(INPUT_DF)
X_scaled = np.load(INPUT_X)

# first PCA pass to get full variance curve
pca_full = PCA()
pca_full.fit(X_scaled)

#computing cumulative explained variance ratio
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = np.argmax(cum_var >= 0.95) + 1

print("95% variance achieved with:", n_components_95, "components")

# reducing to 95% variance (retaining exactly 95 components)
pca = PCA(n_components=n_components_95)
X_pca = pca.fit_transform(X_scaled)

# saves PCA-reduced matrix as NumPy file, input for clustering/UMAP
np.save(OUTPUT_PCA, X_pca)

#DataFrame aligned with PCA rowss
df.to_csv(OUTPUT_DF, index=False) 

print("Step 3 done → saved PCA matrix:", OUTPUT_PCA)
