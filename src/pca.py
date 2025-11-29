import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

INPUT_X = "results/X_scaled.npy"
INPUT_DF = "results/cleaned_with_index.csv"
OUTPUT_PCA = "results/X_pca.npy"
OUTPUT_DF = "results/pca_ready.csv"

df = pd.read_csv(INPUT_DF)
X_scaled = np.load(INPUT_X)

# First PCA pass to compute total variance curve
pca_full = PCA()
pca_full.fit(X_scaled)

cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = np.argmax(cum_var >= 0.95) + 1

print("95% variance achieved with:", n_components_95, "components")

# Reduce to 95% variance
pca = PCA(n_components=n_components_95)
X_pca = pca.fit_transform(X_scaled)

# Save outputs for your teammate
np.save(OUTPUT_PCA, X_pca)
df.to_csv(OUTPUT_DF, index=False)

print("Step 3 done → saved PCA matrix:", OUTPUT_PCA)
