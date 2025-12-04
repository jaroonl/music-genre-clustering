import pandas as pd
from sklearn.preprocessing import StandardScaler #transforms each feature to have mean = 0 and sd = 1 bc PCA and K-Means perform better on normalized data
import numpy as np

INPUT_FILE = "results/cleaned.csv"
OUTPUT_X_FILE = "results/X_scaled.npy" #NumPy matrix of noramlized features
OUTPUT_DF_FILE = "results/cleaned_with_index.csv" #CSV copy with consistent indexing

#loading cleaned dataset
df = pd.read_csv(INPUT_FILE)

feature_cols = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "duration_ms"
]

X = df[feature_cols].to_numpy() #creating NumPy matrix to normalize

# normalizing features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) #computing mean/std and applying to scaling

# saving results
np.save(OUTPUT_X_FILE, X_scaled)
df.to_csv(OUTPUT_DF_FILE, index=False)

print("Step 2 done → saved normalized matrix:", OUTPUT_X_FILE)
