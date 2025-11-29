import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

INPUT_FILE = "results/cleaned.csv"
OUTPUT_X_FILE = "results/X_scaled.npy"
OUTPUT_DF_FILE = "results/cleaned_with_index.csv"

df = pd.read_csv(INPUT_FILE)

feature_cols = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "duration_ms"
]

X = df[feature_cols].to_numpy()

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save results
np.save(OUTPUT_X_FILE, X_scaled)
df.to_csv(OUTPUT_DF_FILE, index=False)

print("Step 2 done → saved normalized matrix:", OUTPUT_X_FILE)
