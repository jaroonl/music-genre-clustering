import pandas as pd

INPUT_FILE = "data/tracks_features.csv"      # change if needed
OUTPUT_FILE = "results/cleaned.csv"

# Load dataset
df = pd.read_csv(INPUT_FILE)

# Keep only needed audio feature columns
feature_cols = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "duration_ms"
]

# Drop rows with missing feature values
df_clean = df.dropna(subset=feature_cols)

# Save cleaned dataset
df_clean.to_csv(OUTPUT_FILE, index=False)

print("Step 1 done → saved cleaned dataset:", OUTPUT_FILE)
