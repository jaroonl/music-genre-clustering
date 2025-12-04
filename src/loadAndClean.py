import pandas as pd

INPUT_FILE = "data/tracks_features.csv" #data file given
OUTPUT_FILE = "results/cleaned.csv" #cleaned data file

#loads the dataset
df = pd.read_csv(INPUT_FILE)

# keeps only needed audio feature columns for PCA and clustering
feature_cols = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "duration_ms"
]

# removes any rows with missing feature values
# reason: PCA and KMeans cannot handle missing values
df_clean = df.dropna(subset=feature_cols)

# saves and stores the cleaned dataset so later script can use it
df_clean.to_csv(OUTPUT_FILE, index=False)

print("Step 1 done → saved cleaned dataset:", OUTPUT_FILE)
