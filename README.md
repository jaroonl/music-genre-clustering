# Music Genre Clustering – Project Pipeline

This project performs unsupervised clustering on a large Spotify dataset using audio features such as danceability, energy, acousticness, etc.  
The final goal: identify natural music “genre-like” groupings using PCA, K-Means, and UMAP.

---

## Project Steps Overview

### **Step 1 — Load + Clean (`src/loadAndClean.py`)**
- Loads the raw Spotify dataset from `data/`.
- Selects the audio feature columns required for the project.
- Drops rows with missing values.
- Saves a cleaned CSV into:  
  ➤ `results/cleaned.csv`

### **Step 2 — Normalize (`src/normalize.py`)**
- Loads the cleaned file.
- Applies `StandardScaler` normalization to all numerical audio features.
- Saves:
  - Normalized feature matrix → `results/X_scaled.npy`
  - Clean DataFrame with stable indexing → `results/cleaned_with_index.csv`

### **Step 3 — PCA (to 95% variance) (`src/pca.py`)**
- Loads normalized feature matrix.
- Performs PCA, determines #components needed for **95% explained variance**.
- Transforms data with PCA.
- Saves PCA-reduced matrix → `results/X_pca.npy`  
- Saves DataFrame aligned with PCA data → `results/pca_ready.csv`

---

## Steps Remaining

### **Step 4 — K-Means Clustering**
- Load `results/X_pca.npy`
- Use Elbow & Silhouette methods to pick K
- Fit KMeans
- Save cluster labels and optionally append to `pca_ready.csv`

### **Step 5 — UMAP 2D Visualization**
- Load PCA matrix and cluster labels
- Run UMAP to reduce to 2D
- Plot clusters using Plotly or Matplotlib
- Produce visuals for final report

### **Step 6 — Cluster Interpretation**
- Compute mean feature values per cluster
- Interpret clusters (e.g., “high energy + high tempo → EDM-type cluster")
- Add sample songs per cluster
- Produce final write-up / report

---

## 📁 Repository Structure

