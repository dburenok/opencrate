import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import umap

# ---- Settings ----
INDEX_DIR = Path("../opencrate_index")
EMB_BIN = INDEX_DIR / "embeddings.bin"
DIM = 512
SAMPLE_SIZE = 100_000
# ------------------

print("Loading embeddings...")
embeddings = np.fromfile(EMB_BIN, dtype=np.float32).reshape(-1, DIM)
total_points = len(embeddings)
print(f"Total points: {total_points:,}")

# Downsample for speed
if total_points > SAMPLE_SIZE:
    idx = np.random.choice(total_points, SAMPLE_SIZE, replace=False)
    embeddings = embeddings[idx]
    print(f"Using random sample of {SAMPLE_SIZE:,} points.")

# Compute UMAP projection
print("Computing UMAP projection...")
coords = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine"
).fit_transform(embeddings)

# Plot
print("Plotting...")
plt.figure(figsize=(10, 8))
plt.scatter(coords[:, 0], coords[:, 1], s=1, alpha=0.6)
plt.title("Audio Embeddings (UMAP 2D Projection)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.show()
