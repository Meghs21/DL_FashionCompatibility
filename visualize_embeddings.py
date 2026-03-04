import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Load embeddings
print("Loading embeddings...")
data = torch.load("embeddings_valid.pt")
embeddings = data['embeddings']  # [num_samples, num_items, 512]
labels = data['labels'].numpy()

# Flatten to [num_samples * num_items, 512] for visualization
num_samples, num_items, embed_dim = embeddings.shape
embeddings_flat = embeddings.reshape(num_samples * num_items, embed_dim).numpy()
labels_flat = np.repeat(labels, num_items)  # repeat each label for num_items

# Standardize
print("Standardizing features...")
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings_flat)

# t-SNE reduction to 2D (can take a few minutes)
print("Applying t-SNE (this may take a minute)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
embeddings_2d = tsne.fit_transform(embeddings_scaled)

# Plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                     c=labels_flat, cmap='viridis', alpha=0.6, s=30)
plt.colorbar(scatter, label="Label (0=Incompatible, 1=Compatible)")
plt.title("Fashion Outfit Embeddings (t-SNE)", fontsize=14, fontweight='bold')
plt.xlabel("t-SNE Dimension 1", fontsize=12)
plt.ylabel("t-SNE Dimension 2", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("embeddings_tsne.png", dpi=300)
print("✅ Saved t-SNE plot to embeddings_tsne.png")
plt.show()
