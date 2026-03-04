import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

# Load model and embeddings
print("Loading model and embeddings...")
from dataset.polyvore_dataset import PolyvoreDataset
from models.resnet_encoder import ResNetEncoder
from models.lstm_model import OutfitLSTM
from torchvision import transforms
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load embeddings and original data
embeddings_data = torch.load("embeddings_valid.pt")
embeddings = embeddings_data['embeddings']  # [num_samples, num_items, 512]
labels = embeddings_data['labels'].numpy()

# Load original outfit data for reference
with open("data/polyvore_outfits/disjoint/combined_valid.json") as f:
    outfits = json.load(f)

print(f"Loaded {len(outfits)} outfits from validation set")

# Function to get outfit embedding (average of item embeddings)
def get_outfit_embedding(embed, averaging=True):
    """embed: [num_items, 512]"""
    if averaging:
        return embed.mean(dim=0)  # [512]
    else:
        return embed[-1]  # use last item representation

# Convert to outfit-level embeddings
outfit_embeddings = torch.stack([
    get_outfit_embedding(embeddings[i]) for i in range(len(embeddings))
])  # [num_samples, 512]

outfit_embeddings_np = outfit_embeddings.numpy()

# Function: Find similar outfits
def find_similar_outfits(query_idx, top_k=5):
    """Find top-k most similar outfits to query outfit"""
    query_embed = outfit_embeddings_np[query_idx:query_idx+1]
    similarities = cosine_similarity(query_embed, outfit_embeddings_np)[0]
    
    # Get top-k (excluding query itself)
    top_indices = np.argsort(-similarities)[1:top_k+1]
    
    print(f"\n📌 Query Outfit #{query_idx}")
    print(f"   Items: {outfits[query_idx]['item_ids']}")
    print(f"   Label: {'✅ Compatible' if labels[query_idx] else '❌ Incompatible'}")
    print(f"\n🔍 Top {top_k} Similar Outfits:")
    
    for rank, idx in enumerate(top_indices, 1):
        sim_score = similarities[idx]
        print(f"   {rank}. Outfit #{idx} (Similarity: {sim_score:.4f})")
        print(f"      Items: {outfits[idx]['item_ids']}")
        print(f"      Label: {'✅ Compatible' if labels[idx] else '❌ Incompatible'}")

# Function: Find compatible items to add to outfit
def recommend_items_for_outfit(query_idx, candidate_indices=None, top_k=5):
    """Given an outfit, recommend items to add"""
    if candidate_indices is None:
        candidate_indices = list(range(len(outfits)))
    
    query_embed = outfit_embeddings_np[query_idx]
    
    print(f"\n👗 Base Outfit #{query_idx}")
    print(f"   Items: {outfits[query_idx]['item_ids']}")
    
    print(f"\n💡 Recommended similar outfits to mix with:")
    
    best_matches = []
    for candidate_idx in candidate_indices:
        if candidate_idx == query_idx:
            continue
        candidate_embed = outfit_embeddings_np[candidate_idx]
        similarity = cosine_similarity([query_embed], [candidate_embed])[0, 0]
        best_matches.append((candidate_idx, similarity))
    
    best_matches.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (idx, sim) in enumerate(best_matches[:top_k], 1):
        print(f"   {rank}. Outfit #{idx} (Match Score: {sim:.4f})")
        print(f"      Items: {outfits[idx]['item_ids']}")

# Example usage
if __name__ == "__main__":
    print("\n" + "="*60)
    print("FASHION OUTFIT RECOMMENDATION SYSTEM")
    print("="*60)
    
    # Test: Find similar outfits
    query_outfit_idx = 0
    find_similar_outfits(query_outfit_idx, top_k=5)
    
    # Test: Recommend items for an outfit
    print("\n" + "="*60)
    recommend_items_for_outfit(query_outfit_idx, top_k=5)
    
    print("\n✅ To use these functions:")
    print("   - Import this module")
    print("   - Call find_similar_outfits(outfit_id)")
    print("   - Call recommend_items_for_outfit(outfit_id)")
