import torch

print("="*60)
print("EMBEDDINGS INSPECTION")
print("="*60)

for split in ["train", "valid", "test"]:
    try:
        data = torch.load(f"embeddings_{split}.pt")
        
        embeddings_list = data['embeddings']
        labels = data['labels']
        outfit_lengths = data['outfit_lengths']
        
        print(f"\n📊 {split.upper()} EMBEDDINGS:")
        print(f"   Number of outfits: {len(embeddings_list)}")
        print(f"   Number of labels: {len(labels)}")
        
        # Check embedding dimensions
        print(f"\n   Embedding Details:")
        for i in range(min(3, len(embeddings_list))):
            embedding = embeddings_list[i]
            print(f"     Outfit {i}: {embedding.shape} (items × embedding_dim)")
        print(f"     ...")
        
        # Stats
        total_items = sum(outfit_lengths)
        avg_items_per_outfit = total_items / len(embeddings_list)
        max_items = max(outfit_lengths)
        min_items = min(outfit_lengths)
        
        print(f"\n   Statistics:")
        print(f"     Total items across all outfits: {total_items}")
        print(f"     Avg items per outfit: {avg_items_per_outfit:.2f}")
        print(f"     Min items in an outfit: {min_items}")
        print(f"     Max items in an outfit: {max_items}")
        print(f"     Embedding dimension (vector size): 512")
        
        # Label distribution
        num_compatible = (labels == 1).sum().item()
        num_incompatible = (labels == 0).sum().item()
        print(f"\n   Label Distribution:")
        print(f"     Compatible (1):   {num_compatible} outfits")
        print(f"     Incompatible (0): {num_incompatible} outfits")
        
    except FileNotFoundError:
        print(f"\n❌ {split.upper()}: embeddings_{split}.pt not found")

print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print("Each outfit is represented as a list of item embeddings.")
print("Each item embedding is 512-dimensional (512 features).")
print("Outfits have variable number of items (padded during training).")
print("="*60)
