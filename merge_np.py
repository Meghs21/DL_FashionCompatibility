import json
import random

with open("data/polyvore_outfits/disjoint/postives.json") as f:
    pos = json.load(f)

with open("data/polyvore_outfits/disjoint/negatives.json") as f:
    neg = json.load(f)

combined = pos + neg

random.shuffle(combined)

with open("data/polyvore_outfits/disjoint/combined_valid.json", "w") as f:
    json.dump(combined, f, indent=2)

print(f"✅ Merged {len(pos)} positive + {len(neg)} negative = {len(combined)} total samples")
