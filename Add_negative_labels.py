import json

# Load test.json and build a reverse lookup: (set_id, index) → item_id
with open("data/polyvore_outfits/disjoint/valid.json") as f:
    json_data = json.load(f)

lookup = {}
for entry in json_data:
    set_id = entry["set_id"]
    for item in entry["items"]:
        lookup[(set_id, item["index"])] = item["item_id"]

# Parse compatibility_test.txt and extract NEGATIVE samples only
negative_data = []

with open("data/polyvore_outfits/disjoint/compatibility_valid.txt") as f:
    for line in f:
        parts = line.strip().split()
        label = int(parts[0])
        if label != 0:
            continue  # Skip positives

        item_ids = []
        for ref in parts[1:]:
            set_id, index = ref.split("_")
            index = int(index)
            item_id = lookup.get((set_id, index))
            if item_id:
                item_ids.append(item_id)

        if item_ids:
            negative_data.append({
                "item_ids": item_ids,
                "label": 0
            })

# Save to file
with open("data/polyvore_outfits/disjoint/negatives.json", "w") as f:
    json.dump(negative_data, f, indent=2)

print(f"✅ Saved {len(negative_data)} negative samples to 'negative_data.json'")