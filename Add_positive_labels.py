import json

# Load test.json
with open("data/polyvore_outfits/disjoint/valid.json") as f:
    data = json.load(f)

# Add label = 1 to each
for entry in data:
    entry["label"] = 1
    entry["item_ids"] = [item["item_id"] for item in entry["items"]]
    del entry["items"]
    if "set_id" in entry:
        del entry["set_id"]  # optional: remove extra metadata

# Save to new file
with open("data/polyvore_outfits/disjoint/postives.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"✅ Done: Added label=1 to {len(data)} outfits from valid.json")