import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


class PolyvoreDataset(Dataset):
    def __init__(self, json_path, image_dir, transform=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    
    
    def __getitem__(self, idx):
        outfit = self.data[idx]
        item_ids = outfit["item_ids"]


        images = []
        for item_id in item_ids:
            img_path = os.path.join(self.image_dir, f"{item_id}.jpg")
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)

        images = torch.stack(images, dim=0)  # Shape: [num_items, 3, H, W]

        label = outfit["label"]

        return images, label
