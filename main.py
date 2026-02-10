import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from dataset.polyvore_dataset import PolyvoreDataset
from models.resnet_encoder import ResNetEncoder
from models.lstm_model import OutfitLSTM

import random
import os
import csv
import numpy as np

# -----------------------------
# Setup
os.makedirs("checkpoints", exist_ok=True)
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Data preparation
def get_dataloaders(train_path, val_path, test_path, image_dir, batch_size=8, subset_ratio=0.03):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    def custom_collate_fn(batch):
        outfits, labels = zip(*batch)
        max_len = max(len(o) for o in outfits)

        padded_outfits = []
        for outfit in outfits:
            if isinstance(outfit, torch.Tensor):
                outfit = list(outfit)
            pad_len = max_len - len(outfit)
            if pad_len > 0:
                outfit += [torch.zeros_like(outfit[0]) for _ in range(pad_len)]
            padded_outfits.append(torch.stack(outfit))

        return torch.stack(padded_outfits), torch.tensor(labels)

    def get_subset(dataset):
        num_samples = int(subset_ratio * len(dataset))
        indices = random.sample(range(len(dataset)), num_samples)
        return Subset(dataset, indices)

    train_dataset = PolyvoreDataset(train_path, image_dir, transform)
    val_dataset = PolyvoreDataset(val_path, image_dir, transform)
    test_dataset = PolyvoreDataset(test_path, image_dir, transform)

    return (
        DataLoader(get_subset(train_dataset), batch_size, shuffle=True, collate_fn=custom_collate_fn),
        DataLoader(get_subset(val_dataset), batch_size, shuffle=True, collate_fn=custom_collate_fn),
        DataLoader(get_subset(test_dataset), batch_size, shuffle=False, collate_fn=custom_collate_fn),
    )

# -----------------------------
# Model setup
class FashionCompatibilityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNetEncoder()
        self.lstm = OutfitLSTM()

    def forward(self, outfit_images):
        B, N, C, H, W = outfit_images.shape
        outfit_images = outfit_images.view(B * N, C, H, W)
        features = self.encoder(outfit_images)
        return self.lstm(features.view(B, N, -1))

def build_model():
    return FashionCompatibilityModel().to(device)

criterion = nn.BCEWithLogitsLoss()

# -----------------------------
# Validation
@torch.no_grad()
def validate(model, valid_loader):
    model.eval()
    total_loss = 0.0
    for images, labels in valid_loader:
        images, labels = images.to(device), labels.float().to(device)
        outputs = model(images).squeeze(1)
        total_loss += criterion(outputs, labels).item()
    return total_loss / len(valid_loader)

# -----------------------------
# Test
@torch.no_grad()
def test(model, test_loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(test_loader, desc="Testing", leave=False):
        images, labels = images.to(device), labels.float().to(device)
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        preds = torch.sigmoid(outputs) > 0.5
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    print(f"\nTest Loss: {total_loss/len(test_loader):.4f} | Accuracy: {100 * correct / total:.2f}%\n")

# -----------------------------
# Training
def train(model, train_loader, val_loader, num_epochs=1, lr=1e-4, save_path="checkpoints/best_model.pth"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_val_loss = float("inf")

    with open("checkpoints/loss_log.csv", mode='w', newline='') as f:
        csv.writer(f).writerow(["Epoch", "Train Loss", "Validation Loss"])

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)

        for images, labels in pbar:
            images, labels = images.to(device), labels.float().to(device)
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        val_loss = validate(model, val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

        with open("checkpoints/loss_log.csv", mode='a', newline='') as f:
            csv.writer(f).writerow([epoch+1, avg_loss, val_loss])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print("Best model updated and saved!")

    return model

# -----------------------------
# Main
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(
        train_path="data/polyvore_outfits/disjoint/combined_train.json",
        val_path="data/polyvore_outfits/disjoint/combined_valid.json",
        test_path="data/polyvore_outfits/disjoint/combined_test.json",
        image_dir="data/polyvore_outfits/images",
        batch_size=8
    )

    model = build_model()
    model = train(model, train_loader, val_loader, num_epochs=1)
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    test(model, test_loader)