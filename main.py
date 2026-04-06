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

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# AMP helpers (new torch.amp API with fallback for older torch versions).
if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
    def amp_autocast():
        return torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda")

    def build_grad_scaler():
        return torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
else:
    from torch.cuda.amp import autocast as _legacy_autocast, GradScaler as _LegacyGradScaler

    def amp_autocast():
        return _legacy_autocast(enabled=device.type == "cuda")

    def build_grad_scaler():
        return _LegacyGradScaler(enabled=device.type == "cuda")


def custom_collate_fn(batch):
    outfits, labels = zip(*batch)
    lengths = torch.tensor([len(o) for o in outfits], dtype=torch.long)
    max_len = max(len(o) for o in outfits)

    padded_outfits = []
    for outfit in outfits:
        if isinstance(outfit, torch.Tensor):
            outfit = list(outfit)
        pad_len = max_len - len(outfit)
        if pad_len > 0:
            outfit += [torch.zeros_like(outfit[0]) for _ in range(pad_len)]
        padded_outfits.append(torch.stack(outfit))

    return torch.stack(padded_outfits), torch.tensor(labels), lengths

# -----------------------------
# Data preparation
def get_dataloaders(train_path, val_path, test_path, image_dir, batch_size=8, subset_ratio=0.03):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    def get_subset(dataset, ratio):
        if ratio is None or ratio >= 1.0:
            return dataset
        num_samples = max(1, int(ratio * len(dataset)))
        indices = random.sample(range(len(dataset)), num_samples)
        return Subset(dataset, indices)

    train_dataset = PolyvoreDataset(train_path, image_dir, train_transform)
    val_dataset = PolyvoreDataset(val_path, image_dir, eval_transform)
    test_dataset = PolyvoreDataset(test_path, image_dir, eval_transform)

    use_cuda = device.type == "cuda"
    cpu_workers = os.cpu_count() or 1
    is_windows = os.name == "nt"

    # Windows can hit shared file mapping limits (error 1455) with many workers.
    if is_windows:
        train_workers = min(2, cpu_workers)
        eval_workers = 0
        train_persistent = False
        train_prefetch = 1
    else:
        train_workers = min(8, cpu_workers)
        eval_workers = min(4, cpu_workers)
        train_persistent = train_workers > 0
        train_prefetch = 2

    train_loader_kwargs = {
        "batch_size": batch_size,
        "pin_memory": use_cuda,
        "num_workers": train_workers,
        "collate_fn": custom_collate_fn,
    }
    if train_workers > 0:
        train_loader_kwargs["persistent_workers"] = train_persistent
        train_loader_kwargs["prefetch_factor"] = train_prefetch

    eval_loader_kwargs = {
        "batch_size": batch_size,
        "pin_memory": use_cuda,
        "num_workers": eval_workers,
        "collate_fn": custom_collate_fn,
    }
    if eval_workers > 0:
        eval_loader_kwargs["persistent_workers"] = False
        eval_loader_kwargs["prefetch_factor"] = 1

    return (
        DataLoader(get_subset(train_dataset, subset_ratio), shuffle=True, **train_loader_kwargs),
        DataLoader(val_dataset, shuffle=False, **eval_loader_kwargs),
        DataLoader(test_dataset, shuffle=False, **eval_loader_kwargs),
    )

# -----------------------------
# Model setup
class FashionCompatibilityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNetEncoder()
        self.lstm = OutfitLSTM()

    def forward(self, outfit_images, lengths=None):
        B, N, C, H, W = outfit_images.shape
        if lengths is None:
            lengths = torch.full((B,), N, device=outfit_images.device, dtype=torch.long)
        else:
            lengths = lengths.to(device=outfit_images.device, dtype=torch.long)

        # Encode only real (unpadded) items to avoid wasted CNN compute on padding.
        valid_images = torch.cat(
            [outfit_images[b, :int(lengths[b].item())] for b in range(B)],
            dim=0,
        )
        valid_features = self.encoder(valid_images)
        feature_dim = valid_features.size(1)

        features = valid_features.new_zeros((B, N, feature_dim))
        start = 0
        for b in range(B):
            seq_len = int(lengths[b].item())
            end = start + seq_len
            features[b, :seq_len] = valid_features[start:end]
            start = end

        return self.lstm(features, lengths)

def build_model():
    return FashionCompatibilityModel().to(device)

criterion = nn.BCEWithLogitsLoss()

# -----------------------------
# Validation
@torch.no_grad()
def validate(model, valid_loader):
    model.eval()
    total_loss = 0.0
    for images, labels, lengths in valid_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        with amp_autocast():
            outputs = model(images, lengths).squeeze(1)
            loss = criterion(outputs, labels)
        total_loss += loss.item()
    return total_loss / len(valid_loader)

# -----------------------------
# Test
@torch.no_grad()
def test(model, test_loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels, lengths in tqdm(test_loader, desc="Testing", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        with amp_autocast():
            outputs = model(images, lengths).squeeze(1)
            loss = criterion(outputs, labels)
        total_loss += loss.item()
        preds = torch.sigmoid(outputs) > 0.5
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    print(f"\nTest Loss: {total_loss/len(test_loader):.4f} | Accuracy: {100 * correct / total:.2f}%\n")

# -----------------------------
# Training
def train(model, train_loader, val_loader, num_epochs=10, lr=1e-4, save_path="checkpoints/best_model.pth"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = build_grad_scaler()
    best_val_loss = float("inf")

    with open("checkpoints/loss_log.csv", mode='w', newline='') as f:
        csv.writer(f).writerow(["Epoch", "Train Loss", "Validation Loss"])

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)

        for images, labels, lengths in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp_autocast():
                outputs = model(images, lengths).squeeze(1)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        val_loss = validate(model, val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

        with open("checkpoints/loss_log.csv", mode='a', newline='') as f:
            csv.writer(f).writerow([epoch+1, avg_loss, val_loss])

        # Save checkpoint after every epoch
        epoch_checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), epoch_checkpoint_path)
        print(f"Epoch checkpoint saved: {epoch_checkpoint_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print("Best model updated and saved!")

    return model

# -----------------------------
# Extract and save embeddings
@torch.no_grad()
def save_embeddings(model, data_loader, split_name="valid"):
    model.eval()
    all_embeddings = []
    all_labels = []
    outfit_lengths = []  # Track number of items per outfit
    
    print(f"\nExtracting {split_name} embeddings...")
    for images, labels, lengths in tqdm(data_loader, desc=f"Embedding {split_name}", leave=False):
        images = images.to(device, non_blocking=True)
        B, N, C, H, W = images.shape
        lengths = lengths.tolist()
        
        # Process each outfit individually (variable lengths)
        for b in range(B):
            real_len = lengths[b]
            outfit_images = images[b, :real_len]  # [real_len, C, H, W]
            with amp_autocast():
                outfit_embeddings = model.encoder(outfit_images)  # [N, 512]
            
            all_embeddings.append(outfit_embeddings.cpu())
            all_labels.append(labels[b].item())
            outfit_lengths.append(real_len)
    
    # Save as list of variable-length tensors
    save_path = f"embeddings_{split_name}.pt"
    torch.save({
        'embeddings': all_embeddings,  # list of [num_items, 512] tensors
        'labels': torch.tensor(all_labels),
        'outfit_lengths': outfit_lengths
    }, save_path)
    print(f"✅ Saved {len(all_embeddings)} outfit embeddings to {save_path}")
    return all_embeddings, all_labels


# -----------------------------
# Main
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(
        train_path="data/polyvore_outfits/disjoint/combined_train.json",
        val_path="data/polyvore_outfits/disjoint/combined_valid.json",
        test_path="data/polyvore_outfits/disjoint/combined_test.json",
        image_dir="data/polyvore_outfits/images",
        batch_size=8,
        subset_ratio=0.5  # Using 50% of data
    )

    model = build_model()
    
    # Resume from best saved model if it exists
    if os.path.exists("checkpoints/best_model.pth"):
        model.load_state_dict(torch.load("checkpoints/best_model.pth"))
        print("✅ Resumed training from best_model.pth")
    
    model = train(model, train_loader, val_loader, num_epochs=15)
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    test(model, test_loader)
    
    # Save embeddings for all splits
    save_embeddings(model, train_loader, "train")
    save_embeddings(model, val_loader, "valid")
    save_embeddings(model, test_loader, "test")
