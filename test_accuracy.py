import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from dataset.polyvore_dataset import PolyvoreDataset
from models.resnet_encoder import ResNetEncoder
from models.lstm_model import OutfitLSTM
from tqdm import tqdm
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load best model
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

model = FashionCompatibilityModel().to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pth"))

# Create test loader
transform = transforms.Compose([
    transforms.Resize((224, 224)),
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

test_dataset = PolyvoreDataset("data/polyvore_outfits/disjoint/combined_test.json", "data/polyvore_outfits/images", transform)

# Use subset for speed
num_samples = int(0.03 * len(test_dataset))
indices = random.sample(range(len(test_dataset)), num_samples)
test_subset = Subset(test_dataset, indices)
test_loader = DataLoader(test_subset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)

criterion = nn.BCEWithLogitsLoss()

# Evaluate
@torch.no_grad()
def evaluate(model, test_loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    TP = TN = FP = FN = 0
    
    for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.float().to(device)
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        preds = torch.sigmoid(outputs) > 0.5
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Confusion matrix
        TP += ((preds == 1) & (labels == 1)).sum().item()
        TN += ((preds == 0) & (labels == 0)).sum().item()
        FP += ((preds == 1) & (labels == 0)).sum().item()
        FN += ((preds == 0) & (labels == 1)).sum().item()
    
    accuracy = 100 * correct / total
    loss_avg = total_loss / len(test_loader)
    
    # Precision, Recall, F1
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*60)
    print("TEST RESULTS (Best Model)")
    print("="*60)
    print(f"Loss: {loss_avg:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {TP}")
    print(f"  True Negatives:  {TN}")
    print(f"  False Positives: {FP}")
    print(f"  False Negatives: {FN}")
    print("="*60 + "\n")

evaluate(model, test_loader)
