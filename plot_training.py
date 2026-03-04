import pandas as pd
import matplotlib.pyplot as plt

# Load loss log
df = pd.read_csv("checkpoints/loss_log.csv")

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(df["Epoch"], df["Train Loss"], marker='o', label="Train Loss", linewidth=2)
plt.plot(df["Epoch"], df["Validation Loss"], marker='s', label="Validation Loss", linewidth=2)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("Training vs Validation Loss", fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("checkpoints/loss_plot.png", dpi=300)
print("✅ Saved loss plot to checkpoints/loss_plot.png")
plt.show()
