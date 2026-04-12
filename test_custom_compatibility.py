import argparse
import os
from typing import List

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from models.lstm_model import OutfitLSTM
from models.resnet_encoder import ResNetEncoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class FashionCompatibilityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNetEncoder()
        self.lstm = OutfitLSTM()

    def forward(self, outfit_images: torch.Tensor) -> torch.Tensor:
        # Input shape: [B, N, C, H, W]
        batch_size, outfit_len, channels, height, width = outfit_images.shape
        outfit_images = outfit_images.view(batch_size * outfit_len, channels, height, width)
        features = self.encoder(outfit_images)
        features = features.view(batch_size, outfit_len, -1)
        return self.lstm(features)


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )


def load_image(path: str, tfm: transforms.Compose) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return tfm(image)


def validate_image_paths(paths: List[str]) -> None:
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "These image files were not found:\n" + "\n".join(missing)
        )


def collect_images_from_dir(image_dir: str) -> List[str]:
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = [
        os.path.join(image_dir, name)
        for name in sorted(os.listdir(image_dir))
        if name.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)
    ]

    if not image_paths:
        raise ValueError(
            f"No supported images found in {image_dir}. "
            f"Supported: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}"
        )

    return image_paths


def create_preview(image_paths: List[str], output_path: str, thumb_size=(224, 224), padding=10) -> str:
    images = [Image.open(path).convert("RGB").resize(thumb_size) for path in image_paths]
    count = len(images)
    width = count * thumb_size[0] + (count + 1) * padding
    height = thumb_size[1] + 2 * padding

    canvas = Image.new("RGB", (width, height), color=(245, 245, 245))
    x = padding
    for image in images:
        canvas.paste(image, (x, padding))
        x += thumb_size[0] + padding

    canvas.save(output_path)
    return output_path


def open_preview_if_requested(preview_path: str, should_open: bool) -> None:
    if not should_open:
        return
    if hasattr(os, "startfile"):
        os.startfile(preview_path)
    else:
        print("Preview generated, but auto-open is not supported on this OS.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict outfit compatibility from custom images."
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.pth",
        help="Path to .pth checkpoint (default: checkpoints/best_model.pth)",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        help="Image paths for one outfit. Example: --images top.jpg jeans.jpg shoes.jpg",
    )
    parser.add_argument(
        "--image-dir",
        help="Folder containing outfit images. All supported image files will be loaded.",
    )
    parser.add_argument(
        "--preview-out",
        default="custom_outfit_preview.jpg",
        help="Path to save a collage preview of provided images.",
    )
    parser.add_argument(
        "--open-preview",
        action="store_true",
        help="Open the generated preview image after inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.images and not args.image_dir:
        raise ValueError("Provide either --images or --image-dir.")
    if args.images and args.image_dir:
        raise ValueError("Use either --images or --image-dir, not both.")

    image_paths = args.images if args.images else collect_images_from_dir(args.image_dir)
    if args.image_dir:
        print(
            "Note: --image-dir loads files in sorted filename order. "
            "Use names like 01_top.jpg, 02_bottom.jpg, 03_shoes.jpg to control sequence."
        )

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    validate_image_paths(image_paths)

    model = FashionCompatibilityModel().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    transform = build_transform()
    item_tensors = [load_image(path, transform) for path in image_paths]
    preview_path = create_preview(image_paths, args.preview_out)

    # Shape expected by model: [B, N, C, H, W]
    outfit_tensor = torch.stack(item_tensors, dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(outfit_tensor).squeeze().item()
        probability = torch.sigmoid(torch.tensor(logit)).item()

    print("=" * 58)
    print("CUSTOM OUTFIT COMPATIBILITY TEST")
    print("=" * 58)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Items in outfit: {len(image_paths)}")
    print("Images:")
    for index, path in enumerate(image_paths, start=1):
        print(f"  {index}. {path}")

    print("\nResults:")
    print(f"  Compatibility probability: {probability:.4f}")
    print(f"  Predicted class: {'Compatible' if probability >= 0.5 else 'Incompatible'}")
    print(f"  Preview saved at: {preview_path}")
    print("=" * 58)

    open_preview_if_requested(preview_path, args.open_preview)


if __name__ == "__main__":
    main()
