#!/usr/bin/env python3
"""Download sample images for testing.

Downloads 100 sample images at 64x64 resolution from CIFAR-100 dataset.
"""

from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image


def main():
    output_dir = Path("samples/original")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading CIFAR-100 dataset...")
    
    # Transform to resize to 64x64
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
    ])

    dataset = torchvision.datasets.CIFAR100(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    print(f"Saving 100 sample images to {output_dir}/...")
    
    # Save 100 images
    for i in range(100):
        img_tensor, label = dataset[i]
        
        # Convert tensor to PIL Image
        img = img_tensor.permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).numpy()
        pil_img = Image.fromarray(img)
        
        # Save with label info in filename
        class_name = dataset.classes[label]
        output_path = output_dir / f"sample_{i:03d}_{class_name}.png"
        pil_img.save(output_path)
        
        if (i + 1) % 20 == 0:
            print(f"  Saved {i + 1}/100 images")

    print(f"\nDone! Saved 100 images to {output_dir}/")


if __name__ == "__main__":
    main()
