#!/bin/bash
# VAE Inference Script
# Reconstructs images through VAE encoder/decoder to verify VAE quality
#
# Usage:
#   ./scripts/inference-vae.sh [input_image] [output_image]
#
# Example:
#   ./scripts/inference-vae.sh samples/test.png samples/reconstructed.png

set -e

CHECKPOINT="checkpoints/vae_e20.pt"
INPUT_IMAGE="${1:-samples/test.png}"
OUTPUT_IMAGE="${2:-samples/vae_reconstructed.png}"

echo "VAE Inference"
echo "============="
echo "Checkpoint: $CHECKPOINT"
echo "Input: $INPUT_IMAGE"
echo "Output: $OUTPUT_IMAGE"
echo ""

uv run python -c "
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

from src.models.vae import create_vae
from src.utils.common import get_device
from src.config import get_config

# Config
device = get_device('auto')
checkpoint_path = '${CHECKPOINT}'
input_path = '${INPUT_IMAGE}'
output_path = '${OUTPUT_IMAGE}'

print(f'Using device: {device}')

# Load VAE config
config = get_config('vae_train')
image_size = config.get('image_size', 64)

# Create and load VAE
print('Loading VAE...')
vae = create_vae(
    image_size=image_size,
    z_channels=config.get('latent_channels', 16),
    ch=config.get('vae_ch', 64),
    ch_mult=tuple(config.get('vae_ch_mult', [1, 2, 4, 4])),
)

ckpt = torch.load(checkpoint_path, map_location=device)
vae.load_state_dict(ckpt['model_state_dict'])
vae = vae.to(device)
vae.eval()
print(f'VAE loaded from {checkpoint_path}')
print(f'  Epoch: {ckpt.get(\"epoch\", \"unknown\")}')

# Load and preprocess input image
print(f'Loading image: {input_path}')
img = Image.open(input_path).convert('RGB')
transform = T.Compose([
    T.Resize((image_size, image_size)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
x = transform(img).unsqueeze(0).to(device)

# Encode and decode
print('Running VAE reconstruction...')
with torch.no_grad():
    # Encode to latent
    mean, logvar = vae.encode(x)
    # Use mean for deterministic reconstruction
    z = mean
    print(f'  Latent shape: {z.shape}')
    
    # Decode back to image
    recon = vae.decode(z)

# Post-process and save
recon = (recon + 1) / 2  # [-1, 1] -> [0, 1]
recon = recon.clamp(0, 1)
recon = recon[0].permute(1, 2, 0).cpu().numpy()
recon = (recon * 255).astype('uint8')

output_img = Image.fromarray(recon)
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
output_img.save(output_path)
print(f'Saved reconstruction to: {output_path}')

# Also save side-by-side comparison
orig_resized = img.resize((image_size, image_size))
comparison = Image.new('RGB', (image_size * 2, image_size))
comparison.paste(orig_resized, (0, 0))
comparison.paste(output_img, (image_size, 0))
comparison_path = output_path.replace('.png', '_comparison.png')
comparison.save(comparison_path)
print(f'Saved comparison to: {comparison_path}')
"
