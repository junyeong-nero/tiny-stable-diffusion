"""Inference script for generating emoji images from text prompts."""

from __future__ import annotations

from pathlib import Path

import torch

from src.config import InferenceConfig, get_parser
from src.models.diffusion import Diffusion
from src.models.dit import DiT
from src.text_encoder.clip_encoder import CLIPTextEncoder


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    """Main inference function."""
    parser = get_parser()
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="File containing prompts (one per line)",
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Comma-separated prompts for batch generation",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        default=False,
        help="Create grid visualization for batch generation",
    )
    args = parser.parse_args()

    prompts = []
    prompt = getattr(args, "prompt", None)
    prompts_file = getattr(args, "prompts_file", None)
    batch_prompts = getattr(args, "batch", None)

    if prompts_file:
        prompts_path = Path(prompts_file)
        if not prompts_path.exists():
            print(f"Error: Prompts file not found: {prompts_path}")
            return
        prompts = [
            line.strip()
            for line in prompts_path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        print(f"Loaded {len(prompts)} prompts from {prompts_path}")
    elif batch_prompts:
        prompts = [p.strip() for p in batch_prompts.split(",") if p.strip()]
        print(f"Batch mode: {len(prompts)} prompts")
    elif prompt:
        prompts = [prompt]
    else:
        print("Error: One of --prompt, --batch, or --prompts-file is required")
        return

    seed = getattr(args, "seed", 42) or 42
    set_seed(seed)

    num_samples_per_prompt = getattr(args, "num_samples", 1) or 1
    if len(prompts) > 1:
        num_samples_per_prompt = 1
        print("Batch mode: generating 1 sample per prompt")

    inference_config = InferenceConfig(
        checkpoint=getattr(args, "checkpoint", None),
        num_samples=num_samples_per_prompt,
        num_steps=getattr(args, "steps", 50) or 50,
        guidance_scale=getattr(args, "guidance_scale", 7.5) or 7.5,
        seed=seed,
    )

    # Get device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Load CLIP encoder
    print("Loading CLIP text encoder...")
    clip_encoder = CLIPTextEncoder()
    clip_encoder = clip_encoder.to(device)
    clip_encoder.eval()

    all_generated_images = []
    all_prompts_flat = []

    # Initialize diffusion
    diffusion = Diffusion(
        num_timesteps=1000,
        beta_schedule="cosine",
        guidance_scale=inference_config.guidance_scale,
    )

    # Load model
    checkpoint_path = inference_config.checkpoint
    if checkpoint_path is None:
        # Look for latest checkpoint
        checkpoints_dir = Path("checkpoints")
        checkpoints = list(checkpoints_dir.glob("*.pt"))
        if checkpoints:
            checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
            print(f"Using latest checkpoint: {checkpoint_path}")
        else:
            print("Error: No checkpoint found!")
            print("Please train the model first or provide --checkpoint path")
            return
    else:
        checkpoint_path = Path(checkpoint_path)

    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model config from checkpoint
    model_config = checkpoint.get("model_config", {})
    model_size = model_config.get("model_size", "S")
    patch_size = model_config.get("patch_size", 2)
    image_size = model_config.get("image_size", 32)

    # Initialize model
    model = DiT(
        in_channels=3,
        image_size=image_size,
        patch_size=patch_size,
        model_size=model_size,
        clip_embed_dim=clip_encoder.embedding_dim,
    )
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Print model info
    model_info = model.get_model_size_info()
    print(f"Model: DiT-{model_size} with {model_info['num_parameters']:,} parameters")

    for prompt_text in prompts:
        print(f"Encoding prompt: '{prompt_text}'")
        text_embeds, text_mask = clip_encoder.encode([prompt_text] * inference_config.num_samples)
        text_embeds = text_embeds.to(device)
        text_mask = text_mask.to(device)

        print(f"Generating {inference_config.num_samples} image(s) for {prompt_text}...")
        shape = (
            inference_config.num_samples,
            3,
            image_size,
            image_size,
        )

        with torch.no_grad():
            images = diffusion.sample(
                model=model,
                shape=shape,
                text_embeds=text_embeds,
                text_mask=text_mask,
                num_steps=inference_config.num_steps,
                use_ddim=getattr(args, "ddim", True),
                use_cfg=True,
                seed=seed,
            )

        all_generated_images.extend(images)
        all_prompts_flat.extend([prompt_text] * len(images))

    output_dir = Path(inference_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    saved_images = []

    for i, (img_tensor, prompt_text) in enumerate(
        zip(all_generated_images, all_prompts_flat, strict=True)
    ):
        img = (img_tensor * 255).clamp(0, 255).permute(1, 2, 0).to(torch.uint8)
        img = Image.fromarray(img.cpu().numpy())

        safe_prompt = prompt_text.replace(" ", "_")[:20]
        original_path = output_dir / f"{i:04d}_{safe_prompt}.png"
        img.save(original_path)
        print(f"Saved: {original_path}")

        upscale_size = inference_config.upscale_size
        img_upscaled = img.resize((upscale_size, upscale_size), Image.NEAREST)
        upscaled_path = output_dir / f"{i:04d}_{safe_prompt}_upscaled.png"
        img_upscaled.save(upscaled_path)
        print(f"Saved upscaled: {upscaled_path}")

        saved_images.append(img_upscaled)

    if len(saved_images) > 1 and getattr(args, "grid", False):
        print("Creating grid visualization...")
        cols = min(4, len(saved_images))
        rows = (len(saved_images) + cols - 1) // cols
        img_w, img_h = saved_images[0].size

        grid = Image.new("RGB", (cols * img_w, rows * img_h), (255, 255, 255))

        for idx, img in enumerate(saved_images):
            row = idx // cols
            col = idx % cols
            grid.paste(img, (col * img_w, row * img_h))

        grid_path = output_dir / "grid.png"
        grid.save(grid_path)
        print(f"Saved grid: {grid_path}")

    print(f"Done! Generated {len(all_generated_images)} image(s)")


if __name__ == "__main__":
    main()
