"""Animation/GIF generation utilities for tiny-stable-diffusion.

Implements video generation using AnimatedMMDiT with Motion Modules:
1. Sample noise in latent space with temporal dimension
2. Denoise using AnimatedMMDiT
3. Decode each frame using VAE decoder
4. Save as GIF or video
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from PIL import Image

from src.data.video_transforms import video_to_gif, denormalize_video
from src.models.animated_diffusion import AnimatedDiffusion
from src.models.animated_mmdit import load_animated_mmdit
from src.models.vae import create_vae
from src.text_encoder.clip_encoder import CLIPTextEncoder
from src.training.checkpoint import find_latest_checkpoint
from src.utils.common import get_device, set_seed


class AnimationGenerator:
    """Generator for creating GIFs and video animations from text prompts.

    Uses pretrained VAE, base DiT/MMDiT, and Motion Module to generate
    temporally coherent video sequences.
    """

    def __init__(
        self,
        vae_checkpoint: str | Path,
        diffusion_checkpoint: str | Path,
        motion_checkpoint: str | Path | None = None,
        device: str = "auto",
        num_frames: int = 16,
        motion_num_layers: int = 2,
        motion_num_heads: int = 8,
    ) -> None:
        """Initialize AnimationGenerator.

        Args:
            vae_checkpoint: Path to VAE checkpoint
            diffusion_checkpoint: Path to base diffusion model checkpoint
            motion_checkpoint: Path to motion module checkpoint (optional)
            device: Device to use
            num_frames: Default number of frames
            motion_num_layers: Motion module layers
            motion_num_heads: Motion module attention heads
        """
        self.device = get_device(device)
        self.num_frames = num_frames

        print(f"Initializing AnimationGenerator on {self.device}")

        # Load CLIP encoder
        print("Loading CLIP text encoder...")
        self.clip_encoder = CLIPTextEncoder()
        self.clip_encoder = self.clip_encoder.to(self.device)
        self.clip_encoder.eval()

        # Compute unconditional embedding
        with torch.no_grad():
            self.uncond_embed = self.clip_encoder.encode([""])
        self.uncond_embed = self.uncond_embed.to(self.device)

        # Load VAE
        vae_checkpoint = Path(vae_checkpoint)
        if not vae_checkpoint.exists():
            raise FileNotFoundError(f"VAE checkpoint not found: {vae_checkpoint}")

        print(f"Loading VAE from {vae_checkpoint}...")
        vae_state = torch.load(vae_checkpoint, map_location=self.device)

        self.vae = create_vae(
            image_size=64,
            z_channels=16,
        )
        self.vae.load_state_dict(vae_state["model_state_dict"])
        self.vae = self.vae.to(self.device)
        self.vae.eval()

        # Set scaling factor
        if "scaling_factor" in vae_state:
            self.vae.set_scaling_factor(vae_state["scaling_factor"])
        print(f"VAE loaded (scaling_factor={self.vae.scaling_factor:.4f})")

        # Load AnimatedMMDiT
        diffusion_checkpoint = Path(diffusion_checkpoint)
        if not diffusion_checkpoint.exists():
            raise FileNotFoundError(f"Diffusion checkpoint not found: {diffusion_checkpoint}")

        print(f"Loading AnimatedMMDiT from {diffusion_checkpoint}...")
        self.model = load_animated_mmdit(
            base_checkpoint_path=str(diffusion_checkpoint),
            motion_checkpoint_path=str(motion_checkpoint) if motion_checkpoint else None,
            device=self.device,
            in_channels=16,
            image_size=8,
            num_frames=num_frames,
            motion_num_layers=motion_num_layers,
            motion_num_heads=motion_num_heads,
            freeze_base=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        param_counts = self.model.parameters_count()
        print(f"Model loaded:")
        print(f"  Base: {param_counts['base_total'] / 1e6:.2f}M params")
        print(f"  Motion: {param_counts['motion_trainable'] / 1e6:.2f}M params")

        # Initialize diffusion
        self.diffusion = AnimatedDiffusion(
            num_timesteps=1000,
            num_frames=num_frames,
            guidance_scale=7.5,
            uncond_embed=self.uncond_embed,
        )

        # Cache model config
        self.latent_size = 8
        self.latent_channels = 16
        self.image_size = 64

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        num_frames: int | None = None,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int | None = None,
        fps: int = 8,
    ) -> list[Image.Image]:
        """Generate animation frames from text prompt.

        Args:
            prompt: Text description of the animation
            num_frames: Number of frames to generate
            num_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility
            fps: Frames per second (for metadata)

        Returns:
            List of PIL Images (frames)
        """
        if seed is not None:
            set_seed(seed)

        num_frames = num_frames or self.num_frames

        print(f"Generating {num_frames} frames for: '{prompt}'")

        # Encode prompt
        text_embeds = self.clip_encoder.encode([prompt])
        text_embeds = text_embeds.to(self.device)

        # Set guidance scale
        original_scale = self.diffusion.guidance_scale
        self.diffusion.guidance_scale = guidance_scale

        # Sample video
        video = self.diffusion.sample_video(
            model=self.model,
            batch_size=1,
            num_frames=num_frames,
            latent_channels=self.latent_channels,
            latent_size=self.latent_size,
            text_embeds=text_embeds,
            num_steps=num_steps,
            use_cfg=True,
            vae_decoder=self.vae,
            device=self.device,
        )

        self.diffusion.guidance_scale = original_scale

        # Convert to PIL Images
        # video: (1, F, 3, H, W) in [0, 1]
        video = video[0]  # (F, 3, H, W)
        frames = []

        for i in range(video.shape[0]):
            frame = video[i]  # (3, H, W)
            frame = frame.permute(1, 2, 0).mul(255).clamp(0, 255)
            frame = frame.to(torch.uint8).cpu().numpy()
            frames.append(Image.fromarray(frame))

        print(f"Generated {len(frames)} frames")
        return frames

    def save_gif(
        self,
        frames: list[Image.Image],
        output_path: str | Path,
        fps: int = 8,
        loop: int = 0,
        optimize: bool = True,
    ) -> Path:
        """Save frames as GIF.

        Args:
            frames: List of PIL Images
            output_path: Output file path
            fps: Frames per second
            loop: Number of loops (0 = infinite)
            optimize: Optimize GIF file size

        Returns:
            Path to saved GIF
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        duration = int(1000 / fps)  # milliseconds per frame

        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=loop,
            optimize=optimize,
        )

        print(f"Saved GIF: {output_path} ({len(frames)} frames, {fps} fps)")
        return output_path

    def generate_and_save(
        self,
        prompt: str,
        output_path: str | Path,
        num_frames: int | None = None,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int | None = None,
        fps: int = 8,
    ) -> Path:
        """Generate animation and save as GIF.

        Args:
            prompt: Text description
            output_path: Output file path
            num_frames: Number of frames
            num_steps: Diffusion steps
            guidance_scale: CFG scale
            seed: Random seed
            fps: Frames per second

        Returns:
            Path to saved GIF
        """
        frames = self.generate(
            prompt=prompt,
            num_frames=num_frames,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            fps=fps,
        )

        return self.save_gif(frames, output_path, fps=fps)


@torch.no_grad()
def generate_animation(
    prompts: list[str],
    vae_checkpoint: str | Path | None = None,
    diffusion_checkpoint: str | Path | None = None,
    motion_checkpoint: str | Path | None = None,
    num_frames: int = 16,
    num_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int | None = None,
    device: str = "auto",
    fps: int = 8,
) -> list[list[Image.Image]]:
    """Generate animations from text prompts.

    Convenience function that creates AnimationGenerator internally.

    Args:
        prompts: List of text prompts
        vae_checkpoint: Path to VAE checkpoint
        diffusion_checkpoint: Path to diffusion checkpoint
        motion_checkpoint: Path to motion module checkpoint
        num_frames: Frames per animation
        num_steps: Diffusion steps
        guidance_scale: CFG scale
        seed: Random seed
        device: Device to use
        fps: Frames per second

    Returns:
        List of frame lists (one per prompt)
    """
    if seed is not None:
        set_seed(seed)

    device = get_device(device)

    # Find checkpoints
    if vae_checkpoint is None:
        vae_checkpoint = Path("checkpoints/vae.pt")
    if diffusion_checkpoint is None:
        diffusion_checkpoint = find_latest_checkpoint("checkpoints", prefix="diffusion")
        if diffusion_checkpoint is None:
            diffusion_checkpoint = Path("checkpoints/diffusion.pt")
    if motion_checkpoint is None:
        motion_checkpoint = find_latest_checkpoint("checkpoints", prefix="motion")

    # Create generator
    generator = AnimationGenerator(
        vae_checkpoint=vae_checkpoint,
        diffusion_checkpoint=diffusion_checkpoint,
        motion_checkpoint=motion_checkpoint,
        device=device,
        num_frames=num_frames,
    )

    # Generate for each prompt
    all_frames = []
    for i, prompt in enumerate(prompts):
        print(f"\n[{i + 1}/{len(prompts)}] Generating: '{prompt}'")
        frames = generator.generate(
            prompt=prompt,
            num_frames=num_frames,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed + i if seed is not None else None,
            fps=fps,
        )
        all_frames.append(frames)

    return all_frames


def animation_demo(
    vae_checkpoint: str | Path | None = None,
    diffusion_checkpoint: str | Path | None = None,
    motion_checkpoint: str | Path | None = None,
) -> None:
    """Interactive animation generation demo.

    Args:
        vae_checkpoint: Path to VAE checkpoint
        diffusion_checkpoint: Path to diffusion checkpoint
        motion_checkpoint: Path to motion module checkpoint
    """
    print("=" * 60)
    print("tiny-stable-diffusion Animation Demo")
    print("=" * 60)
    print("\nEnter prompts to generate GIFs. Type 'quit' to exit.\n")

    # Find checkpoints
    if vae_checkpoint is None:
        vae_checkpoint = Path("checkpoints/vae.pt")
    if diffusion_checkpoint is None:
        diffusion_checkpoint = find_latest_checkpoint("checkpoints", prefix="diffusion")
        if diffusion_checkpoint is None:
            diffusion_checkpoint = Path("checkpoints/diffusion.pt")
    if motion_checkpoint is None:
        motion_checkpoint = find_latest_checkpoint("checkpoints", prefix="motion")

    # Check checkpoints exist
    if not Path(vae_checkpoint).exists():
        print(f"Error: VAE checkpoint not found: {vae_checkpoint}")
        return
    if not Path(diffusion_checkpoint).exists():
        print(f"Error: Diffusion checkpoint not found: {diffusion_checkpoint}")
        return

    if motion_checkpoint is None or not Path(motion_checkpoint).exists():
        print("Warning: Motion checkpoint not found. Using untrained motion module.")
        motion_checkpoint = None

    # Create generator
    try:
        generator = AnimationGenerator(
            vae_checkpoint=vae_checkpoint,
            diffusion_checkpoint=diffusion_checkpoint,
            motion_checkpoint=motion_checkpoint,
            num_frames=16,
        )
    except Exception as e:
        print(f"Error initializing generator: {e}")
        return

    print("\nReady! Generate 64x64 GIFs with 16 frames.")

    output_dir = Path("demo_gifs")
    output_dir.mkdir(exist_ok=True)

    count = 0
    while True:
        try:
            prompt = input("\nEnter prompt (or 'quit'): ").strip()
            if prompt.lower() in ["quit", "exit", "q"]:
                break
            if not prompt:
                continue

            count += 1
            safe_prompt = prompt.replace(" ", "_")[:20]
            output_path = output_dir / f"{count:03d}_{safe_prompt}.gif"

            generator.generate_and_save(
                prompt=prompt,
                output_path=output_path,
                num_frames=16,
                num_steps=50,
                guidance_scale=7.5,
                fps=8,
            )

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nGoodbye!")
