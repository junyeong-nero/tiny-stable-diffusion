"""Streamlit demo app for tiny-stable-diffusion.

Run with:
    uv run streamlit run src/demo/app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image

from src.config import get_config
from src.models.diffusion import Diffusion
from src.models.animated_diffusion import AnimatedDiffusion
from src.models.factory import DiT
from src.models.vae import create_vae
from src.models.animated_mmdit import load_animated_mmdit
from src.text_encoder.clip_encoder import CLIPTextEncoder
from src.training.checkpoint import find_latest_checkpoint
from src.utils.common import get_device


def get_vae_checkpoint_path() -> Path | None:
    """Find available VAE checkpoint."""
    checkpoint = find_latest_checkpoint("checkpoints", prefix="vae")
    if checkpoint is None:
        default_path = Path("checkpoints/vae.pt")
        if default_path.exists():
            return default_path
        return None
    return Path(checkpoint)


def get_diffusion_checkpoint_path() -> Path | None:
    """Find available diffusion checkpoint."""
    checkpoint = find_latest_checkpoint("checkpoints", prefix="diffusion")
    if checkpoint is None:
        default_path = Path("checkpoints/diffusion.pt")
        if default_path.exists():
            return default_path
        return None
    return Path(checkpoint)


def get_motion_checkpoint_path() -> Path | None:
    """Find available motion checkpoint."""
    checkpoint = find_latest_checkpoint("checkpoints", prefix="motion")
    if checkpoint is None:
        default_path = Path("checkpoints/motion.pt")
        if default_path.exists():
            return default_path
        return None
    return Path(checkpoint)


@st.cache_resource
def load_vae(checkpoint_path: str):
    """Load VAE model (cached)."""
    device = get_device("auto")
    config = get_config("vae_train")
    image_size = config.get("image_size", 64)

    vae = create_vae(
        image_size=image_size,
        z_channels=config.get("latent_channels", 16),
        ch=config.get("vae_ch", 64),
        ch_mult=tuple(config.get("vae_ch_mult", [1, 2, 4, 4])),
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vae.load_state_dict(ckpt["model_state_dict"])
    vae = vae.to(device)
    vae.eval()

    return vae, image_size, device


@st.cache_resource
def load_diffusion_models(checkpoint_path: str):
    """Load diffusion model, VAE decoder, and CLIP encoder (cached)."""
    device = get_device("auto")

    # Load CLIP encoder
    clip_encoder = CLIPTextEncoder()
    clip_encoder = clip_encoder.to(device)
    clip_encoder.eval()

    # Compute unconditional embedding
    with torch.no_grad():
        uncond_embed = clip_encoder.encode([""])
    uncond_embed = uncond_embed.to(device)

    # Load diffusion checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = ckpt.get("model_config", ckpt.get("config", {}))

    model_size = model_config.get("model_size", "S")
    patch_size = model_config.get("patch_size", 2)
    model_type = model_config.get("model_type", "dit")
    qk_rmsnorm = model_config.get("qk_rmsnorm", True)
    register_tokens = model_config.get("register_tokens", 0)
    latent_size = model_config.get("latent_size", 8)
    in_channels = model_config.get("in_channels", 16)
    image_size = model_config.get("image_size", 64)

    # Load VAE for decoding
    vae_checkpoint = model_config.get("vae_checkpoint", "checkpoints/vae.pt")
    if not Path(vae_checkpoint).exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_checkpoint}")

    vae = create_vae(
        image_size=image_size,
        z_channels=in_channels,
        ch=model_config.get("vae_ch", 64),
        ch_mult=tuple(model_config.get("vae_ch_mult", [1, 2, 4, 4])),
    )
    vae_state = torch.load(vae_checkpoint, map_location=device, weights_only=False)
    vae.load_state_dict(vae_state["model_state_dict"])
    vae = vae.to(device)
    vae.eval()

    # Load DiT model
    model = DiT(
        in_channels=in_channels,
        image_size=latent_size,
        patch_size=patch_size,
        model_size=model_size,
        clip_embed_dim=clip_encoder.embedding_dim,
        model_type=model_type,
        qk_rmsnorm=qk_rmsnorm,
        register_tokens=register_tokens,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    return {
        "model": model,
        "vae": vae,
        "clip_encoder": clip_encoder,
        "uncond_embed": uncond_embed,
        "device": device,
        "latent_size": latent_size,
        "in_channels": in_channels,
        "image_size": image_size,
    }


@st.cache_resource
def load_animation_models(
    diffusion_checkpoint_path: str,
    motion_checkpoint_path: str | None,
):
    """Load AnimatedMMDiT model for GIF generation (cached)."""
    device = get_device("auto")

    # Load CLIP encoder
    clip_encoder = CLIPTextEncoder()
    clip_encoder = clip_encoder.to(device)
    clip_encoder.eval()

    # Compute unconditional embedding
    with torch.no_grad():
        uncond_embed = clip_encoder.encode([""])
    uncond_embed = uncond_embed.to(device)

    # Load VAE for decoding
    diffusion_ckpt = torch.load(
        diffusion_checkpoint_path, map_location=device, weights_only=False
    )
    model_config = diffusion_ckpt.get("model_config", diffusion_ckpt.get("config", {}))
    vae_checkpoint = model_config.get("vae_checkpoint", "checkpoints/vae.pt")
    in_channels = model_config.get("in_channels", 16)
    image_size = model_config.get("image_size", 64)
    latent_size = model_config.get("latent_size", 8)

    if not Path(vae_checkpoint).exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_checkpoint}")

    vae = create_vae(
        image_size=image_size,
        z_channels=in_channels,
        ch=model_config.get("vae_ch", 64),
        ch_mult=tuple(model_config.get("vae_ch_mult", [1, 2, 4, 4])),
    )
    vae_state = torch.load(vae_checkpoint, map_location=device, weights_only=False)
    vae.load_state_dict(vae_state["model_state_dict"])
    vae = vae.to(device)
    vae.eval()

    # Load AnimatedMMDiT
    model = load_animated_mmdit(
        base_checkpoint_path=diffusion_checkpoint_path,
        motion_checkpoint_path=motion_checkpoint_path,
        device=device,
        in_channels=in_channels,
        image_size=latent_size,
        num_frames=16,
        motion_num_layers=2,
        motion_num_heads=8,
        freeze_base=True,
    )
    model = model.to(device)
    model.eval()

    return {
        "model": model,
        "vae": vae,
        "clip_encoder": clip_encoder,
        "uncond_embed": uncond_embed,
        "device": device,
        "latent_size": latent_size,
        "in_channels": in_channels,
        "image_size": image_size,
    }


def vae_reconstruction_page():
    """VAE reconstruction demo page."""
    st.header("VAE Reconstruction")
    st.write("Upload an image to see how the VAE encodes and reconstructs it.")

    # Check for VAE checkpoint
    vae_path = get_vae_checkpoint_path()
    if vae_path is None:
        st.error("VAE checkpoint not found. Please train a VAE first:")
        st.code("uv run main.py --train-vae --epochs 100")
        return

    st.success(f"VAE checkpoint: `{vae_path}`")

    # Load VAE
    try:
        vae, image_size, device = load_vae(str(vae_path))
    except Exception as e:
        st.error(f"Failed to load VAE: {e}")
        return

    st.info(f"Device: **{device}** | Image size: **{image_size}x{image_size}**")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "webp"],
        help="Upload an image to reconstruct through the VAE",
    )

    if uploaded_file is not None:
        # Load image
        original_img = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original")
            st.image(original_img, use_container_width=True)

        # Reconstruct
        with st.spinner("Reconstructing..."):
            transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
            x = transform(original_img).unsqueeze(0).to(device)

            with torch.no_grad():
                mean, logvar = vae.encode(x)
                recon = vae.decode(mean)

            # Post-process
            recon = (recon + 1) / 2
            recon = recon.clamp(0, 1)
            recon = recon[0].permute(1, 2, 0).cpu().numpy()
            recon = (recon * 255).astype("uint8")
            recon_img = Image.fromarray(recon)

        with col2:
            st.subheader("Reconstructed")
            st.image(recon_img, use_container_width=True)

        # Show latent space info
        with st.expander("Latent Space Info"):
            st.write(f"Latent shape: `{mean.shape}`")
            st.write(f"Latent mean: `{mean.mean().item():.4f}`")
            st.write(f"Latent std: `{mean.std().item():.4f}`")


def diffusion_generation_page():
    """Diffusion generation demo page."""
    st.header("Text-to-Image Generation")
    st.write("Generate images from text prompts using the trained diffusion model.")

    # Check for diffusion checkpoint
    diffusion_path = get_diffusion_checkpoint_path()
    if diffusion_path is None:
        st.error("Diffusion checkpoint not found. Please train a diffusion model first:")
        st.code("uv run main.py --train-diffusion --epochs 200")
        return

    st.success(f"Diffusion checkpoint: `{diffusion_path}`")

    # Load models
    try:
        models = load_diffusion_models(str(diffusion_path))
    except FileNotFoundError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return

    st.info(
        f"Device: **{models['device']}** | "
        f"Image size: **{models['image_size']}x{models['image_size']}**"
    )

    # Input controls
    prompt = st.text_input(
        "Prompt",
        value="a photo of a cat",
        help="Enter a text description of the image you want to generate",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        num_steps = st.slider("Sampling Steps", 10, 100, 50, help="More steps = better quality")
    with col2:
        guidance_scale = st.slider(
            "Guidance Scale", 1.0, 20.0, 7.5, help="Higher = more faithful to prompt"
        )
    with col3:
        seed = st.number_input("Seed", value=42, min_value=0, help="Random seed for reproducibility")

    # Generate button
    if st.button("Generate", type="primary", use_container_width=True):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
            return

        with st.spinner(f"Generating with {num_steps} steps..."):
            # Set seed
            torch.manual_seed(seed)

            # Initialize diffusion
            diffusion = Diffusion(
                num_timesteps=1000,
                beta_schedule="cosine",
                guidance_scale=guidance_scale,
                uncond_embed=models["uncond_embed"],
            )

            # Encode prompt
            text_embeds = models["clip_encoder"].encode([prompt])
            text_embeds = text_embeds.to(models["device"])

            # Generate
            with torch.no_grad():
                images = diffusion.sample(
                    model=models["model"],
                    shape=(1, models["in_channels"], models["latent_size"], models["latent_size"]),
                    text_embeds=text_embeds,
                    num_steps=num_steps,
                    use_ddim=True,
                    use_cfg=True,
                    vae_decoder=models["vae"],
                )

            # Convert to PIL
            img = images[0]
            img = img.permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            pil_img = Image.fromarray(img)

        # Display result
        st.subheader("Generated Image")
        st.image(pil_img, use_container_width=True)

        # Download button
        from io import BytesIO

        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        st.download_button(
            label="Download Image",
            data=buf.getvalue(),
            file_name=f"generated_{prompt[:20].replace(' ', '_')}.png",
            mime="image/png",
        )


def gif_generation_page():
    """GIF generation demo page."""
    st.header("GIF Generation")
    st.write("Generate animated GIFs from text prompts using the Motion Module.")

    # Check for diffusion checkpoint
    diffusion_path = get_diffusion_checkpoint_path()
    if diffusion_path is None:
        st.error("Diffusion checkpoint not found. Please train a diffusion model first:")
        st.code("uv run main.py --train-diffusion --epochs 200")
        return

    # Check for motion checkpoint (optional)
    motion_path = get_motion_checkpoint_path()
    if motion_path is None:
        st.warning(
            "Motion checkpoint not found. Using untrained motion module. "
            "For better results, train with:\n"
            "`uv run main.py --train-motion --epochs 50`"
        )

    st.success(f"Diffusion checkpoint: `{diffusion_path}`")
    if motion_path:
        st.success(f"Motion checkpoint: `{motion_path}`")

    # Load models
    try:
        models = load_animation_models(
            str(diffusion_path),
            str(motion_path) if motion_path else None,
        )
    except FileNotFoundError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return

    st.info(
        f"Device: **{models['device']}** | "
        f"Image size: **{models['image_size']}x{models['image_size']}**"
    )

    # Input controls
    prompt = st.text_input(
        "Prompt",
        value="a cat walking",
        help="Enter a text description of the animation you want to generate",
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        num_frames = st.slider("Frames", 4, 32, 16, help="Number of animation frames")
    with col2:
        num_steps = st.slider("Steps", 10, 100, 50, help="Diffusion steps")
    with col3:
        guidance_scale = st.slider("Guidance", 1.0, 20.0, 7.5, help="CFG scale")
    with col4:
        fps = st.slider("FPS", 4, 24, 8, help="Frames per second")

    seed = st.number_input("Seed", value=42, min_value=0, help="Random seed")

    # Generate button
    if st.button("Generate GIF", type="primary", use_container_width=True):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
            return

        with st.spinner(f"Generating {num_frames} frames with {num_steps} steps..."):
            # Set seed
            torch.manual_seed(seed)

            # Initialize diffusion
            diffusion = AnimatedDiffusion(
                num_timesteps=1000,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                uncond_embed=models["uncond_embed"],
            )

            # Encode prompt
            text_embeds = models["clip_encoder"].encode([prompt])
            text_embeds = text_embeds.to(models["device"])

            # Generate video
            with torch.no_grad():
                video = diffusion.sample_video(
                    model=models["model"],
                    batch_size=1,
                    num_frames=num_frames,
                    latent_channels=models["in_channels"],
                    latent_size=models["latent_size"],
                    text_embeds=text_embeds,
                    num_steps=num_steps,
                    use_cfg=True,
                    vae_decoder=models["vae"],
                    device=models["device"],
                )

            # Convert to PIL Images
            video = video[0]  # (F, 3, H, W)
            frames = []
            for i in range(video.shape[0]):
                frame = video[i]  # (3, H, W)
                frame = frame.permute(1, 2, 0).mul(255).clamp(0, 255)
                frame = frame.to(torch.uint8).cpu().numpy()
                frames.append(Image.fromarray(frame))

        # Display result
        st.subheader("Generated GIF")

        # Create GIF in memory
        from io import BytesIO

        gif_buffer = BytesIO()
        duration = int(1000 / fps)  # milliseconds per frame

        frames[0].save(
            gif_buffer,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
            optimize=True,
        )
        gif_buffer.seek(0)

        # Display GIF
        st.image(gif_buffer, use_container_width=True)

        # Download button
        gif_buffer.seek(0)
        st.download_button(
            label="Download GIF",
            data=gif_buffer.getvalue(),
            file_name=f"generated_{prompt[:20].replace(' ', '_')}.gif",
            mime="image/gif",
        )

        # Show individual frames in expander
        with st.expander("View Individual Frames"):
            cols = st.columns(min(4, num_frames))
            for i, frame in enumerate(frames):
                cols[i % 4].image(frame, caption=f"Frame {i + 1}")


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="tiny-stable-diffusion Demo",
        page_icon="🎨",
        layout="wide",
    )

    st.title("tiny-stable-diffusion Demo")
    st.markdown("An educational implementation of Stable Diffusion 3 from scratch.")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Demo",
        ["VAE Reconstruction", "Text-to-Image Generation", "GIF Generation"],
        help="Choose which model to demo",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Quick Start:**
    ```bash
    # Train VAE
    uv run main.py --train-vae

    # Train Diffusion
    uv run main.py --train-diffusion

    # Train Motion (for GIF)
    uv run main.py --train-motion
    ```
    """)

    # Route to selected page
    if page == "VAE Reconstruction":
        vae_reconstruction_page()
    elif page == "Text-to-Image Generation":
        diffusion_generation_page()
    else:
        gif_generation_page()


if __name__ == "__main__":
    main()
