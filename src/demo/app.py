"""Streamlit demo app for tiny-stable-diffusion.

Run with:
    uv run streamlit run src/demo/app.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path

import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image

from src.config import get_config
from src.models.animated_diffusion import AnimatedDiffusion
from src.models.animated_mmdit import load_animated_mmdit
from src.models.diffusion import Diffusion
from src.models.factory import DiT
from src.models.vae import create_vae
from src.text_encoder.clip_encoder import CLIPTextEncoder
from src.training.checkpoint import find_latest_checkpoint
from src.utils.common import get_device

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = Path("checkpoints")

_CUSTOM_CSS = """
<style>
div[data-testid="stMetric"] {
    background-color: #f0f2f6;
    border-radius: 8px;
    padding: 12px 16px;
}
div[data-testid="stMetric"] > div {
    overflow: hidden;
}
</style>
"""


# ---------------------------------------------------------------------------
# Checkpoint scanning utilities
# ---------------------------------------------------------------------------

def _format_bytes(size: int) -> str:
    """Format byte size into a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _format_mtime(mtime: float) -> str:
    """Format modification timestamp to a readable date string."""
    return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")


def scan_checkpoints(
    checkpoint_dir: Path, prefix: str
) -> list[dict[str, str | int | float]]:
    """Scan checkpoint directory for files matching *prefix*.

    Returns file-level metadata only (no torch.load) sorted by mtime desc.
    """
    if not checkpoint_dir.exists():
        return []

    results = []
    for p in checkpoint_dir.glob(f"{prefix}*.pt"):
        stat = p.stat()
        results.append({
            "path": str(p),
            "name": p.name,
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "display": f"{p.name}  ({_format_bytes(stat.st_size)}, {_format_mtime(stat.st_mtime)})",
        })

    return sorted(results, key=lambda d: d["mtime"], reverse=True)


# ---------------------------------------------------------------------------
# Sidebar: checkpoint selectors
# ---------------------------------------------------------------------------

def _checkpoint_selectbox(
    label: str,
    prefix: str,
    session_key: str,
) -> Path | None:
    """Render a selectbox for checkpoint files and persist selection."""
    candidates = scan_checkpoints(CHECKPOINT_DIR, prefix)
    if not candidates:
        st.sidebar.warning(f"No {prefix} checkpoints found.")
        return None

    options = [c["display"] for c in candidates]
    current = st.session_state.get(session_key)

    # Resolve index from session state
    idx = 0
    if current:
        for i, c in enumerate(candidates):
            if c["path"] == current:
                idx = i
                break

    selection = st.sidebar.selectbox(label, options, index=idx, key=f"_sel_{session_key}")
    selected_idx = options.index(selection)
    selected_path = candidates[selected_idx]["path"]
    st.session_state[session_key] = selected_path

    return Path(selected_path)


def render_sidebar_checkpoints() -> None:
    """Render checkpoint selection widgets in the sidebar."""
    st.sidebar.markdown("### Checkpoints")

    _checkpoint_selectbox("VAE Checkpoint", "vae", "vae_checkpoint")
    _checkpoint_selectbox("Diffusion Checkpoint", "diffusion", "diffusion_checkpoint")
    _checkpoint_selectbox("Motion Checkpoint", "motion", "motion_checkpoint")

    st.sidebar.divider()


def render_sidebar_system_info() -> None:
    """Display device / system info badge in the sidebar."""
    device = get_device("auto")
    device_label = str(device).upper()
    if "cuda" in str(device):
        gpu_name = torch.cuda.get_device_name(0)
        device_label = f"CUDA ({gpu_name})"
    elif "mps" in str(device):
        device_label = "Apple MPS"

    st.sidebar.markdown("### System")
    st.sidebar.info(f"**Device:** {device_label}  \n**PyTorch:** {torch.__version__}")
    st.sidebar.divider()


# ---------------------------------------------------------------------------
# Checkpoint path helpers (read from session_state, fallback to auto-detect)
# ---------------------------------------------------------------------------

def get_vae_checkpoint_path() -> Path | None:
    """Return user-selected VAE checkpoint path or auto-detect."""
    selected = st.session_state.get("vae_checkpoint")
    if selected and Path(selected).exists():
        return Path(selected)

    checkpoint = find_latest_checkpoint("checkpoints", prefix="vae")
    if checkpoint is not None:
        return Path(checkpoint)

    default_path = Path("checkpoints/vae.pt")
    return default_path if default_path.exists() else None


def get_diffusion_checkpoint_path() -> Path | None:
    """Return user-selected diffusion checkpoint path or auto-detect."""
    selected = st.session_state.get("diffusion_checkpoint")
    if selected and Path(selected).exists():
        return Path(selected)

    checkpoint = find_latest_checkpoint("checkpoints", prefix="diffusion")
    if checkpoint is not None:
        return Path(checkpoint)

    default_path = Path("checkpoints/diffusion.pt")
    return default_path if default_path.exists() else None


def get_motion_checkpoint_path() -> Path | None:
    """Return user-selected motion checkpoint path or auto-detect."""
    selected = st.session_state.get("motion_checkpoint")
    if selected and Path(selected).exists():
        return Path(selected)

    checkpoint = find_latest_checkpoint("checkpoints", prefix="motion")
    if checkpoint is not None:
        return Path(checkpoint)

    default_path = Path("checkpoints/motion.pt")
    return default_path if default_path.exists() else None


# ---------------------------------------------------------------------------
# Checkpoint info display
# ---------------------------------------------------------------------------

def show_checkpoint_info(ckpt: dict, label: str = "Checkpoint Info") -> None:
    """Display model config from a loaded checkpoint in an expander."""
    model_config = ckpt.get("model_config", ckpt.get("config", {}))
    if not model_config:
        return

    with st.expander(f"**{label}**", expanded=False):
        cols = st.columns(4)
        cols[0].metric("Model Type", model_config.get("model_type", "n/a"))
        cols[1].metric("Model Size", model_config.get("model_size", "n/a"))
        cols[2].metric("Epoch", model_config.get("epoch", "n/a"))
        cols[3].metric(
            "Loss",
            f"{model_config.get('loss', 'n/a'):.4f}"
            if isinstance(model_config.get("loss"), (int, float))
            else "n/a",
        )

        st.code(json.dumps(model_config, indent=2, default=str), language="json")


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------

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

    return vae, image_size, device, ckpt


@st.cache_resource
def load_diffusion_models(checkpoint_path: str):
    """Load diffusion model, VAE decoder, and CLIP encoder (cached)."""
    device = get_device("auto")

    clip_encoder = CLIPTextEncoder()
    clip_encoder = clip_encoder.to(device)
    clip_encoder.eval()

    with torch.no_grad():
        uncond_embed = clip_encoder.encode([""])
    uncond_embed = uncond_embed.to(device)

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
        "ckpt": ckpt,
    }


@st.cache_resource
def load_animation_models(
    diffusion_checkpoint_path: str,
    motion_checkpoint_path: str | None,
):
    """Load AnimatedMMDiT model for GIF generation (cached)."""
    device = get_device("auto")

    clip_encoder = CLIPTextEncoder()
    clip_encoder = clip_encoder.to(device)
    clip_encoder.eval()

    with torch.no_grad():
        uncond_embed = clip_encoder.encode([""])
    uncond_embed = uncond_embed.to(device)

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
        "ckpt": diffusion_ckpt,
    }


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def vae_reconstruction_page():
    """VAE reconstruction demo page."""
    st.header("VAE Reconstruction")
    st.caption("Upload an image to see how the VAE encodes and reconstructs it.")

    vae_path = get_vae_checkpoint_path()
    if vae_path is None:
        st.error(
            "**No VAE checkpoint found.**  \n"
            "Train a VAE first:  \n"
            "`uv run main.py --train-vae --epochs 100`"
        )
        st.stop()

    with st.status("Loading VAE model...", expanded=False) as status:
        try:
            vae, image_size, device, ckpt = load_vae(str(vae_path))
            status.update(label=f"VAE loaded from `{vae_path.name}`", state="complete")
        except Exception as e:
            status.update(label="Failed to load VAE", state="error")
            st.error(f"Could not load VAE checkpoint `{vae_path.name}`:  \n{e}")
            st.stop()

    show_checkpoint_info(ckpt, label="VAE Checkpoint Info")

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "webp"],
        help="Upload an image to reconstruct through the VAE",
    )

    if uploaded_file is None:
        return

    original_img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        st.image(original_img, use_container_width=True)

    with st.spinner("Encoding & decoding through VAE..."):
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        x = transform(original_img).unsqueeze(0).to(device)

        with torch.no_grad():
            mean, logvar = vae.encode(x)
            recon = vae.decode(mean)

        recon_tensor = (recon + 1) / 2
        recon_tensor = recon_tensor.clamp(0, 1)
        recon_np = recon_tensor[0].permute(1, 2, 0).cpu().numpy()
        recon_np = (recon_np * 255).astype("uint8")
        recon_img = Image.fromarray(recon_np)

    with col2:
        st.subheader("Reconstructed")
        st.image(recon_img, use_container_width=True)

    st.divider()

    # Compression metrics
    info_tab, latent_tab = st.tabs(["Compression Metrics", "Latent Details"])

    with info_tab:
        m1, m2, m3 = st.columns(3)
        input_pixels = image_size * image_size * 3
        latent_elements = mean.shape[1] * mean.shape[2] * mean.shape[3]
        ratio = input_pixels / latent_elements
        m1.metric("Input Pixels", f"{input_pixels:,}")
        m2.metric("Latent Elements", f"{latent_elements:,}")
        m3.metric("Compression Ratio", f"{ratio:.1f}x")

    with latent_tab:
        st.write(f"**Shape:** `{list(mean.shape)}`")
        st.write(f"**Mean:** `{mean.mean().item():.4f}`")
        st.write(f"**Std:** `{mean.std().item():.4f}`")
        st.write(f"**Min:** `{mean.min().item():.4f}`")
        st.write(f"**Max:** `{mean.max().item():.4f}`")


def diffusion_generation_page():
    """Diffusion generation demo page."""
    st.header("Text-to-Image Generation")
    st.caption("Generate images from text prompts using the trained diffusion model.")

    diffusion_path = get_diffusion_checkpoint_path()
    if diffusion_path is None:
        st.error(
            "**No diffusion checkpoint found.**  \n"
            "Train a diffusion model first:  \n"
            "`uv run main.py --train-diffusion --epochs 200`"
        )
        st.stop()

    with st.status("Loading diffusion model...", expanded=False) as status:
        try:
            models = load_diffusion_models(str(diffusion_path))
            status.update(
                label=f"Diffusion model loaded from `{diffusion_path.name}`",
                state="complete",
            )
        except FileNotFoundError as e:
            status.update(label="Missing dependency", state="error")
            st.error(f"**Missing checkpoint:**  \n{e}")
            st.stop()
        except Exception as e:
            status.update(label="Failed to load model", state="error")
            st.error(f"Could not load diffusion model `{diffusion_path.name}`:  \n{e}")
            st.stop()

    show_checkpoint_info(models["ckpt"], label="Diffusion Checkpoint Info")

    st.divider()

    # Generation controls in a form
    with st.form("diffusion_form"):
        prompt = st.text_input(
            "Prompt",
            value="a photo of a cat",
            help="Enter a text description of the image you want to generate",
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            num_steps = st.slider(
                "Sampling Steps", 10, 100, 50, help="More steps = better quality"
            )
        with col2:
            guidance_scale = st.slider(
                "Guidance Scale", 1.0, 20.0, 7.5, help="Higher = more faithful to prompt"
            )
        with col3:
            seed = st.number_input(
                "Seed", value=42, min_value=0, help="Random seed for reproducibility"
            )

        submitted = st.form_submit_button(
            "Generate", type="primary", use_container_width=True
        )

    if not submitted:
        return

    if not prompt.strip():
        st.warning("Please enter a prompt.")
        st.stop()

    with st.spinner(f"Generating with {num_steps} DDIM steps..."):
        torch.manual_seed(seed)

        diffusion = Diffusion(
            num_timesteps=1000,
            beta_schedule="cosine",
            guidance_scale=guidance_scale,
            uncond_embed=models["uncond_embed"],
        )

        text_embeds = models["clip_encoder"].encode([prompt])
        text_embeds = text_embeds.to(models["device"])

        with torch.no_grad():
            images = diffusion.sample(
                model=models["model"],
                shape=(
                    1,
                    models["in_channels"],
                    models["latent_size"],
                    models["latent_size"],
                ),
                text_embeds=text_embeds,
                num_steps=num_steps,
                use_ddim=True,
                use_cfg=True,
                vae_decoder=models["vae"],
            )

        img = images[0]
        img = img.permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        pil_img = Image.fromarray(img)

    st.subheader("Generated Image")
    st.image(pil_img, use_container_width=True)

    # Generation settings & download
    with st.expander("Generation Settings"):
        st.json({
            "prompt": prompt,
            "steps": num_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "checkpoint": diffusion_path.name,
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = prompt[:20].replace(" ", "_")
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    st.download_button(
        label="Download Image",
        data=buf.getvalue(),
        file_name=f"generated_{safe_prompt}_{timestamp}.png",
        mime="image/png",
    )


def gif_generation_page():
    """GIF generation demo page."""
    st.header("GIF Generation")
    st.caption("Generate animated GIFs from text prompts using the Motion Module.")

    diffusion_path = get_diffusion_checkpoint_path()
    if diffusion_path is None:
        st.error(
            "**No diffusion checkpoint found.**  \n"
            "Train a diffusion model first:  \n"
            "`uv run main.py --train-diffusion --epochs 200`"
        )
        st.stop()

    motion_path = get_motion_checkpoint_path()
    if motion_path is None:
        st.warning(
            "Motion checkpoint not found. Using untrained motion module.  \n"
            "For better results, train with:  \n"
            "`uv run main.py --train-motion --epochs 50`"
        )

    with st.status("Loading animation models...", expanded=False) as status:
        try:
            models = load_animation_models(
                str(diffusion_path),
                str(motion_path) if motion_path else None,
            )
            loaded_label = f"Models loaded (diffusion: `{diffusion_path.name}`"
            if motion_path:
                loaded_label += f", motion: `{motion_path.name}`"
            loaded_label += ")"
            status.update(label=loaded_label, state="complete")
        except FileNotFoundError as e:
            status.update(label="Missing dependency", state="error")
            st.error(f"**Missing checkpoint:**  \n{e}")
            st.stop()
        except Exception as e:
            status.update(label="Failed to load models", state="error")
            st.error(f"Could not load animation models:  \n{e}")
            st.stop()

    show_checkpoint_info(models["ckpt"], label="Diffusion Checkpoint Info")

    st.divider()

    # Generation controls in a form
    with st.form("gif_form"):
        prompt = st.text_input(
            "Prompt",
            value="a cat walking",
            help="Enter a text description of the animation you want to generate",
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            num_frames = st.slider("Frames", 4, 32, 16, help="Number of animation frames")
        with col2:
            num_steps = st.slider("Steps", 10, 100, 50, help="Diffusion sampling steps")
        with col3:
            guidance_scale = st.slider("Guidance", 1.0, 20.0, 7.5, help="CFG scale")
        with col4:
            fps = st.slider("FPS", 4, 24, 8, help="Frames per second")

        seed = st.number_input("Seed", value=42, min_value=0, help="Random seed")

        submitted = st.form_submit_button(
            "Generate GIF", type="primary", use_container_width=True
        )

    if not submitted:
        return

    if not prompt.strip():
        st.warning("Please enter a prompt.")
        st.stop()

    with st.spinner(f"Generating {num_frames} frames with {num_steps} steps..."):
        torch.manual_seed(seed)

        diffusion = AnimatedDiffusion(
            num_timesteps=1000,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            uncond_embed=models["uncond_embed"],
        )

        text_embeds = models["clip_encoder"].encode([prompt])
        text_embeds = text_embeds.to(models["device"])

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

        video = video[0]  # (F, 3, H, W)
        frames = []
        for i in range(video.shape[0]):
            frame = video[i]
            frame = frame.permute(1, 2, 0).mul(255).clamp(0, 255)
            frame = frame.to(torch.uint8).cpu().numpy()
            frames.append(Image.fromarray(frame))

    # Frame preview grid before full animation
    st.subheader("Frame Preview")
    preview_count = min(8, len(frames))
    preview_cols = st.columns(preview_count)
    step = max(1, len(frames) // preview_count)
    for i in range(preview_count):
        idx = min(i * step, len(frames) - 1)
        preview_cols[i].image(frames[idx], caption=f"#{idx + 1}", use_container_width=True)

    st.divider()

    # Assembled GIF
    st.subheader("Generated GIF")

    gif_buffer = BytesIO()
    duration = int(1000 / fps)

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

    st.image(gif_buffer, use_container_width=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = prompt[:20].replace(" ", "_")
    gif_buffer.seek(0)
    st.download_button(
        label="Download GIF",
        data=gif_buffer.getvalue(),
        file_name=f"generated_{safe_prompt}_{timestamp}.gif",
        mime="image/gif",
    )

    # All frames
    with st.expander("View All Frames"):
        cols = st.columns(min(4, num_frames))
        for i, frame in enumerate(frames):
            cols[i % 4].image(frame, caption=f"Frame {i + 1}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="tiny-stable-diffusion Demo",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

    st.title("tiny-stable-diffusion Demo")
    st.caption("An educational implementation of Stable Diffusion 3 from scratch.")

    # -- Sidebar --
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Demo",
        ["VAE Reconstruction", "Text-to-Image Generation", "GIF Generation"],
        help="Choose which model to demo",
    )
    st.sidebar.divider()

    render_sidebar_checkpoints()
    render_sidebar_system_info()

    st.sidebar.markdown("### Quick Start")
    st.sidebar.code(
        "# Train VAE\n"
        "uv run main.py --train-vae\n\n"
        "# Train Diffusion\n"
        "uv run main.py --train-diffusion\n\n"
        "# Train Motion (for GIF)\n"
        "uv run main.py --train-motion",
        language="bash",
    )

    # -- Route to selected page --
    if page == "VAE Reconstruction":
        vae_reconstruction_page()
    elif page == "Text-to-Image Generation":
        diffusion_generation_page()
    else:
        gif_generation_page()


if __name__ == "__main__":
    main()
