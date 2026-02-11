"""Streamlit demo app for tiny-stable-diffusion.

Run with:
    uv run streamlit run src/demo/app.py
"""

from __future__ import annotations

import json
from datetime import datetime
from io import BytesIO
from pathlib import Path

import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image

from src.config import get_config
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
SAMPLE_IMAGE_DIR = Path("assets/samples")

DIFFUSION_PROMPT_HINTS = [
    "a fluffy orange cat on a sofa",
    "a red sports car on a rainy street",
    "a small cabin in snowy mountains",
    "a sunflower field at sunset",
    "a bowl of ramen on a wooden table",
    "a futuristic city skyline at night",
    "a corgi wearing sunglasses",
    "a lighthouse by rough ocean waves",
    "a watercolor painting of a tulip",
    "an astronaut walking on the moon",
]

_CUSTOM_CSS = """
<style>
div.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.app-hero {
    border: 1px solid #d9e7ff;
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
    background: linear-gradient(135deg, #f8fbff 0%, #f5f8ff 100%);
}

.app-hero h1 {
    margin: 0;
    font-size: 2rem;
    letter-spacing: -0.02em;
    color: #0f172a;
}

.app-hero p {
    margin: 0.5rem 0 0 0;
    color: #334155;
    font-size: 0.98rem;
}

div[data-testid="stMetric"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px 16px !important;
    min-height: 98px;
}

div[data-testid="stMetric"] * {
    color: #0f172a !important;
}

div[data-testid="stMetricLabel"] p {
    color: #334155 !important;
    margin-bottom: 0.35rem;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #0f172a !important;
}

div[data-testid="stTabContent"] > div {
    padding: 0.35rem 0.2rem 0.15rem 0.2rem;
}

div[data-testid="stMetric"] > div {
    overflow: hidden;
}

div[data-testid="stHorizontalBlock"] > div {
    align-self: stretch;
}

div[data-testid="stFileUploader"] {
    border-radius: 12px;
    border: 1px dashed #bfd2ff;
    background-color: #f8fbff;
}

.hint-box {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    background: #f8fafc;
    padding: 0.75rem 0.9rem;
    margin-bottom: 0.75rem;
    color: #334155;
}

button[data-baseweb="tab"] {
    color: #ffffff !important;
    background: transparent;
    border-radius: 10px 10px 0 0;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #ffffff !important;
    background: transparent;
}

div[data-testid="stTabContent"] p,
div[data-testid="stTabContent"] code {
    color: #0f172a;
}
</style>
"""


# ---------------------------------------------------------------------------
# Checkpoint scanning utilities
# ---------------------------------------------------------------------------


def _format_bytes(size: int) -> str:
    """Format byte size into a human-readable string."""
    size_f = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if size_f < 1024:
            return f"{size_f:.1f} {unit}"
        size_f /= 1024
    return f"{size_f:.1f} TB"


def _format_mtime(mtime: float) -> str:
    """Format modification timestamp to a readable date string."""
    return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")


def scan_checkpoints(checkpoint_dir: Path, prefix: str) -> list[dict[str, str | int | float]]:
    """Scan checkpoint directory for files matching *prefix*.

    Returns file-level metadata only (no torch.load) sorted by mtime desc.
    """
    if not checkpoint_dir.exists():
        return []

    results = []
    for p in checkpoint_dir.glob(f"{prefix}*.pt"):
        stat = p.stat()
        results.append(
            {
                "path": str(p),
                "name": p.name,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "display": f"{p.name}  ({_format_bytes(stat.st_size)}, {_format_mtime(stat.st_mtime)})",
            }
        )

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
    selected_path = str(candidates[selected_idx]["path"])
    st.session_state[session_key] = selected_path

    return Path(selected_path)


def render_sidebar_checkpoints() -> None:
    """Render checkpoint selection widgets in the sidebar."""
    st.sidebar.markdown("### Checkpoints")

    _checkpoint_selectbox("VAE Checkpoint", "vae", "vae_checkpoint")
    _checkpoint_selectbox("Diffusion Checkpoint", "diffusion", "diffusion_checkpoint")

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
    diffusion_config = get_config("diffusion_train")

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

    config_scaling_factor = diffusion_config.get("scaling_factor", 0.4869)
    if not isinstance(config_scaling_factor, (int, float)):
        config_scaling_factor = 0.4869
    vae.set_scaling_factor(float(config_scaling_factor))

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


def _sample_label(path: Path) -> str:
    """Build a readable label from sample image filename."""
    stem = path.stem
    if stem.startswith("sample_") and "_" in stem[7:]:
        stem = stem.split("_", 2)[-1]
    return stem.replace("_", " ").title()


def list_sample_images(limit: int = 48) -> list[Path]:
    """Return sample images for quick VAE testing."""
    if not SAMPLE_IMAGE_DIR.exists():
        return []

    candidates = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        candidates.extend(SAMPLE_IMAGE_DIR.glob(ext))
    return sorted(candidates, key=lambda p: p.name)[:limit]


def _render_hero() -> None:
    st.markdown(
        """
        <section class="app-hero">
            <h1>tiny-stable-diffusion studio</h1>
            <p>Explore VAE reconstruction and text-to-image generation with checkpoint-aware controls.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def vae_reconstruction_page():
    """VAE reconstruction demo page."""
    st.header("VAE Reconstruction")
    st.caption("Try built-in samples or upload your own image to inspect VAE compression.")

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

    source_col, input_col = st.columns([1, 2])
    with source_col:
        source = st.radio(
            "Input Source",
            ["Sample Image", "Upload Image"],
            horizontal=False,
            help="Use a project sample or your own file",
        )

    original_img = None
    with input_col:
        if source == "Sample Image":
            sample_images = list_sample_images()
            if not sample_images:
                st.warning("No sample images found in `assets/samples`.")
                return

            sample_choice = st.selectbox(
                "Sample",
                options=[str(p) for p in sample_images],
                format_func=lambda p: _sample_label(Path(p)),
                help="Pick a sample image bundled with this workspace",
            )
            original_img = Image.open(sample_choice).convert("RGB")
            st.caption(f"Using sample: `{Path(sample_choice).name}`")
        else:
            uploaded_file = st.file_uploader(
                "Upload an image",
                type=["png", "jpg", "jpeg", "webp"],
                help="Upload an image to reconstruct through the VAE",
            )
            if uploaded_file is None:
                return
            original_img = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Encoding & decoding through VAE..."):
        transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        transformed = transform(original_img)
        x = transformed if isinstance(transformed, torch.Tensor) else T.ToTensor()(transformed)
        x = x.unsqueeze(0).to(device)

        with torch.no_grad():
            mean, logvar = vae.encode(x)
            recon = vae.decode(mean)

        recon_tensor = (recon + 1) / 2
        recon_tensor = recon_tensor.clamp(0, 1)
        recon_np = recon_tensor[0].permute(1, 2, 0).cpu().numpy()
        recon_np = (recon_np * 255).astype("uint8")
        recon_img = Image.fromarray(recon_np)

    input_pixels = image_size * image_size * 3
    latent_elements = mean.shape[1] * mean.shape[2] * mean.shape[3]
    ratio = input_pixels / latent_elements
    latent_details = {
        "shape": list(mean.shape),
        "mean": round(mean.mean().item(), 4),
        "std": round(mean.std().item(), 4),
        "min": round(mean.min().item(), 4),
        "max": round(mean.max().item(), 4),
    }

    preview_col, details_col = st.columns([1.2, 1.0])
    with preview_col:
        st.subheader("Reconstruction Preview")
        original_col, reconstructed_col = st.columns(2)
        with original_col:
            st.caption("Original")
            st.image(original_img, width="content")
        with reconstructed_col:
            st.caption("Reconstructed")
            st.image(recon_img, width="content")

    with details_col:
        st.subheader("Compression Config")
        m1, m2, m3 = st.columns(3)
        m1.metric("Input Pixels", f"{input_pixels:,}")
        m2.metric("Latent Elements", f"{latent_elements:,}")
        m3.metric("Compression Ratio", f"{ratio:.1f}x")
        st.json(latent_details)


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

    st.markdown(
        f"""
        <div class="hint-box">
            Need inspiration? Start from one of {len(DIFFUSION_PROMPT_HINTS)} curated sample prompts,
            then tweak style/subjects before generating.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "diffusion_prompt_text" not in st.session_state:
        st.session_state["diffusion_prompt_text"] = DIFFUSION_PROMPT_HINTS[0]

    hint_choice = st.selectbox(
        "Sample Prompt Hint",
        options=["Custom"] + DIFFUSION_PROMPT_HINTS,
        index=1,
        help="Pick a ready-to-use prompt as a starting point",
    )
    if hint_choice != "Custom":
        st.session_state["diffusion_prompt_text"] = hint_choice

    # Generation controls in a form
    with st.form("diffusion_form"):
        prompt = st.text_input(
            "Prompt",
            key="diffusion_prompt_text",
            help="Enter a text description of the image you want to generate",
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            num_steps = st.slider(
                "Sampling Steps", 10, 100, 50, help="More steps usually improve quality"
            )
        with col2:
            guidance_scale = st.slider(
                "Guidance Scale", 1.0, 20.0, 7.5, help="Higher = more faithful to prompt"
            )
        with col3:
            seed = st.number_input(
                "Seed", value=42, min_value=0, help="Random seed for reproducibility"
            )

        submitted = st.form_submit_button("Generate", type="primary", width="stretch")

    if not submitted:
        return

    if not prompt.strip():
        st.warning("Please enter a prompt.")
        st.stop()

    with st.spinner(f"Generating with {num_steps} sampling steps..."):
        torch.manual_seed(seed)

        diffusion = Diffusion(
            num_timesteps=1000,
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
                use_cfg=True,
                vae_decoder=models["vae"],
            )

        img = images[0]
        img = img.permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        pil_img = Image.fromarray(img)

    generation_settings = {
        "prompt": prompt,
        "steps": num_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "checkpoint": diffusion_path.name,
        "image_size": "64x64",
    }

    image_col, config_col = st.columns([1, 1.2])
    with image_col:
        st.subheader("Generated Image")
        st.image(pil_img, width="content")
        st.caption("Displayed at content width (native 64x64 output)")

    with config_col:
        st.subheader("Generation Config")
        st.json(generation_settings)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = prompt[:20].replace(" ", "_")
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    with config_col:
        st.download_button(
            label="Download Image",
            data=buf.getvalue(),
            file_name=f"generated_{safe_prompt}_{timestamp}.png",
            mime="image/png",
            width="stretch",
        )


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
    _render_hero()

    # -- Sidebar --
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Demo",
        ["VAE Reconstruction", "Text-to-Image Generation"],
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
        "uv run main.py --train-diffusion",
        language="bash",
    )

    # -- Route to selected page --
    if page == "VAE Reconstruction":
        vae_reconstruction_page()
    else:
        diffusion_generation_page()


if __name__ == "__main__":
    main()
