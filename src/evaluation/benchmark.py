"""Sampling speed and memory benchmarking for diffusion models.

Measures:
- Total generation time and sec/img speed
- Per-stage breakdown (text encoding, diffusion sampling, VAE decoding)
- Peak accelerator memory usage (CUDA/MPS)
- Peak process RAM usage
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from src.models.diffusion import Diffusion
from src.models.factory import DiT
from src.models.vae import create_vae
from src.text_encoder.clip_encoder import CLIPTextEncoder
from src.training.checkpoint import find_latest_checkpoint
from src.utils.common import get_device


@dataclass(frozen=True)
class BenchmarkResult:
    """Result from a single benchmark run."""

    num_samples: int
    num_steps: int
    batch_size: int
    total_time_sec: float
    time_per_image_sec: float
    sec_per_image: float
    peak_memory_mb: float
    peak_memory_reserved_mb: float
    peak_ram_mb: float
    ram_delta_mb: float
    text_encoding_ms: float
    diffusion_sampling_ms: float
    vae_decoding_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


class _Timer:
    """Context manager for timing code blocks using appropriate method per device."""

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._elapsed_ms = 0.0
        self._start_time = 0.0
        self._end_time = 0.0
        self._start_event: torch.cuda.Event | None = None
        self._end_event: torch.cuda.Event | None = None

        if device.type == "cuda":
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self) -> _Timer:
        if self._device.type == "cuda":
            assert self._start_event is not None
            self._start_event.record()
        else:
            self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        if self._device.type == "cuda":
            assert self._start_event is not None
            assert self._end_event is not None
            self._end_event.record()
            torch.cuda.synchronize()
            self._elapsed_ms = self._start_event.elapsed_time(self._end_event)
        else:
            self._end_time = time.perf_counter()
            self._elapsed_ms = (self._end_time - self._start_time) * 1000.0

    @property
    def elapsed_ms(self) -> float:
        return self._elapsed_ms


def _get_process_rss_mb() -> float:
    """Get current process resident memory (RAM) in MB."""
    try:
        output = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            text=True,
        ).strip()
        rss_kb = float(output)
        return rss_kb / 1024.0
    except (subprocess.SubprocessError, ValueError):
        return 0.0


def _get_mps_memory_mb() -> tuple[float, float]:
    """Return (allocated, reserved/driver) memory in MB for MPS, if available."""
    if not hasattr(torch, "mps"):
        return 0.0, 0.0
    if not torch.backends.mps.is_available():
        return 0.0, 0.0

    allocated_mb = 0.0
    reserved_mb = 0.0

    try:
        allocated_mb = float(torch.mps.current_allocated_memory()) / (1024 * 1024)
    except (AttributeError, RuntimeError):
        allocated_mb = 0.0

    try:
        reserved_mb = float(torch.mps.driver_allocated_memory()) / (1024 * 1024)
    except (AttributeError, RuntimeError):
        reserved_mb = 0.0

    return allocated_mb, reserved_mb


def _load_models(
    checkpoint: str | Path | None,
    vae_checkpoint: str | Path | None,
    device: torch.device,
) -> tuple:
    """Load all models needed for benchmarking.

    Returns:
        Tuple of (dit_model, vae, clip_encoder, diffusion, model_config)
    """
    from src.inference.generator import infer_model_config_from_state_dict

    clip_encoder = CLIPTextEncoder()
    clip_encoder = clip_encoder.to(str(device))
    clip_encoder.eval()

    with torch.no_grad():
        uncond_embed = clip_encoder.encode([""])
    uncond_embed = uncond_embed.to(device)

    if checkpoint is None:
        checkpoint = find_latest_checkpoint("checkpoints", prefix="diffusion")
        if checkpoint is None:
            checkpoint = Path("checkpoints/diffusion.pt")
    checkpoint = Path(checkpoint)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Diffusion checkpoint not found: {checkpoint}")

    ckpt = torch.load(checkpoint, map_location=device)
    model_config = ckpt.get("model_config", ckpt.get("config", {}))

    if not model_config and "model_state_dict" in ckpt:
        model_config = infer_model_config_from_state_dict(ckpt["model_state_dict"])

    latent_size = model_config.get("latent_size", 8)
    in_channels = model_config.get("in_channels", 16)
    image_size = model_config.get("image_size", 64)

    if vae_checkpoint is None:
        ckpt_candidate = model_config.get("vae_checkpoint")
        if isinstance(ckpt_candidate, (str, Path)):
            vae_checkpoint = ckpt_candidate
        else:
            vae_checkpoint = "checkpoints/vae.pt"
    vae_checkpoint = Path(vae_checkpoint)

    if not vae_checkpoint.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_checkpoint}")

    vae = create_vae(
        image_size=image_size,
        z_channels=in_channels,
        ch=model_config.get("vae_ch", 64),
        ch_mult=tuple(model_config.get("vae_ch_mult", [1, 2, 4, 4])),
    )
    vae_state = torch.load(vae_checkpoint, map_location=device)
    vae.load_state_dict(vae_state["model_state_dict"])
    vae = vae.to(device)
    vae.eval()

    if "scaling_factor" in vae_state:
        vae.set_scaling_factor(vae_state["scaling_factor"])
    else:
        sf = model_config.get("scaling_factor", 1.0)
        if isinstance(sf, (int, float)):
            vae.set_scaling_factor(float(sf))

    model = DiT(
        in_channels=in_channels,
        image_size=latent_size,
        patch_size=model_config.get("patch_size", 2),
        model_size=model_config.get("model_size", "S"),
        clip_embed_dim=clip_encoder.embedding_dim,
        model_type=model_config.get("model_type", "dit"),
        qk_rmsnorm=model_config.get("qk_rmsnorm", True),
        register_tokens=model_config.get("register_tokens", 0),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    diffusion = Diffusion(
        num_timesteps=1000,
        guidance_scale=model_config.get("guidance_scale", 7.5),
        uncond_embed=uncond_embed,
    )

    return model, vae, clip_encoder, diffusion, model_config


@torch.no_grad()
def benchmark_generation(
    checkpoint: str | Path | None = None,
    vae_checkpoint: str | Path | None = None,
    prompt: str = "a cat sitting on a couch",
    num_steps: int = 50,
    guidance_scale: float = 7.5,
    batch_size: int = 1,
    warmup_runs: int = 3,
    device: str = "auto",
) -> BenchmarkResult:
    """Benchmark a single generation configuration.

    Args:
        checkpoint: Path to diffusion checkpoint
        vae_checkpoint: Path to VAE checkpoint
        prompt: Text prompt for generation
        num_steps: Number of diffusion steps
        guidance_scale: CFG guidance scale
        batch_size: Number of images per batch
        warmup_runs: Number of warmup iterations before timing
        device: Device to use

    Returns:
        BenchmarkResult with timing and memory data
    """
    device_obj = get_device(device)
    print(f"Loading models on {device_obj}...")

    model, vae, clip_encoder, diffusion, model_config = _load_models(
        checkpoint, vae_checkpoint, device_obj
    )
    diffusion.guidance_scale = guidance_scale

    latent_size = model_config.get("latent_size", 8)
    in_channels = model_config.get("in_channels", 16)
    shape = (batch_size, in_channels, latent_size, latent_size)

    prompts = [prompt] * batch_size

    # Reset accelerator memory stats
    if device_obj.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_obj)

    # RAM tracker for CPU/MPS/CUDA process memory
    ram_start_mb = _get_process_rss_mb()
    ram_peak_mb = ram_start_mb

    # MPS memory tracker (sampled manually since there is no peak API like CUDA)
    mps_alloc_peak_mb = 0.0
    mps_driver_peak_mb = 0.0

    def sample_memory() -> None:
        nonlocal ram_peak_mb, mps_alloc_peak_mb, mps_driver_peak_mb

        current_ram = _get_process_rss_mb()
        if current_ram > ram_peak_mb:
            ram_peak_mb = current_ram

        if device_obj.type == "mps":
            mps_alloc_mb, mps_driver_mb = _get_mps_memory_mb()
            if mps_alloc_mb > mps_alloc_peak_mb:
                mps_alloc_peak_mb = mps_alloc_mb
            if mps_driver_mb > mps_driver_peak_mb:
                mps_driver_peak_mb = mps_driver_mb

    sample_memory()

    # Warmup runs (not timed)
    print(f"Running {warmup_runs} warmup iterations...")
    for _ in range(warmup_runs):
        text_embeds = clip_encoder.encode(prompts).to(device_obj)
        sample_memory()
        x_t = torch.randn(shape, device=device_obj)
        sample_memory()
        timesteps = torch.linspace(999, 0, num_steps + 1, device=device_obj).long()
        for i in range(num_steps):
            t_curr = torch.full(
                (batch_size,), int(timesteps[i].item()), device=device_obj, dtype=torch.long
            )
            t_next = torch.full(
                (batch_size,), int(timesteps[i + 1].item()), device=device_obj, dtype=torch.long
            )
            x_t = diffusion.euler_step(model, x_t, t_curr, t_next, text_embeds, use_cfg=True)
            sample_memory()
        _ = vae.decode_from_latent(x_t)
        sample_memory()

    # Reset memory stats after warmup
    if device_obj.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_obj)
    sample_memory()

    # Timed run: text encoding
    text_timer = _Timer(device_obj)
    with text_timer:
        text_embeds = clip_encoder.encode(prompts).to(device_obj)
    sample_memory()

    # Timed run: diffusion sampling
    diffusion_timer = _Timer(device_obj)
    with diffusion_timer:
        x_t = torch.randn(shape, device=device_obj)
        sample_memory()
        timesteps = torch.linspace(999, 0, num_steps + 1, device=device_obj).long()
        for i in range(num_steps):
            t_curr = torch.full(
                (batch_size,), int(timesteps[i].item()), device=device_obj, dtype=torch.long
            )
            t_next = torch.full(
                (batch_size,), int(timesteps[i + 1].item()), device=device_obj, dtype=torch.long
            )
            x_t = diffusion.euler_step(model, x_t, t_curr, t_next, text_embeds, use_cfg=True)
            sample_memory()

    # Timed run: VAE decoding
    vae_timer = _Timer(device_obj)
    with vae_timer:
        _ = vae.decode_from_latent(x_t)
    sample_memory()

    # Memory stats
    peak_memory_mb = 0.0
    peak_memory_reserved_mb = 0.0
    if device_obj.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device_obj) / (1024 * 1024)
        peak_memory_reserved_mb = torch.cuda.max_memory_reserved(device_obj) / (1024 * 1024)
    elif device_obj.type == "mps":
        peak_memory_mb = mps_alloc_peak_mb
        peak_memory_reserved_mb = mps_driver_peak_mb

    ram_delta_mb = max(0.0, ram_peak_mb - ram_start_mb)

    total_time_sec = (
        text_timer.elapsed_ms + diffusion_timer.elapsed_ms + vae_timer.elapsed_ms
    ) / 1000.0
    time_per_image = total_time_sec / batch_size
    sec_per_image = time_per_image

    return BenchmarkResult(
        num_samples=batch_size,
        num_steps=num_steps,
        batch_size=batch_size,
        total_time_sec=total_time_sec,
        time_per_image_sec=time_per_image,
        sec_per_image=sec_per_image,
        peak_memory_mb=peak_memory_mb,
        peak_memory_reserved_mb=peak_memory_reserved_mb,
        peak_ram_mb=ram_peak_mb,
        ram_delta_mb=ram_delta_mb,
        text_encoding_ms=text_timer.elapsed_ms,
        diffusion_sampling_ms=diffusion_timer.elapsed_ms,
        vae_decoding_ms=vae_timer.elapsed_ms,
    )


def benchmark_sweep(
    checkpoint: str | Path | None = None,
    vae_checkpoint: str | Path | None = None,
    prompt: str = "a cat sitting on a couch",
    num_steps_list: list[int] | None = None,
    batch_sizes: list[int] | None = None,
    warmup_runs: int = 3,
    device: str = "auto",
) -> list[BenchmarkResult]:
    """Run benchmarks across multiple configurations.

    Args:
        checkpoint: Path to diffusion checkpoint
        vae_checkpoint: Path to VAE checkpoint
        prompt: Text prompt
        num_steps_list: List of step counts to test
        batch_sizes: List of batch sizes to test
        warmup_runs: Number of warmup iterations
        device: Device to use

    Returns:
        List of BenchmarkResult, one per configuration
    """
    if num_steps_list is None:
        num_steps_list = [10, 25, 50, 100]
    if batch_sizes is None:
        batch_sizes = [1, 4, 8]

    results = []
    total_configs = len(num_steps_list) * len(batch_sizes)
    config_idx = 0

    for steps in num_steps_list:
        for bs in batch_sizes:
            config_idx += 1
            print(f"\n[{config_idx}/{total_configs}] steps={steps}, batch_size={bs}")
            try:
                result = benchmark_generation(
                    checkpoint=checkpoint,
                    vae_checkpoint=vae_checkpoint,
                    prompt=prompt,
                    num_steps=steps,
                    batch_size=bs,
                    warmup_runs=warmup_runs,
                    device=device,
                )
                results.append(result)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM with batch_size={bs}, skipping...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise

    return results


def format_benchmark_results(
    results: list[BenchmarkResult],
    detail_result: BenchmarkResult | None = None,
) -> str:
    """Format benchmark results into a readable table.

    Args:
        results: List of BenchmarkResult
        detail_result: Optional result to show per-stage breakdown

    Returns:
        Formatted string
    """
    lines = [
        "",
        "\u2550" * 58,
        "  Sampling Benchmark Results",
        "\u2550" * 58,
        f"  {'Steps':>5}  {'Batch':>5}  {'Total(s)':>10}  {'Sec/img':>12}  {'Peak VRAM':>10}",
        "  {}  {}  {}  {}  {}".format(
            "\u2500" * 5, "\u2500" * 5, "\u2500" * 10, "\u2500" * 12, "\u2500" * 10
        ),
    ]

    for r in results:
        vram_str = f"{r.peak_memory_mb:.1f} MB" if r.peak_memory_mb > 0 else "N/A"
        lines.append(
            f"  {r.num_steps:>5}  {r.batch_size:>5}  "
            f"{r.total_time_sec:>8.2f}s  "
            f"{r.sec_per_image:>8.2f} s/img  "
            f"{vram_str:>10}"
        )

    lines.append("\u2550" * 58)

    # Breakdown section
    if detail_result is not None:
        total_ms = (
            detail_result.text_encoding_ms
            + detail_result.diffusion_sampling_ms
            + detail_result.vae_decoding_ms
        )
        if total_ms > 0:
            text_pct = detail_result.text_encoding_ms / total_ms * 100
            diff_pct = detail_result.diffusion_sampling_ms / total_ms * 100
            vae_pct = detail_result.vae_decoding_ms / total_ms * 100
        else:
            text_pct = diff_pct = vae_pct = 0.0

        lines.extend(
            [
                f"  Breakdown (steps={detail_result.num_steps}, batch={detail_result.batch_size}):",
                f"    Text encoding:       {detail_result.text_encoding_ms:>7.0f}ms ({text_pct:.1f}%)",
                f"    Diffusion sampling:  {detail_result.diffusion_sampling_ms:>7.0f}ms ({diff_pct:.1f}%)",
                f"    VAE decoding:        {detail_result.vae_decoding_ms:>7.0f}ms ({vae_pct:.1f}%)",
            ]
        )

    return "\n".join(lines)
