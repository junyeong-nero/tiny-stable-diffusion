"""Inference profiling utilities for VRAM/RAM and latency measurement.

This module wraps diffusion benchmark runs and aggregates repeated measurements
to provide stable estimates of inference latency, accelerator memory, and RAM.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev

from src.evaluation.benchmark import BenchmarkResult, benchmark_generation


@dataclass(frozen=True)
class InferenceProfileSummary:
    """Aggregated summary over repeated inference benchmark runs."""

    repeats: int
    warmup_runs: int
    num_steps: int
    batch_size: int
    prompt: str
    checkpoint: str | None
    vae_checkpoint: str | None
    device: str
    latency_mean_ms: float
    latency_std_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_min_ms: float
    latency_max_ms: float
    sec_per_image_mean: float
    sec_per_image_p50: float
    sec_per_image_p95: float
    peak_vram_mean_mb: float
    peak_vram_max_mb: float
    peak_reserved_max_mb: float
    peak_ram_mean_mb: float
    peak_ram_max_mb: float
    ram_delta_mean_mb: float
    ram_delta_max_mb: float
    text_encoding_mean_ms: float
    diffusion_sampling_mean_ms: float
    vae_decoding_mean_ms: float
    runs: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def _percentile(values: list[float], q: float) -> float:
    """Compute percentile with linear interpolation."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]

    lo_val = sorted_values[lo]
    hi_val = sorted_values[hi]
    alpha = pos - lo
    return lo_val + (hi_val - lo_val) * alpha


def profile_diffusion_inference(
    checkpoint: str | Path | None = None,
    vae_checkpoint: str | Path | None = None,
    prompt: str = "a cat sitting on a couch",
    num_steps: int = 50,
    batch_size: int = 1,
    repeats: int = 5,
    warmup_runs: int = 3,
    device: str = "auto",
) -> InferenceProfileSummary:
    """Profile diffusion inference latency, accelerator memory, and RAM usage.

    Args:
        checkpoint: Diffusion checkpoint path
        vae_checkpoint: VAE checkpoint path
        prompt: Prompt used for generation
        num_steps: Diffusion sampling steps
        batch_size: Number of generated images per run
        repeats: Number of measured runs
        warmup_runs: Warmup iterations (applied only on first run)
        device: Device string (auto/cuda/mps/cpu)

    Returns:
        InferenceProfileSummary with aggregate statistics and per-run details
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    run_results: list[BenchmarkResult] = []

    for run_idx in range(repeats):
        current_warmup = warmup_runs if run_idx == 0 else 0
        result = benchmark_generation(
            checkpoint=checkpoint,
            vae_checkpoint=vae_checkpoint,
            prompt=prompt,
            num_steps=num_steps,
            batch_size=batch_size,
            warmup_runs=current_warmup,
            device=device,
        )
        run_results.append(result)

    latency_ms = [r.total_time_sec * 1000.0 for r in run_results]
    per_image_sec = [r.time_per_image_sec for r in run_results]
    peak_allocated = [r.peak_memory_mb for r in run_results]
    peak_reserved = [r.peak_memory_reserved_mb for r in run_results]
    peak_ram = [r.peak_ram_mb for r in run_results]
    ram_delta = [r.ram_delta_mb for r in run_results]
    text_ms = [r.text_encoding_ms for r in run_results]
    diffusion_ms = [r.diffusion_sampling_ms for r in run_results]
    vae_ms = [r.vae_decoding_ms for r in run_results]

    std_ms = stdev(latency_ms) if len(latency_ms) > 1 else 0.0

    return InferenceProfileSummary(
        repeats=repeats,
        warmup_runs=warmup_runs,
        num_steps=num_steps,
        batch_size=batch_size,
        prompt=prompt,
        checkpoint=str(checkpoint) if checkpoint is not None else None,
        vae_checkpoint=str(vae_checkpoint) if vae_checkpoint is not None else None,
        device=device,
        latency_mean_ms=mean(latency_ms),
        latency_std_ms=std_ms,
        latency_p50_ms=_percentile(latency_ms, 0.50),
        latency_p95_ms=_percentile(latency_ms, 0.95),
        latency_min_ms=min(latency_ms),
        latency_max_ms=max(latency_ms),
        sec_per_image_mean=mean(per_image_sec),
        sec_per_image_p50=_percentile(per_image_sec, 0.50),
        sec_per_image_p95=_percentile(per_image_sec, 0.95),
        peak_vram_mean_mb=mean(peak_allocated),
        peak_vram_max_mb=max(peak_allocated),
        peak_reserved_max_mb=max(peak_reserved),
        peak_ram_mean_mb=mean(peak_ram),
        peak_ram_max_mb=max(peak_ram),
        ram_delta_mean_mb=mean(ram_delta),
        ram_delta_max_mb=max(ram_delta),
        text_encoding_mean_ms=mean(text_ms),
        diffusion_sampling_mean_ms=mean(diffusion_ms),
        vae_decoding_mean_ms=mean(vae_ms),
        runs=[
            {
                "run_index": idx,
                **result.to_dict(),
            }
            for idx, result in enumerate(run_results, start=1)
        ],
    )


def format_inference_profile(summary: InferenceProfileSummary) -> str:
    """Format inference profiling summary for CLI output."""
    lines = [
        "",
        "=" * 66,
        "  Inference Profile (Latency + Memory)",
        "=" * 66,
        f"  Runs: {summary.repeats} (warmup: {summary.warmup_runs}, only first run)",
        f"  Steps: {summary.num_steps} | Batch: {summary.batch_size} | Device: {summary.device}",
        f"  Prompt: {summary.prompt}",
        "-" * 66,
        f"  Latency mean:      {summary.latency_mean_ms:8.2f} ms",
        f"  Latency std:       {summary.latency_std_ms:8.2f} ms",
        f"  Latency p50/p95:   {summary.latency_p50_ms:8.2f} / {summary.latency_p95_ms:8.2f} ms",
        f"  Latency min/max:   {summary.latency_min_ms:8.2f} / {summary.latency_max_ms:8.2f} ms",
        f"  Speed mean:        {summary.sec_per_image_mean:8.3f} sec/img",
        f"  Speed p50/p95:     {summary.sec_per_image_p50:8.3f} / {summary.sec_per_image_p95:8.3f} sec/img",
        f"  Peak VRAM mean:    {summary.peak_vram_mean_mb:8.2f} MB",
        f"  Peak VRAM max:     {summary.peak_vram_max_mb:8.2f} MB",
        f"  Peak reserved max: {summary.peak_reserved_max_mb:8.2f} MB",
        f"  Peak RAM mean:     {summary.peak_ram_mean_mb:8.2f} MB",
        f"  Peak RAM max:      {summary.peak_ram_max_mb:8.2f} MB",
        f"  RAM delta mean:    {summary.ram_delta_mean_mb:8.2f} MB",
        f"  RAM delta max:     {summary.ram_delta_max_mb:8.2f} MB",
        "-" * 66,
        f"  Stage mean (ms): text={summary.text_encoding_mean_ms:.2f}, "
        f"diffusion={summary.diffusion_sampling_mean_ms:.2f}, "
        f"vae={summary.vae_decoding_mean_ms:.2f}",
        "=" * 66,
    ]
    return "\n".join(lines)


def save_inference_profile(summary: InferenceProfileSummary, output_path: str | Path) -> Path:
    """Save inference profile result as JSON."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(summary.to_dict(), f, indent=2)
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile diffusion inference latency, accelerator memory, and RAM"
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Diffusion checkpoint path")
    parser.add_argument("--vae-checkpoint", type=str, default=None, help="VAE checkpoint path")
    parser.add_argument(
        "--prompt", type=str, default="a cat sitting on a couch", help="Prompt text"
    )
    parser.add_argument("--steps", type=int, default=50, help="Number of diffusion steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--repeats", type=int, default=5, help="Number of measured runs")
    parser.add_argument(
        "--warmup-runs", type=int, default=3, help="Warmup runs before first measure"
    )
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/cuda/mps/cpu")
    parser.add_argument(
        "--save",
        type=str,
        default="results/benchmarks/inference_profile.json",
        help="Path to save JSON profile",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    summary = profile_diffusion_inference(
        checkpoint=args.checkpoint,
        vae_checkpoint=args.vae_checkpoint,
        prompt=args.prompt,
        num_steps=args.steps,
        batch_size=args.batch_size,
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
        device=args.device,
    )

    print(format_inference_profile(summary))
    save_path = save_inference_profile(summary, args.save)
    print(f"Saved profile JSON: {save_path}")


if __name__ == "__main__":
    main()
