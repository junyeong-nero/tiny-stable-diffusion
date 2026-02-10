"""Evaluation tools for VAE and diffusion models."""

from .vae_evaluator import evaluate_vae, evaluate_vae_on_dataset

__all__ = [
    "evaluate_vae",
    "evaluate_vae_on_dataset",
]


# Lazy imports for optional evaluation dependencies
def __getattr__(name: str):
    if name == "evaluate_diffusion":
        from .diffusion_evaluator import evaluate_diffusion

        return evaluate_diffusion
    if name == "benchmark_generation":
        from .benchmark import benchmark_generation

        return benchmark_generation
    if name == "benchmark_sweep":
        from .benchmark import benchmark_sweep

        return benchmark_sweep
    if name == "profile_diffusion_inference":
        from .inference_profile import profile_diffusion_inference

        return profile_diffusion_inference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
