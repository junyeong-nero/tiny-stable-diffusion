#!/usr/bin/env python3
"""tiny-stable-diffusion - Stable Diffusion 3 from Scratch.

A minimal implementation of Stable Diffusion 3 pipeline for educational purposes.
Implements the complete SD3 training pipeline:
1. VAE training (image compression to latent space)
2. Diffusion training (on latent space)
3. Image generation

Usage:
    python main.py --train-vae        # Stage 1: Train VAE
    python main.py --train-diffusion  # Stage 2: Train Diffusion (requires VAE)
    python main.py --generate         # Generate images
    python main.py --demo             # Interactive demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import get_config, get_training_stage


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="tiny-stable-diffusion - Stable Diffusion 3 from Scratch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode arguments
    parser.add_argument(
        "--train-vae", action="store_true", help="Stage 1: Train VAE (encoder + decoder)"
    )
    parser.add_argument(
        "--train-diffusion", action="store_true", help="Stage 2: Train Diffusion on latent space"
    )
    parser.add_argument("--train", action="store_true", help="Train using config.yaml settings")
    parser.add_argument("--generate", action="store_true", help="Generate images from prompts")
    parser.add_argument(
        "--reconstruct-vae", action="store_true", help="Reconstruct image through VAE"
    )
    parser.add_argument(
        "--decode-random",
        action="store_true",
        help="Generate images from random latent vectors (VAE decoder only)",
    )
    parser.add_argument("--demo", action="store_true", help="Run interactive demo")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate with FID/IS/CLIPScore")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark sampling speed/memory")

    # Training arguments
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset path or name")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--vae-checkpoint", type=str, default=None, help="Path to VAE checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate")

    # Generation arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt for generation (use '||' to separate multiple prompts)",
    )
    parser.add_argument(
        "--input", type=str, default=None, help="Input image path (for --reconstruct-vae)"
    )
    parser.add_argument(
        "--input-dir", type=str, default=None, help="Input directory for batch VAE reconstruction"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory for batch VAE reconstruction"
    )
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--steps", type=int, default=50, help="Number of diffusion steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--scaling-factor", type=float, default=None, help="VAE scaling factor")
    parser.add_argument("--output", type=str, default="results/output.png", help="Output file path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--latent-scale", type=float, default=1.0, help="Scale for random latent vectors"
    )
    parser.add_argument(
        "--reference-dir", type=str, default=None, help="Reference images dir for latent statistics"
    )

    # Evaluation arguments
    parser.add_argument(
        "--eval-metrics",
        type=str,
        default="fid,clip_fid,is,clip_score",
        help="Metrics to compute, comma-separated (fid,clip_fid,is,clip_score)",
    )
    parser.add_argument(
        "--num-eval-samples", type=int, default=None, help="Number of samples for evaluation"
    )
    parser.add_argument(
        "--real-images-dir",
        type=str,
        default=None,
        help="Directory with real images for FID computation",
    )
    parser.add_argument(
        "--eval-prompts-file", type=str, default=None, help="JSON file with evaluation prompts"
    )

    # Benchmark arguments
    parser.add_argument(
        "--benchmark-steps",
        type=str,
        default=None,
        help="Comma-separated step counts for benchmark sweep (e.g., '10,25,50,100')",
    )
    parser.add_argument(
        "--benchmark-batch-sizes",
        type=str,
        default=None,
        help="Comma-separated batch sizes for benchmark sweep (e.g., '1,4,8')",
    )

    # Wandb arguments
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument(
        "--wandb-project", type=str, default="tiny-stable-diffusion", help="Wandb project"
    )
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Wandb run name")

    # HuggingFace Hub arguments
    parser.add_argument(
        "--push-to-hub", action="store_true", help="Push trained model to HuggingFace Hub"
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="HuggingFace model ID (e.g., username/model-name)",
    )
    parser.add_argument(
        "--hub-private", action="store_true", help="Create private repository on HuggingFace Hub"
    )

    args = parser.parse_args()

    if args.train_vae:
        _run_vae_training(args)

    elif args.train_diffusion:
        _run_diffusion_training(args)

    elif args.train:
        stage = get_training_stage()
        if stage == "vae_train":
            _run_vae_training(args)
        else:
            _run_diffusion_training(args)

    elif args.generate:
        _run_generation(args)

    elif args.reconstruct_vae:
        _run_vae_reconstruction(args)

    elif args.decode_random:
        _run_decode_random(args)

    elif args.demo:
        _run_demo(args)

    elif args.evaluate:
        _run_evaluation(args)

    elif args.benchmark:
        _run_benchmark(args)

    else:
        parser.print_help()


def _push_model_to_hub(
    checkpoint_path: str,
    model_type: str,
    hub_model_id: str | None,
    private: bool,
    config: dict,
) -> None:
    """Push trained model to HuggingFace Hub."""
    from src.utils.hf_upload import check_hf_hub_available, push_to_hub

    if not check_hf_hub_available():
        print("Error: huggingface_hub not installed. Install with: pip install huggingface_hub")
        return

    if hub_model_id is None:
        print("Error: --hub-model-id required for --push-to-hub")
        return

    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return

    print(f"\nPushing {model_type} model to HuggingFace Hub...")
    try:
        url = push_to_hub(
            checkpoint_path=checkpoint_path,
            repo_id=hub_model_id,
            model_type=model_type,
            config=config,
            private=private,
        )
        print(f"Model uploaded successfully: {url}")
    except Exception as e:
        print(f"Error uploading to HuggingFace Hub: {e}")


def _run_vae_training(args: argparse.Namespace) -> None:
    """Run VAE training (Stage 1)."""
    from src.training.vae_trainer import train_vae

    config = get_config("vae_train")

    # Override with CLI args
    if args.dataset is not None:
        config["dataset_name"] = args.dataset

    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate

    config["wandb_project"] = args.wandb_project
    config["wandb_run_name"] = args.wandb_run_name or "vae-training"

    if args.resume:
        config["resume"] = True

    train_vae(config, use_wandb=args.wandb)

    # Push to HuggingFace Hub if requested
    if args.push_to_hub:
        _push_model_to_hub(
            checkpoint_path=config["checkpoint_path"],
            model_type="vae",
            hub_model_id=args.hub_model_id,
            private=args.hub_private,
            config=config,
        )


def _run_diffusion_training(args: argparse.Namespace) -> None:
    """Run Diffusion training (Stage 2)."""
    from src.training.trainer import train_diffusion

    config = get_config("diffusion_train")

    # Override with CLI args
    if args.dataset is not None:
        config["dataset_name"] = args.dataset

    if args.vae_checkpoint is not None:
        config["vae_checkpoint"] = args.vae_checkpoint

    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate

    config["wandb_project"] = args.wandb_project
    config["wandb_run_name"] = args.wandb_run_name or "diffusion-training"

    if args.resume:
        config["resume"] = True

    train_diffusion(config, use_wandb=args.wandb)

    # Push to HuggingFace Hub if requested
    if args.push_to_hub:
        _push_model_to_hub(
            checkpoint_path=config["checkpoint_path"],
            model_type="diffusion",
            hub_model_id=args.hub_model_id,
            private=args.hub_private,
            config=config,
        )


def _parse_prompts(prompt_arg: str) -> list[str]:
    """Parse prompt string while allowing commas inside a single prompt.

    Multiple prompts can be passed by separating them with '||'.
    """
    if "||" in prompt_arg:
        prompts = [p.strip() for p in prompt_arg.split("||")]
        return [p for p in prompts if p]
    return [prompt_arg.strip()]


def _resolve_results_path(path_str: str) -> Path:
    """Resolve relative output paths under results/ by default."""
    output_path = Path(path_str)
    if output_path.is_absolute():
        return output_path
    if output_path.parts and output_path.parts[0] == "results":
        return output_path
    return Path("results") / output_path


def _run_generation(args: argparse.Namespace) -> None:
    """Run image generation."""
    from src.inference.generator import generate

    if args.prompt is None:
        print("Error: --prompt required for --generate")
        return

    prompts = _parse_prompts(args.prompt)

    images = generate(
        prompts=prompts,
        checkpoint=args.checkpoint,
        vae_checkpoint=args.vae_checkpoint,
        num_samples=args.num_samples,
        num_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        scaling_factor=args.scaling_factor,
    )

    for i, img in enumerate(images):
        if len(prompts) > 1 or args.num_samples > 1:
            output_path = Path("results") / f"output_{i}.png"
        else:
            output_path = _resolve_results_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        print(f"Saved: {output_path}")


def _run_vae_reconstruction(args: argparse.Namespace) -> None:
    """Run VAE reconstruction."""
    from src.inference.vae_inference import reconstruct_vae, reconstruct_vae_batch

    # Batch mode
    if args.input_dir is not None:
        output_dir = args.output_dir or "results/inference/vae/reconstructed"
        output_dir = str(_resolve_results_path(output_dir))
        reconstruct_vae_batch(
            input_dir=args.input_dir,
            output_dir=output_dir,
            checkpoint=args.vae_checkpoint,
        )
        return

    # Single image mode
    if args.input is None:
        print("Error: --input or --input-dir required for --reconstruct-vae")
        return

    output_path = _resolve_results_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reconstruct_vae(
        input_path=args.input,
        output_path=str(output_path),
        checkpoint=args.vae_checkpoint,
    )


def _run_decode_random(args: argparse.Namespace) -> None:
    """Generate images from random latent vectors using VAE decoder only."""
    from src.inference.vae_inference import decode_random_latent

    output_path = _resolve_results_path(args.output)

    decode_random_latent(
        output_path=output_path,
        checkpoint=args.vae_checkpoint,
        num_samples=args.num_samples,
        seed=args.seed,
        latent_scale=args.latent_scale,
        reference_dir=args.reference_dir,
    )


def _run_demo(args: argparse.Namespace) -> None:
    """Run interactive demo."""
    from src.inference.generator import demo

    demo(
        checkpoint=args.checkpoint,
        vae_checkpoint=args.vae_checkpoint,
    )


def _run_evaluation(args: argparse.Namespace) -> None:
    """Run quantitative evaluation (FID/IS/CLIPScore)."""
    import json

    from src.config import get_config
    from src.evaluation.diffusion_evaluator import (
        evaluate_diffusion,
        format_evaluation_results,
    )

    # Load eval config defaults
    try:
        config = get_config("evaluation")
    except Exception:
        config = {}

    # Load diffusion_train config for dataset defaults
    try:
        diffusion_config = get_config("diffusion_train")
    except Exception:
        diffusion_config = {}

    metrics = args.eval_metrics.split(",")
    num_eval_samples = (
        args.num_eval_samples
        if args.num_eval_samples is not None
        else config.get("num_eval_samples", 1000)
    )
    eval_batch_size = config.get("eval_batch_size", 32)
    eval_num_steps = args.steps  # argparse default: 50
    eval_guidance = args.guidance  # argparse default: 7.5
    eval_seed = args.seed if args.seed is not None else config.get("eval_seed", 42)
    results_dir = config.get("results_dir", "results/evaluation/diffusion")
    save_path = Path(results_dir) / "eval_results.json"

    # Dataset: use --dataset or fall back to diffusion_train.dataset_name
    dataset_name = args.dataset or diffusion_config.get("dataset_name")
    image_field = diffusion_config.get("image_field", "jpg")
    caption_field = diffusion_config.get("caption_field", "txt")
    use_webdataset = diffusion_config.get("use_webdataset", False)

    # Load prompts
    prompts = None
    if args.eval_prompts_file is not None:
        with open(args.eval_prompts_file) as f:
            prompts = json.load(f)
    elif args.prompt is not None:
        prompts = _parse_prompts(args.prompt)

    result = evaluate_diffusion(
        checkpoint=args.checkpoint,
        vae_checkpoint=args.vae_checkpoint,
        prompts=prompts,
        real_images_dir=args.real_images_dir,
        dataset_name=dataset_name,
        num_eval_samples=num_eval_samples,
        num_steps=eval_num_steps,
        guidance_scale=eval_guidance,
        metrics=metrics,
        seed=eval_seed,
        eval_batch_size=eval_batch_size,
        save_results=str(save_path),
        image_field=image_field,
        caption_field=caption_field,
        use_webdataset=use_webdataset,
    )

    print(format_evaluation_results(result))


def _run_benchmark(args: argparse.Namespace) -> None:
    """Run sampling speed/memory benchmark."""
    import json

    from src.config import get_config
    from src.evaluation.benchmark import (
        benchmark_generation,
        benchmark_sweep,
        format_benchmark_results,
    )

    # Load benchmark config defaults
    try:
        config = get_config("benchmark")
    except Exception:
        config = {}

    warmup_runs = config.get("warmup_runs", 3)
    prompt = config.get("benchmark_prompt", "a cat sitting on a couch")
    results_dir = config.get("results_dir", "results/benchmarks")

    if args.prompt is not None:
        prompt = args.prompt.split("||")[0].strip()

    # Determine if sweep or single run
    results_data: list[dict]

    if args.benchmark_steps is not None or args.benchmark_batch_sizes is not None:
        # Sweep mode
        steps_list = (
            [int(s) for s in args.benchmark_steps.split(",")]
            if args.benchmark_steps
            else config.get("num_steps_list", [10, 25, 50, 100])
        )
        batch_sizes = (
            [int(b) for b in args.benchmark_batch_sizes.split(",")]
            if args.benchmark_batch_sizes
            else config.get("batch_sizes", [1, 4, 8])
        )

        results = benchmark_sweep(
            checkpoint=args.checkpoint,
            vae_checkpoint=args.vae_checkpoint,
            prompt=prompt,
            num_steps_list=steps_list,
            batch_sizes=batch_sizes,
            warmup_runs=warmup_runs,
        )

        # Find a good detail result (middle steps, batch=1)
        detail = None
        for r in results:
            if r.batch_size == 1:
                detail = r
        if detail is None and results:
            detail = results[0]

        print(format_benchmark_results(results, detail_result=detail))
        results_data = [r.to_dict() for r in results]
    else:
        # Single run
        result = benchmark_generation(
            checkpoint=args.checkpoint,
            vae_checkpoint=args.vae_checkpoint,
            prompt=prompt,
            num_steps=args.steps,
            batch_size=args.batch_size or 1,
            warmup_runs=warmup_runs,
        )

        print(format_benchmark_results([result], detail_result=result))
        results_data = [result.to_dict()]

    # Save results
    save_path = Path(results_dir) / "benchmark_results.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
