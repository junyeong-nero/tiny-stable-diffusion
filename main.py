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
    parser.add_argument("--train-vae", action="store_true", help="Stage 1: Train VAE (encoder + decoder)")
    parser.add_argument("--train-diffusion", action="store_true", help="Stage 2: Train Diffusion on latent space")
    parser.add_argument("--train", action="store_true", help="Train using config.yaml settings")
    parser.add_argument("--generate", action="store_true", help="Generate images from prompts")
    parser.add_argument("--demo", action="store_true", help="Run interactive demo")

    # Training arguments
    parser.add_argument("--dataset", type=str, default=None, help="Dataset path or name")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--vae-checkpoint", type=str, default=None, help="Path to VAE checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate")

    # Generation arguments
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for generation")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--steps", type=int, default=50, help="Number of diffusion steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--output", type=str, default="output.png", help="Output file path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    # Wandb arguments
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="tiny-stable-diffusion", help="Wandb project")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Wandb run name")

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

    elif args.demo:
        _run_demo(args)

    else:
        parser.print_help()


def _run_vae_training(args: argparse.Namespace) -> None:
    """Run VAE training (Stage 1)."""
    from src.training.vae_trainer import train_vae

    config = get_config("vae_train")

    # Override with CLI args
    if args.dataset is not None:
        if Path(args.dataset).exists():
            config["data_source"] = "local"
            config["local_dataset_path"] = args.dataset
        else:
            config["data_source"] = "caption"
            config["dataset_name"] = args.dataset

    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate

    config["wandb_project"] = args.wandb_project
    config["wandb_run_name"] = args.wandb_run_name or "vae-training"

    train_vae(config, use_wandb=args.wandb)


def _run_diffusion_training(args: argparse.Namespace) -> None:
    """Run Diffusion training (Stage 2)."""
    from src.training.trainer import train_diffusion

    config = get_config("diffusion_train")

    # Override with CLI args
    if args.dataset is not None:
        if Path(args.dataset).exists():
            config["data_source"] = "local"
            config["local_dataset_path"] = args.dataset
        else:
            config["data_source"] = "caption"
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

    train_diffusion(config, use_wandb=args.wandb)


def _run_generation(args: argparse.Namespace) -> None:
    """Run image generation."""
    from src.inference.generator import generate

    if args.prompt is None:
        print("Error: --prompt required for --generate")
        return

    prompts = [p.strip() for p in args.prompt.split(",")]

    images = generate(
        prompts=prompts,
        checkpoint=args.checkpoint,
        vae_checkpoint=args.vae_checkpoint,
        num_samples=args.num_samples,
        num_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
    )

    for i, img in enumerate(images):
        if len(prompts) > 1 or args.num_samples > 1:
            output_path = f"output_{i}.png"
        else:
            output_path = args.output
        img.save(output_path)
        print(f"Saved: {output_path}")


def _run_demo(args: argparse.Namespace) -> None:
    """Run interactive demo."""
    from src.inference.generator import demo

    demo(
        checkpoint=args.checkpoint,
        vae_checkpoint=args.vae_checkpoint,
    )


if __name__ == "__main__":
    main()
