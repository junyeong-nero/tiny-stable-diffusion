# Training Guide

Comprehensive guide for training text-to-emoji models.

## Table of Contents

- [Quick Start](#quick-start)
- [Hyperparameters](#hyperparameters)
- [Training Configurations](#training-configurations)
- [Hardware Requirements](#hardware-requirements)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Basic Training

```bash
./scripts/train.sh --epochs 100 --batch-size 64 --model-size S
```

### Training with Weights & Biases

```bash
./scripts/train.sh --epochs 100 --batch-size 64 --wandb
```

### Mixed Precision Training

```bash
./scripts/train.sh --mixed-precision --amp-dtype float16
```

## Hyperparameters

### Model Configuration

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `--model-size` | S | S, B, L, XL | DiT model size |
| `--patch-size` | 2 | 1, 2, 4 | Patch tokenization size |
| `--image-size` | 32 | 32, 64, 128 | Input image resolution |

**Model Size Details:**

| Size | Layers | Hidden | Heads | Parameters | VRAM (BS=8) |
|------|--------|--------|-------|------------|-------------|
| S | 12 | 384 | 6 | ~30M | ~16GB |
| B | 12 | 768 | 12 | ~130M | ~32GB |
| L | 24 | 1024 | 16 | ~300M | ~64GB |

### Training Configuration

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--epochs` | 100 | 50-500 | Number of training epochs |
| `--batch-size` | 64 | 8-128 | Batch size per GPU |
| `--learning-rate` | 5e-4 | 1e-5 to 1e-3 | Peak learning rate |
| `--warmup-steps` | 100 | 50-1000 | LR warmup steps |
| `--gradient-clip-val` | 1.0 | 0.5-2.0 | Gradient clipping threshold |
| `--weight-decay` | 0.01 | 0-0.1 | AdamW weight decay |

### Diffusion Configuration

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `--num-timesteps` | 1000 | 500-1000 | Diffusion timesteps |
| `--beta-schedule` | cosine | linear, cosine | Noise schedule |
| `--guidance-scale` | 7.5 | 1.0-15.0 | CFG guidance scale |
| `--cfg-probability` | 0.1 | 0.05-0.2 | Unconditional dropout rate |

### EMA Configuration

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--use-ema` | True | - | Enable EMA for stable weights |
| `--ema-decay` | 0.9999 | 0.995-0.9999 | EMA decay rate |

### Validation Configuration

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--validation-interval` | 5 | 1-20 | Generate samples every N epochs |

### Mixed Precision Training

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `--mixed-precision` | False | - | Enable AMP training |
| `--amp-dtype` | float16 | float16, bfloat16 | AMP precision type |

## Training Configurations

### Recommended Configurations

#### Small Model (DiT-S) - Quick Experiments

```bash
./scripts/train.sh \
    --model-size S \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 5e-4 \
    --mixed-precision \
    --amp-dtype float16
```

**Best for:**
- Initial experiments and prototyping
- Limited GPU memory (< 16GB)
- Fast iteration cycles

**Expected performance:**
- Training time: ~2-3 hours (RTX 3090)
- Quality: Good for 32×32 pixel art
- Convergence: ~50-80 epochs

#### Medium Model (DiT-B) - Production Quality

```bash
./scripts/train.sh \
    --model-size B \
    --epochs 200 \
    --batch-size 32 \
    --learning-rate 3e-4 \
    --warmup-steps 500 \
    --mixed-precision \
    --amp-dtype float16 \
    --wandb
```

**Best for:**
- Production deployments
- High-quality outputs
- GPU memory: 24-32GB

**Expected performance:**
- Training time: ~8-10 hours (RTX 3090)
- Quality: Excellent detail and prompt adherence
- Convergence: ~100-150 epochs

#### Large Model (DiT-L) - Maximum Quality

```bash
./scripts/train.sh \
    --model-size L \
    --epochs 300 \
    --batch-size 16 \
    --learning-rate 2e-4 \
    --warmup-steps 1000 \
    --mixed-precision \
    --amp-dtype bfloat16 \
    --wandb
```

**Best for:**
- Research and highest quality outputs
- Multi-GPU setups (A100 40GB+)
- Requires: 64GB+ VRAM

**Expected performance:**
- Training time: ~24-36 hours (A100)
- Quality: State-of-the-art
- Convergence: ~150-250 epochs

### Fine-tuning Configuration

Fine-tune a pre-trained model on custom data:

```bash
./scripts/train.sh \
    --resume checkpoints/model_best.pt \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --warmup-steps 100 \
    --dataset-name your-username/custom-emoji-dataset
```

**Tips for fine-tuning:**
- Use 1/5 to 1/10 of original learning rate
- Shorter warmup period
- Monitor for overfitting on small datasets
- Consider using EMA weights from checkpoint

## Hardware Requirements

### GPU Memory Requirements

| Model Size | Batch Size | Mixed Precision | VRAM Required |
|------------|------------|-----------------|---------------|
| DiT-S | 64 | No | ~24GB |
| DiT-S | 64 | Yes (FP16) | ~12GB |
| DiT-S | 32 | Yes (FP16) | ~8GB |
| DiT-B | 32 | No | ~40GB |
| DiT-B | 32 | Yes (FP16) | ~20GB |
| DiT-B | 16 | Yes (FP16) | ~12GB |
| DiT-L | 16 | Yes (BF16) | ~40GB |
| DiT-L | 8 | Yes (BF16) | ~24GB |

### CPU Requirements

- **Minimum**: 8 cores, 16GB RAM
- **Recommended**: 16+ cores, 32GB+ RAM
- DataLoader workers: 4-8 workers per GPU

### Storage Requirements

- **Dataset**: ~2GB (junyeong-nero/emoji-32)
- **Checkpoints**: ~500MB per checkpoint (DiT-S)
- **Validation samples**: ~100MB per epoch
- **Total recommended**: 50GB+ free space

## Best Practices

### Learning Rate Scheduling

The default training uses cosine annealing with linear warmup:

```python
# Warmup phase (0 to warmup_steps)
lr = lr_max * (step / warmup_steps)

# Cosine decay phase (warmup_steps to total_steps)
lr = 0.5 * lr_max * (1 + cos(π * progress))
```

**Recommendations:**
- Start with `lr=5e-4` for DiT-S
- Reduce by 2x for each model size increase
- Warmup: 100 steps for small datasets (< 5k samples)
- Warmup: 500-1000 steps for large datasets (> 10k samples)

### Batch Size Selection

**Rule of thumb:**
```
effective_batch_size = batch_size × num_gpus × gradient_accumulation
```

**Recommendations:**
- Target effective batch size: 128-256
- Larger batches → more stable gradients
- Smaller batches → better generalization
- Adjust learning rate with batch size: `lr_new = lr_base × sqrt(bs_new / bs_base)`

### EMA Configuration

**EMA formula:**
```
θ_ema = decay × θ_ema + (1 - decay) × θ
```

**Decay recommendations:**
- Small datasets (< 5k): 0.999
- Medium datasets (5k-20k): 0.9995
- Large datasets (> 20k): 0.9999

**When to use EMA:**
- ✅ Always enable for production models
- ✅ Smoother convergence and better final quality
- ✅ Negligible computational overhead
- ❌ Disable only for quick experiments

### Monitoring Training

**Key metrics to watch:**

1. **Training loss**: Should decrease smoothly
   - Typical final loss: 0.01-0.05
   - Spiky loss → reduce learning rate or increase batch size

2. **Learning rate**: Check warmup and decay
   - Use `--wandb` to visualize LR schedule

3. **Validation samples**: Visual quality check
   - Generated every `--validation-interval` epochs
   - Compare regular model vs EMA model
   - Look for: detail, prompt adherence, artifacts

4. **Gradient norms**: Should stay bounded
   - Enable gradient clipping if exploding
   - Typical range: 0.1-2.0

### Early Stopping

While not implemented by default, monitor these signals:

**Stop training if:**
- Validation quality plateaus for 20+ epochs
- Training loss < 0.01 and quality is acceptable
- Overfitting detected (validation samples degrade)

**Continue training if:**
- Validation samples still improving
- Loss still decreasing
- Want higher prompt adherence (train longer with CFG)

## Troubleshooting

### Out of Memory (OOM)

**Solutions:**
1. Enable mixed precision: `--mixed-precision`
2. Reduce batch size: `--batch-size 32` → `--batch-size 16`
3. Use smaller model: `--model-size B` → `--model-size S`
4. Reduce image size (if using > 32): `--image-size 32`

### Loss Not Decreasing

**Checklist:**
- [ ] Check learning rate (try reducing by 2-5x)
- [ ] Increase warmup steps
- [ ] Verify dataset is loading correctly
- [ ] Check for NaN gradients (reduce LR)
- [ ] Try different beta schedule: `--beta-schedule linear`

### Poor Generation Quality

**Potential causes:**

1. **Undertraining**
   - Solution: Train longer (100+ epochs)

2. **Low guidance scale**
   - Solution: Use `--guidance-scale 7.5` or higher at inference

3. **High CFG dropout**
   - Solution: Reduce `--cfg-probability` to 0.05-0.1

4. **Wrong EMA weights**
   - Solution: Use EMA checkpoint for inference

### Slow Training

**Optimizations:**

1. **Enable mixed precision**
   ```bash
   --mixed-precision --amp-dtype float16
   ```

2. **Increase DataLoader workers**
   ```bash
   --num-workers 8
   ```

3. **Enable pin memory**
   ```bash
   --pin-memory
   ```

4. **Use faster data format**
   - Pre-process dataset to tensors
   - Use local SSD instead of network storage

5. **Reduce validation frequency**
   ```bash
   --validation-interval 10
   ```

### Gradient Explosion

**Symptoms:**
- Loss suddenly spikes to NaN
- Gradients > 100

**Solutions:**
1. Enable gradient clipping: `--gradient-clip-val 1.0`
2. Reduce learning rate by 5-10x
3. Increase warmup steps
4. Check for data issues (corrupted images)

## Advanced Topics

### Multi-GPU Training

(Coming soon - DDP support planned)

```bash
# Future support
torchrun --nproc_per_node=4 src/training/train.py \
    --batch-size 32 \
    --model-size B
```

### Gradient Checkpointing

(Coming soon - memory optimization planned)

Reduces memory usage by ~40% at the cost of ~20% slower training.

### Custom Datasets

Use your own dataset:

```bash
./scripts/train.sh \
    --data-source huggingface \
    --dataset-name your-username/your-dataset \
    --split train
```

**Dataset requirements:**
- Images: 32×32 RGB
- Format: Hugging Face dataset with 'image' and 'caption' columns
- Minimum samples: 1000+ for decent quality

## Performance Benchmarks

### Training Speed (DiT-S, Batch Size 64, FP16)

| GPU | Throughput (imgs/sec) | Epoch Time | 100 Epoch Total |
|-----|----------------------|------------|-----------------|
| RTX 3090 | ~250 | ~1.5 min | ~2.5 hours |
| RTX 4090 | ~400 | ~1 min | ~1.7 hours |
| A100 40GB | ~600 | ~40 sec | ~1.1 hours |
| H100 | ~1000 | ~25 sec | ~0.7 hours |

*Assumes ~4000 sample dataset*

### Quality vs Training Time

| Epochs | Training Time (RTX 3090) | Quality Level |
|--------|-------------------------|---------------|
| 20 | ~30 min | Basic shapes, poor details |
| 50 | ~1.25 hours | Recognizable objects |
| 100 | ~2.5 hours | Good quality, good prompt adherence |
| 200 | ~5 hours | Excellent quality |
| 300+ | ~7.5+ hours | Diminishing returns |

## Reproducibility

### Seeding

All random operations are seeded:

```bash
./scripts/train.sh --seed 42
```

This ensures:
- Same weight initialization
- Same data shuffling
- Same sampling order
- Reproducible validation samples

**Note:** Minor variations may occur across different hardware/drivers.

### Saving Training State

Checkpoints include:
- Model weights
- Optimizer state
- Scheduler state
- EMA state
- Training epoch
- Random state (planned)

Resume training:
```bash
./scripts/train.sh --resume checkpoints/checkpoint_epoch_50.pt
```

## References

- [DiT Paper](https://arxiv.org/abs/2212.09748)
- [DDPM Paper](https://arxiv.org/abs/2006.11239)
- [DDIM Paper](https://arxiv.org/abs/2010.02502)
- [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598)
