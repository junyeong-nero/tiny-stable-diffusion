# MMDiT (Multi-Modal Diffusion Transformer) Documentation

> Detailed guide to the MMDiT architecture used in tiny-stable-diffusion, based on Stable Diffusion 3.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Key Components](#key-components)
4. [Comparison with Vanilla DiT](#comparison-with-vanilla-dit)
5. [Implementation Details](#implementation-details)
6. [References](#references)

---

## Overview

**MMDiT (Multi-Modal Diffusion Transformer)** is the core backbone of Stable Diffusion 3. Unlike previous models (SD1.5, SDXL) that used U-Nets with Cross-Attention, or Vanilla DiT that treated text as a simple condition, MMDiT treats **text and image as equal modalities**.

### Core Concept: Joint Attention

Instead of `Image -> CrossAttention(Text) -> Image`, MMDiT processes both text and image tokens in a single sequence (or separate sequences that interact), allowing **bi-directional information flow**.

- **SD1.5 / SDXL**: Image attends to Text (Cross-Attention). Text does NOT attend to Image.
- **MMDiT**: Image attends to Text. Text attends to Image.

This leads to significantly better text comprehension and typography generation.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                               MMDiT Block                                │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│      Image Tokens (N×D)                  Text Tokens (M×D)               │
│             │                                   │                        │
│             ▼                                   ▼                        │
│      ┌─────────────┐                     ┌─────────────┐                 │
│      │  Layernorm  │                     │  Layernorm  │                 │
│      └──────┬──────┘                     └──────┬──────┘                 │
│             │                                   │                        │
│             ▼                                   ▼                        │
│      ┌─────────────────────────────────────────────────┐                 │
│      │                 Joint Attention                 │                 │
│      │    [ Image_Q | Text_Q ] @ [ Image_K | Text_K ]T │                 │
│      └──────┬─────────────────────┬────────────────────┘                 │
│             │                     │                                      │
│             ▼                     ▼                                      │
│      ┌─────────────┐       ┌─────────────┐                               │
│      │     MLP     │       │     MLP     │                               │
│      └──────┬──────┘       └──────┬──────┘                               │
│             │                     │                                      │
│             ▼                     ▼                                      │
│         New Image             New Text                                   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1.  **Input Processing**:
    *   **Image**: Latent patches (8x8) $\rightarrow$ Flatten $\rightarrow$ Linear Projection $\rightarrow$ Positional Embedding.
    *   **Text**: CLIP embeddings $\rightarrow$ Linear Projection.
2.  **Timestep Modulation**:
    *   Time $t$ is embedded and modulated (AdaLN) into every layer.
3.  **Joint Transformer Blocks**:
    *   Processes combined sequences.
4.  **Unpatchify**:
    *   Image tokens are rearranged back into an 8x8 spatial grid.

---

## Key Components

### 1. Joint Attention Mechanism

The critical innovation in MMDiT.

*   **Q, K, V Projection**: Separate projections for Image and Text.
*   **Concatenation**: Queries, Keys, and Values are concatenated along the sequence dimension.
    *   $Q_{total} = Concat(Q_{img}, Q_{txt})$
    *   $K_{total} = Concat(K_{img}, K_{txt})$
    *   $V_{total} = Concat(V_{img}, V_{txt})$
*   **Attention**: Standard Self-Attention on the concatenated sequence.
*   **Split**: Output is split back into Image and Text streams.

This allows the model to:
1.  Refine text understanding based on the current image state.
2.  Generate image features perfectly aligned with specific text tokens.

### 2. QK-RMSNorm

To stabilize training at scale (especially with fp16/bf16), MMDiT applies **RMSNorm** to the Queries (Q) and Keys (K) before the attention dot product.

$$Attention(Q, K, V) = Softmax(\frac{RMSNorm(Q) \cdot RMSNorm(K)^T}{\sqrt{d}}) V$$

This prevents attention scores from growing too large, a common instability source in large ViTs.

### 3. Register Tokens (Optional)

We support adding **Register Tokens** (from ViT-Resisters paper). These are learnable tokens appended to the sequence that act as "global storage" or "sinks" for information, reducing artifacts in attention maps.

---

## Comparison with Vanilla DiT

| Feature | Vanilla DiT | MMDiT (Ours/SD3) |
| :--- | :--- | :--- |
| **Conditioning** | adaptive Layer Norm (adaLN) + Cross-Attention | Joint Attention |
| **Text Encoder** | Fixed (usually class label or simple embed) | Trainable context (via Joint Attn) |
| **Modality Interaction** | One-way (Text $\rightarrow$ Image) | Two-way (Text $\leftrightarrow$ Image) |
| **Parameter Efficiency** | High (shared weights) | Lower (separate weights for modalities) |
| **Performance** | Good for class-cond | Superior for text-to-image |

---

## Implementation Details

### Model Configuration (`config.yaml`)

```yaml
diffusion_train:
  model_type: mmdit
  model_size: B        # S, B, L, XL
  qk_rmsnorm: true     # Recommended for stability
  register_tokens: 0   # Optional
```

### Parameter Counts (Approx.)

| Size | Layers | Width | Heads | Params |
| :--- | :---: | :---: | :---: | :---: |
| **S** (Small) | 12 | 384 | 6 | 87M |
| **B** (Base) | 12 | 768 | 12 | 187M |
| **L** (Large) | 24 | 1024 | 16 | 559M |
| **XL** (X-Large)| 28 | 1152 | 16 | 780M |

### Code Structure (`src/models/mmdit.py`)

The implementation relies on `src.models.mmdit.MMDiT` which wraps a PyTorch module.

```python
class MMDiT(nn.Module):
    def __init__(self, ...):
        # 1. Patch Embeddings
        self.patch_embed = PatchEmbed(...)
        
        # 2. Main Transformer Backbone
        self.mmdit = MMDitModel(...)
        
        # 3. Final Decoding
        self.final_layer = FinalLayer(...)

    def forward(self, x, t, text_embeds):
        # x: (B, C, H, W)
        # t: (B,)
        # text_embeds: (B, D_clip)
        
        # ... processing ...
        return noise_pred
```

---

## References

*   **Scaling Rectified Flow Transformers for High-Resolution Image Synthesis** (SD3 Paper): [arXiv:2403.03206](https://arxiv.org/abs/2403.03206)
*   **Fast and High-Quality Image Generation with Efficient Multi-Modal Transformer**: Explains the Joint Attention mechanism details.

```