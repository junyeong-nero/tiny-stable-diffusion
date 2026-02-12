# 🧠 MMDiT (Multi-Modal Diffusion Transformer)

> The transformer backbone that treats text and images as equal citizens.

---

## 🔬 Overview

**MMDiT** is the core generative engine of `tiny-stable-diffusion`. Departing from traditional convolutional U-Nets, MMDiT is a pure **Transformer** architecture that processes image and text tokens in a unified latent space.

### Core Innovation: Joint Attention
In standard models, image features are conditioned on text via cross-attention. In **Joint Attention**, text tokens and image tokens are concatenated and passed through the same attention layers.
- **Bidirectional Flow**: Images "listen" to text, but text also "responds" to visual features.
- **Alignment**: This leads to superior prompt adherence and more complex semantic understanding.

---

## 🏗 Architecture Details

### 1. Patchification (Image $\rightarrow$ Tokens)
The $8 \times 8 \times 16$ latent grid is transformed into a sequence of tokens:
- **Patch Size**: $2 \times 2$.
- **Sequence Length**: $(8/2) \times (8/2) = 16$ tokens.
- **Embedding**: Each patch is flattened and projected into a $D$-dimensional space (e.g., $D=768$).

### 2. Time & Text Integration
- **Timestep**: Embedded via sinusoidal MLP and injected using **AdaLN-Single** (Adaptive Layer Norm).
- **Text Tokens**: 77 tokens from the CLIP encoder are projected and concatenated with the 16 image tokens, forming a total sequence of 93 tokens.

### 3. The MMDiT Block
Each layer consists of:
- **QK-RMSNorm**: Normalizing Queries and Keys to prevent attention score explosion.
- **Modality-Specific MLPs**: Separate feed-forward paths for text and image streams to preserve their unique feature distributions.

---

## 📏 Scaling Laws (Model Sizes)

| Size | Layers | Dim | Heads | Backbone Params |
| :--- | :--- | :--- | :--- | :--- |
| **S** (Small) | 12 | 384 | 6 | ~87M |
| **B** (Base) | 12 | 768 | 12 | ~187M |
| **L** (Large) | 24 | 1024 | 16 | ~559M |

---

## ⚙️ Configuration Reference

```yaml
diffusion_train:
  model_type: mmdit
  model_size: B        # S, B, or L
  patch_size: 2        # Spatial size of each image token
  qk_rmsnorm: true     # Recommended for stability
```

---

## 📚 Implementation Reference
- **Block Logic**: `src/models/mmdit.py`
- **Attention Layers**: `src/models/layers.py`
- **Patchification**: `src/models/mmdit.py -> MMDiT.unpatchify()`