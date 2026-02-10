# MMDiT (Multi-Modal Diffusion Transformer)

> The multi-modal backbone for Stable Diffusion 3.

---

## 🔬 Overview

**MMDiT** is the core neural network of the `tiny-stable-diffusion` system. Unlike previous architectures (like SD 1.5/2.1) that relied on convolutional U-Nets, MMDiT is entirely based on the **Transformer** architecture. 

The "Multi-Modal" aspect refers to its ability to treat **Text tokens** and **Image tokens** as equal citizens in a shared latent space.

### Core Innovation: Joint Attention
In traditional cross-attention, image features "look" at text features. In **Joint Attention**, text and image features interact bi-directionally:
- **Image** $\rightarrow$ **Text**
- **Text** $\rightarrow$ **Image**

This allows the model to refine its understanding of the text prompt based on the visual features it is generating, leading to much better prompt adherence.

---

## 🏗 Architecture Details

### 1. Tokenization (Patchification)
- The $8 \times 8 \times 16$ latent image is divided into $2 \times 2$ patches.
- Each patch is flattened and projected into a $D$-dimensional embedding (e.g., $D=768$ for the Base model).
- This results in a sequence of $4 \times 4 = 16$ image tokens.

### 2. Timestep & Text Embedding
- **Timestep**: Embedded via a sinusoidal MLP and added using **AdaLN-Single** modulation.
- **Text**: Pre-encoded CLIP embeddings are projected and concatenated with the image tokens.

### 3. The MMDiT Block
Each block consists of:
- **QK-RMSNorm**: Normalizing queries and keys before attention to stabilize training.
- **Joint Attention**: Both modalities attend to each other in a unified sequence.
- **Modality-Specific MLPs**: Separate feed-forward networks for image and text streams to preserve their unique characteristics.

---

## 📏 Model Sizes

We offer four configurations based on the SD3 scaling laws:

| Size | Layers | Hidden Dim | Heads | Approx. Params |
| :--- | :--- | :--- | :--- | :--- |
| **S** (Small) | 12 | 384 | 6 | ~87M |
| **B** (Base) | 12 | 768 | 12 | ~187M |
| **L** (Large) | 24 | 1024 | 16 | ~559M |
| **XL** (X-Large)| 28 | 1152 | 16 | ~780M |

---

## ⚙️ Configuration

Key settings in `config.yaml`:
```yaml
diffusion_train:
  model_type: mmdit
  model_size: B        # S, B, L, XL
  qk_rmsnorm: true     # Stabilizes training
  patch_size: 2
```

---

## 📚 Implementation Reference
- **Model Definition**: `src/models/mmdit.py`
- **Attention Logic**: `src/models/layers.py`
- **Training Logic**: `src/training/trainer.py`
