# 📚 Dataset Management Guide

> How to source, configure, and stream data for `tiny-stable-diffusion`.

---

## 🏗 Supported Datasets

We leverage the **Hugging Face Hub** for seamless data integration. While the model is "tiny," it thrives on diverse and well-captioned data.

### 1. LAION-300k (The VAE Specialist)
*   **ID**: `hmu013/LAION-300k`
*   **Role**: Stage 1 (VAE Training).
*   **Context**: VAEs need to learn "how to see" (textures, edges, colors). LAION provides the massive visual diversity required for robust image compression.
*   **Usage**: **Streaming is highly recommended** to avoid 100GB+ downloads.

### 2. Oxford Pets Enriched (The Diffusion Specialist)
*   **ID**: `visual-layer/oxford-iiit-pet-vl-enriched`
*   **Role**: Stage 2 (Diffusion Training).
*   **Context**: For a 200M parameter model, focusing on a specific domain (like animals) produces significantly higher quality results than a broad "everything" model.
*   **Features**: Includes high-quality images and detailed, descriptive captions.

### 3. Pokemon BLIP (Rapid Prototyping)
*   **ID**: `reach-vb/pokemon-blip-captions`
*   **Role**: Debugging and "Hello World" runs.
*   **Size**: ~800 images. Perfect for verifying your setup in minutes.

---

## ⚙️ Configuration Patterns

Datasets are configured in `config.yaml`. Here are the recommended patterns:

### VAE Training (Streaming Mode)
```yaml
vae_train:
  dataset_name: hmu013/LAION-300k
  image_field: png
  stream: true           # Recommended: Doesn't store files on disk
  batch_size: 256
```

### Diffusion Training (Local Mode)
```yaml
diffusion_train:
  dataset_name: visual-layer/oxford-iiit-pet-vl-enriched
  image_field: image
  caption_field: caption_enriched
  stream: false          # Recommended: Download once for faster repeated access
```

---

## 🔄 Data Access Strategy

| Mode | Pros | Cons | Best For |
| :--- | :--- | :--- | :--- |
| **Streaming** | No disk space needed; Instant start. | High network load; Potential latency. | LAION, COCO, Large sets. |
| **Local** | Offline capable; Maximum speed. | Requires disk space (1GB - 50GB). | Specific domains (Pets, Pokemon). |

---

## 🛠 Adding Your Own Data

### Option A: Hugging Face (Easiest)
1.  Upload your dataset to Hugging Face.
2.  Specify the `dataset_name`, `image_field`, and `caption_field` in `config.yaml`.
3.  The trainer will automatically handle the loading.

### Option B: Local Image Folder
You can point to a local directory containing images. For captions, use a `metadata.jsonl` file in the same directory.
```yaml
dataset_name: "/Users/name/data/my_photos"
image_field: "image"
caption_field: "text"
```

---

## 📝 Tips for Quality Data
- **Resolution**: All images are automatically resized and center-cropped to $64 \times 64$.
- **Aspect Ratio**: For best results, use source images that are relatively square.
- **Captions**: Detailed, descriptive captions (e.g., *"a fluffy white cat sleeping on a blue velvet sofa"*) perform much better than single words (*"cat"*).

---
*Reference Implementation: `src/data/loader.py`*
