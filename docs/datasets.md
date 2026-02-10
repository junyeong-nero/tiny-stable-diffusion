# Dataset Guide

> A guide to the datasets used for training and evaluating `tiny-stable-diffusion`.

---

## 📚 Supported Datasets

We primarily leverage datasets from the **Hugging Face Hub** for ease of use and reproducibility.

### 1. LAION-300k (Recommended for VAE)
*   **ID**: `hmu013/LAION-300k`
*   **Purpose**: VAE Training (Stage 1).
*   **Why**: VAEs need to learn general image statistics (edges, textures, colors). A diverse, large-scale dataset like LAION is ideal for this.
*   **Usage**: Usually streamed due to size.

### 2. Oxford Pets Enriched (Recommended for Diffusion)
*   **ID**: `visual-layer/oxford-iiit-pet-vl-enriched`
*   **Purpose**: Diffusion Training (Stage 2).
*   **Why**: For a "tiny" model, training on a specific domain (like pets) yields significantly better visual results than attempting to learn a general domain with limited capacity.
*   **Features**: High-quality images with detailed, enriched captions.

### 3. Pokemon BLIP Captions (Fast Testing)
*   **ID**: `reach-vb/pokemon-blip-captions`
*   **Purpose**: Fast debugging or "Hello World" training.
*   **Size**: ~800 images.

---

## ⚙️ Configuration

Datasets are configured in `config.yaml` under their respective training sections.

### VAE Configuration (Streaming)
```yaml
vae_train:
  dataset_name: hmu013/LAION-300k
  image_field: png       # Column name containing the image
  stream: true           # Enable streaming (don't download whole 300k)
  batch_size: 256
```

### Diffusion Configuration (Local)
```yaml
diffusion_train:
  dataset_name: visual-layer/oxford-iiit-pet-vl-enriched
  image_field: image
  caption_field: caption_enriched
  stream: false          # Small enough to download locally for speed
```

---

## 🔄 Streaming vs. Local

| Mode | **Streaming** (`stream: true`) | **Local** (`stream: false`) |
| :--- | :--- | :--- |
| **Storage** | Zero disk space required. | Requires local disk space. |
| **Speed** | Initial start is instant. | Initial download takes time. |
| **Network** | Constant internet required. | Offline after initial download. |
| **Best For** | Massive datasets (LAION, COCO). | Small/Medium datasets (<10GB). |

---

## 🛠 Adding Custom Datasets

### 1. Hugging Face Method (Preferred)
1.  Upload your dataset to Hugging Face (e.g., `my-username/my-dataset`).
2.  Ensure it contains an image column and a text column (for diffusion).
3.  Update `config.yaml`:
    ```yaml
    dataset_name: my-username/my-dataset
    image_field: your_image_column
    caption_field: your_text_column
    ```

### 2. Local Folder Method
To use a local directory of images, the system currently expects a format compatible with the Hugging Face `datasets` library. You can use a local path in `dataset_name`, and the library will attempt to load it as an `ImageFolder`.

```yaml
dataset_name: /path/to/your/local/folder
image_field: image
caption_field: text # If using a metadata.jsonl
```