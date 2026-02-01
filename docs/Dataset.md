# Dataset Guide

> Details on the datasets used for training `tiny-stable-diffusion` and how to add your own.

---

## Table of Contents

1. [Supported Datasets](#supported-datasets)
2. [Configuration](#configuration)
3. [Streaming vs. Local](#streaming-vs-local)
4. [Custom Datasets](#custom-datasets)

---

## Supported Datasets

We primarily support datasets available on the **Hugging Face Hub**.

### 1. LAION-300k (Recommended for VAE)
*   **ID**: `hmu013/LAION-300k`
*   **Type**: General Domain
*   **Size**: 300,000 images
*   **Format**: Parquet/Arrow (Streaming)
*   **Usage**: VAE Training (Stage 1). Since VAE needs to learn general image statistics (edges, textures, colors), a diverse dataset like LAION is ideal.

### 2. Oxford Pets Enriched (Recommended for Diffusion)
*   **ID**: `visual-layer/oxford-iiit-pet-vl-enriched`
*   **Type**: Specific Domain (Cats & Dogs)
*   **Size**: ~7,400 images
*   **Format**: Image Folder
*   **Usage**: Diffusion Training (Stage 2).
*   **Why?**: For a "tiny" model, training on a specific domain (like pets) yields much better visual results than trying to learn "everything" with limited capacity.

### 3. Pokemon BLIP Captions
*   **ID**: `reach-vb/pokemon-blip-captions`
*   **Type**: Specific Domain (Cartoons)
*   **Size**: ~800 images
*   **Usage**: Fast debugging / Hello World.

---

## Configuration

Datasets are configured in `config.yaml` under `vae_train` or `diffusion_train`.

### Example: Streaming VAE Dataset
```yaml
vae_train:
  dataset_name: hmu013/LAION-300k
  image_field: png       # Column name containing the image
  stream: true           # Enable streaming mode
  batch_size: 256
```

### Example: Diffusion Dataset (Oxford Pets)
```yaml
diffusion_train:
  dataset_name: visual-layer/oxford-iiit-pet-vl-enriched
  image_field: image
  caption_field: caption_enriched
  stream: false          # Small enough to download fully
```

---

## Streaming vs. Local

### Streaming (`stream: true`)
*   **Pros**: Starts training immediately, no disk space required for full dataset.
*   **Cons**: Requires constant internet, potential network bottlenecks.
*   **Best for**: Large datasets (LAION, COCO, CC3M).

### Local (`stream: false`)
*   **Pros**: Faster per-epoch training after initial download, offline capable.
*   **Cons**: Uses disk space, initial download wait time.
*   **Best for**: Small datasets (<10GB) like Oxford Pets, Pokemon, Flowers.

---

## Custom Datasets

To use your own dataset, upload it to Hugging Face or use a local folder structure.

### 1. Hugging Face Method (Easiest)
1.  Upload your data to HF (e.g., `my-user/my-dataset`).
2.  Ensure it has an image column and a text column.
3.  Update `config.yaml`:
    ```yaml
    dataset_name: my-user/my-dataset
    image_field: image
    caption_field: text
    ```

### 2. Local Folder Method (Future Support)
*Currently, the code is optimized for HF Hub datasets. To use local files, you would need to modify `src/data/loader.py` to use `ImageFolder` or `WebDataset` locally.*
