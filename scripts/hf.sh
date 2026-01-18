python scripts/upload_to_hub.py --model-type vae --repo-id username/tiny-sd-vae

# Diffusion만 업로드
python scripts/upload_to_hub.py --model-type diffusion --repo-id username/tiny-sd-diffusion

# 둘 다 하나의 repo에 업로드
python scripts/upload_to_hub.py --model-type all --repo-id username/tiny-sd-models

# VAE 다운로드
python scripts/download_from_hub.py --repo-id username/tiny-sd-vae --model-type vae

# 모든 모델 다운로드
python scripts/download_from_hub.py --repo-id username/tiny-sd-models --model-type all