# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# setup env
uv sync
uv pip install git+https://github.com/openai/CLIP.git
uv add wandb
uv run wandb login
