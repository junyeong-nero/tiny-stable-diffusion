# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# setup env
uv sync
uv pip install git+https://github.com/openai/CLIP.git
uv run wandb login

# tmux setup
apt-get update && apt-get install -y tmux