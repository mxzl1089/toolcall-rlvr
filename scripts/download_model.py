"""Explicit network-only model acquisition, separate from offline training."""
from huggingface_hub import snapshot_download
if __name__ == '__main__':
    snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir='models/Qwen2.5-0.5B-Instruct')
