
import logging
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

logger = logging.getLogger(__name__)

DRAMABOX_REPO = "ResembleAI/Dramabox"
GEMMA_REPO = "unsloth/gemma-3-12b-it-bnb-4bit"
REUSE_REPO = "nvidia/RE-USE"


DEFAULT_CACHE = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~")), ".cache", "dramabox")


MODEL_FILES = {
    "transformer": "dramabox-dit-v1.safetensors",
    "audio_components": "dramabox-audio-components.safetensors",
    "silence_latent": "assets/silence_latent_frame.pt",
}


def get_model_path(name: str, cache_dir: str = None) -> str:
    cache_dir = cache_dir or DEFAULT_CACHE

    if name not in MODEL_FILES:
        raise ValueError(f"Unknown model: {name}. Choose from: {list(MODEL_FILES.keys())}")

    repo_path = MODEL_FILES[name]
    logger.info(f"Fetching {name} from {DRAMABOX_REPO}/{repo_path}...")

    local_path = hf_hub_download(
        repo_id=DRAMABOX_REPO,
        filename=repo_path,
        cache_dir=cache_dir,
        token=os.environ.get("HF_TOKEN"),
    )
    logger.info(f"  -> {local_path}")
    return local_path


def get_gemma_path(cache_dir: str = None) -> str:
    cache_dir = cache_dir or DEFAULT_CACHE
    logger.info(f"Fetching Gemma from {GEMMA_REPO}...")

    local_dir = snapshot_download(
        repo_id=GEMMA_REPO,
        cache_dir=cache_dir,
        token=os.environ.get("HF_TOKEN"),
    )
    logger.info(f"  -> {local_dir}")
    return local_dir


def get_reuse_code_path(cache_dir: str = None) -> str:
    env_dir = os.environ.get("REUSE_DIR")
    if env_dir and Path(env_dir).is_dir():
        return env_dir

    repo_root = Path(__file__).resolve().parent.parent
    local_vendor = repo_root / "third_party" / "RE-USE"
    if (local_vendor / "models" / "generator_SEMamba_time_d4.py").is_file():
        return str(local_vendor)

    cache_dir = cache_dir or DEFAULT_CACHE
    logger.info(f"Fetching RE-USE code/configs from {REUSE_REPO}...")
    local_dir = snapshot_download(
        repo_id=REUSE_REPO,
        cache_dir=cache_dir,
        token=os.environ.get("HF_TOKEN"),
        allow_patterns=["*.py", "*.yaml", "*.json",
                        "recipes/*", "models/*.py", "utils/*.py"],
    )
    logger.info(f"  -> {local_dir}")
    return local_dir


def get_all_paths(cache_dir: str = None) -> dict:
    cache_dir = cache_dir or DEFAULT_CACHE
    paths = {}

    for name in MODEL_FILES:
        paths[name] = get_model_path(name, cache_dir)

    paths["gemma_root"] = get_gemma_path(cache_dir)
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    paths = get_all_paths()
    print("\nAll models downloaded:")
    for k, v in paths.items():
        size = os.path.getsize(v) / 1e9 if os.path.isfile(v) else "dir"
        print(f"  {k}: {v} ({size:.2f}GB)" if isinstance(size, float) else f"  {k}: {v} (directory)")
