"""Project-relative paths. Everything is resolved locally; nothing is fetched."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
AUDIO_DIR = PROJECT_ROOT / "audio"
BUNDLED_LABELS = PACKAGE_DIR / "data" / "imagenet_labels.json"

DEFAULT_APP_CONFIG = CONFIG_DIR / "app.yaml"
DEFAULT_MAPPING = CONFIG_DIR / "mapping.json"
DEFAULT_MODEL_CONTRACT = CONFIG_DIR / "model.json"
DEFAULT_VOICE_CONFIG = CONFIG_DIR / "voice.yaml"


def resolve(path: str | Path, base: Path = PROJECT_ROOT) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (base / p)
