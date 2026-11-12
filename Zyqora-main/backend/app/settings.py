from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
RESULTS_DIR = ROOT / "results"
DB_PATH = RESULTS_DIR / "zyqora.sqlite3"

def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_thresholds():
    return load_yaml(CONFIG_DIR / "thresholds.yaml")

def load_label_map():
    return load_yaml(CONFIG_DIR / "label_map.yaml")
