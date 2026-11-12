"""Load and validate config/app.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from .ecoguard import EcoGuardConfig, EcoGuardConfigError
from .paths import DEFAULT_APP_CONFIG, resolve


class ConfigError(ValueError):
    pass


def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Config file not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{p} must contain a mapping at the top level")
    return data


@dataclass(frozen=True)
class AppConfig:
    pte: Path
    contract: Path
    labels: Optional[Path]
    num_threads: Optional[int]
    require_backend: Optional[str]
    mapping: Path
    voice: Path
    ecoguard: EcoGuardConfig
    greenscore_db: Path
    points_correct: int
    points_incorrect: int
    annotate_count: int
    source: Path


def _int(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def load_app_config(path: str | Path = DEFAULT_APP_CONFIG) -> AppConfig:
    data = load_yaml(path)
    try:
        model = data["model"]
        gs = data.get("greenscore", {})
        outputs = data.get("outputs", {})
        threads = model.get("num_threads")
        if threads is not None:
            threads = _int(threads, "model.num_threads", 1)
        try:
            guard = EcoGuardConfig.from_dict(data["ecoguard"])
        except (EcoGuardConfigError, TypeError) as exc:
            raise ConfigError(str(exc)) from exc
        return AppConfig(
            pte=resolve(model["pte"]),
            contract=resolve(model["contract"]),
            labels=resolve(model["labels"]) if model.get("labels") else None,
            num_threads=threads,
            require_backend=model.get("require_backend"),
            mapping=resolve(data["mapping"]),
            voice=resolve(data["voice"]),
            ecoguard=guard,
            greenscore_db=resolve(gs.get("database", "outputs/greenscore.sqlite3")),
            points_correct=_int(gs.get("points_verified_correct", 10), "greenscore.points_verified_correct"),
            points_incorrect=_int(gs.get("points_verified_incorrect", 0), "greenscore.points_verified_incorrect"),
            annotate_count=_int(outputs.get("annotate_count", 2), "outputs.annotate_count"),
            source=Path(path),
        )
    except KeyError as exc:
        raise ConfigError(f"Missing required key {exc} in {path}") from exc
