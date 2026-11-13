"""Model profiles and loading with explicit, logged fallback.

A profile = one .pte + one preprocessing contract + one decoder + one EcoGuard
configuration. Loading verifies (a) the file exists and looks like a .pte,
(b) the optional SHA-256 pin, (c) the input AND output shapes reported by the
.pte match the profile. If the active profile fails and fallback is allowed,
the fallback profile is loaded and the failure is logged and reported; nothing
is silently substituted.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Optional

from .classset import load_class_set
from .config import AppConfig, ConfigError, load_yaml
from .decoders import DirectClassDecoder, ImageNetMappingDecoder
from .ecoguard import EcoGuardConfig, EcoGuardConfigError
from .imagenet_labels import load_labels
from .materialwise import load_mapping
from .model_runtime import ExecuTorchClassifier, ModelLoadError, RuntimeUnavailableError
from .paths import CONFIG_DIR, resolve
from .preprocessing import PreprocessSpec

log = logging.getLogger("zyqora.profiles")
DEFAULT_MODELS_CONFIG = CONFIG_DIR / "models.yaml"
DECODERS = ("direct_classes", "imagenet_mapping")


@dataclass(frozen=True)
class ModelProfile:
    name: str
    decoder: str
    pte: Path
    contract: Path
    ecoguard: EcoGuardConfig
    classes: Optional[Path] = None      # direct_classes
    mapping: Optional[Path] = None      # imagenet_mapping
    labels: Optional[Path] = None       # imagenet_mapping (None = bundled)
    sha256: Optional[str] = None
    description: str = ""


@dataclass(frozen=True)
class ModelsConfig:
    active: str
    fallback: Optional[str]
    profiles: dict[str, ModelProfile]
    source: Path

    def get(self, name: str) -> ModelProfile:
        if name not in self.profiles:
            raise ConfigError(f"unknown model profile '{name}' (available: {sorted(self.profiles)})")
        return self.profiles[name]


def load_models_config(app: AppConfig, path: Path = DEFAULT_MODELS_CONFIG) -> ModelsConfig:
    data = load_yaml(path)
    raw_profiles = data.get("profiles")
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise ConfigError(f"{path}: 'profiles' must be a non-empty mapping")
    profiles: dict[str, ModelProfile] = {}
    for name, p in raw_profiles.items():
        if not isinstance(p, dict):
            raise ConfigError(f"{path}: profile '{name}' must be a mapping")
        decoder = p.get("decoder")
        if decoder not in DECODERS:
            raise ConfigError(f"{path}: profile '{name}': decoder must be one of {DECODERS}, got {decoder!r}")
        if p.get("from_app_config"):
            if decoder != "imagenet_mapping":
                raise ConfigError(f"{path}: profile '{name}': from_app_config is only for the imagenet_mapping decoder")
            profiles[name] = ModelProfile(name, decoder, app.pte, app.contract, app.ecoguard, mapping=app.mapping,
                                          labels=app.labels, description=p.get("description", ""))
            continue
        try:
            guard = EcoGuardConfig.from_dict(p.get("ecoguard") or {})
        except (EcoGuardConfigError, TypeError) as exc:
            raise ConfigError(f"{path}: profile '{name}': {exc}") from exc
        for key in ("pte", "contract"):
            if not p.get(key):
                raise ConfigError(f"{path}: profile '{name}' needs '{key}'")
        if decoder == "direct_classes" and not p.get("classes"):
            raise ConfigError(f"{path}: profile '{name}' (direct_classes) needs 'classes'")
        if decoder == "imagenet_mapping" and not p.get("mapping"):
            raise ConfigError(f"{path}: profile '{name}' (imagenet_mapping) needs 'mapping'")
        sha = p.get("sha256")
        if sha is not None and (not isinstance(sha, str) or len(sha) != 64):
            raise ConfigError(f"{path}: profile '{name}': sha256 must be a 64-character hex string or null")
        profiles[name] = ModelProfile(
            name=name, decoder=decoder, pte=resolve(p["pte"]), contract=resolve(p["contract"]), ecoguard=guard,
            classes=resolve(p["classes"]) if p.get("classes") else None,
            mapping=resolve(p["mapping"]) if p.get("mapping") else None,
            labels=resolve(p["labels"]) if p.get("labels") else None,
            sha256=sha.lower() if sha else None, description=p.get("description", ""),
        )
    active, fallback = data.get("active"), data.get("fallback")
    if active not in profiles:
        raise ConfigError(f"{path}: active profile {active!r} is not defined")
    if fallback is not None and fallback not in profiles:
        raise ConfigError(f"{path}: fallback profile {fallback!r} is not defined")
    return ModelsConfig(active, fallback, profiles, Path(path))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class LoadedModel:
    profile: ModelProfile
    classifier: ExecuTorchClassifier
    decoder: object
    spec: PreprocessSpec
    guard: EcoGuardConfig
    fallback_used: bool = False
    failures: list[str] = field(default_factory=list)

    @property
    def category_set(self):
        return self.decoder.category_set

    def describe(self) -> dict:
        return {"profile": self.profile.name, "decoder": self.profile.decoder, "description": self.profile.description,
                "fallback_used": self.fallback_used, "failed_profiles": self.failures,
                "contract": str(self.profile.contract), **self.classifier.describe()}


def build_decoder(profile: ModelProfile, num_classes: int):
    if profile.decoder == "direct_classes":
        return DirectClassDecoder(load_class_set(profile.classes, num_classes))
    labels = load_labels(profile.labels)
    return ImageNetMappingDecoder(labels, load_mapping(profile.mapping, labels))


def load_profile(profile: ModelProfile, *, num_threads: Optional[int] = None,
                 require_backend: Optional[str] = "XnnpackBackend",
                 portable_lib: Optional[ModuleType] = None) -> LoadedModel:
    contract = json.loads(Path(profile.contract).read_text(encoding="utf-8"))
    spec = PreprocessSpec.from_contract(contract)
    out_shape = contract.get("output", {}).get("shape", [1, 1000])
    num_classes = int(out_shape[-1])
    decoder = build_decoder(profile, num_classes)  # validates class/mapping files before touching the model
    if decoder.num_classes != num_classes:
        raise ConfigError(f"profile '{profile.name}': decoder has {decoder.num_classes} classes, contract {num_classes}")
    if profile.sha256 and Path(profile.pte).is_file():
        actual = file_sha256(Path(profile.pte))
        if actual != profile.sha256:
            raise ModelLoadError(f"{profile.pte.name}: SHA-256 {actual[:12]}... does not match the pinned "
                                 f"{profile.sha256[:12]}... in config/models.yaml (corrupt or different model)")
    clf = ExecuTorchClassifier(profile.pte, input_shape=spec.input_shape, num_classes=num_classes,
                               num_threads=num_threads, require_backend=require_backend, portable_lib=portable_lib)
    log.info("loaded profile %s: %s (%s outputs, %s)", profile.name, profile.pte.name, num_classes, clf.thread_control)
    return LoadedModel(profile, clf, decoder, spec, profile.ecoguard)


def load_with_fallback(models: ModelsConfig, name: Optional[str] = None, *, allow_fallback: bool,
                       num_threads: Optional[int] = None, require_backend: Optional[str] = "XnnpackBackend",
                       portable_lib: Optional[ModuleType] = None) -> LoadedModel:
    """Load `name` (default: active). On failure, load the fallback only if allowed, and say so."""
    primary = models.get(name or models.active)
    try:
        return load_profile(primary, num_threads=num_threads, require_backend=require_backend, portable_lib=portable_lib)
    except (ModelLoadError, RuntimeUnavailableError, ConfigError, OSError, ValueError) as exc:
        if isinstance(exc, RuntimeUnavailableError) or not allow_fallback or not models.fallback \
                or models.fallback == primary.name:
            raise
        failure = f"{primary.name}: {exc}"
        log.error("model profile %s failed to load; falling back to %s. Reason: %s", primary.name, models.fallback, exc)
    loaded = load_profile(models.get(models.fallback), num_threads=num_threads, require_backend=require_backend,
                          portable_lib=portable_lib)
    loaded.fallback_used = True
    loaded.failures.append(failure)
    return loaded
