"""Organiser label profiles, detector material hints and disposal recommendations.

All three are configuration-driven and keep the *model label* separate from the
displayed organiser label, the material category and the recommended stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import ConfigError, load_yaml
from .paths import CONFIG_DIR

MATERIALS = ("Plastic", "Paper", "Cardboard", "Metal", "Glass", "Organic", "E-Waste", "Other", "Unknown")
CERTAINTIES = ("likely", "unknown")


# --------------------------------------------------------------------------- label profiles
@dataclass(frozen=True)
class LabelProfile:
    name: str
    title: str
    description: str
    labels: dict[str, str]          # model class -> organiser label
    unknown_label: str = "Unknown"

    def label(self, model_class: Optional[str], reliable: bool) -> str:
        if not reliable or model_class is None:
            return self.unknown_label
        return self.labels.get(model_class, self.unknown_label)

    def categories(self) -> list[str]:
        out = list(dict.fromkeys(self.labels.values()))
        return out + [self.unknown_label]

    def as_dict(self) -> dict:
        return {"name": self.name, "title": self.title, "description": self.description, "labels": self.labels,
                "unknown_label": self.unknown_label, "categories": self.categories()}


@dataclass(frozen=True)
class LabelProfiles:
    active: str
    profiles: dict[str, LabelProfile]

    def get(self, name: Optional[str] = None) -> LabelProfile:
        name = name or self.active
        if name not in self.profiles:
            raise ConfigError(f"unknown label profile '{name}' (available: {sorted(self.profiles)})")
        return self.profiles[name]


def load_label_profiles(path: Path | str = CONFIG_DIR / "label_profiles.yaml",
                        model_classes: Optional[list[str]] = None) -> LabelProfiles:
    data = load_yaml(path)
    raw = data.get("profiles")
    if not isinstance(raw, dict) or not raw:
        raise ConfigError(f"{path}: 'profiles' must be a non-empty mapping")
    out = {}
    for name, p in raw.items():
        labels = p.get("labels") if isinstance(p, dict) else None
        if not isinstance(labels, dict) or not labels:
            raise ConfigError(f"{path}: label profile '{name}' needs a 'labels' mapping")
        if model_classes is not None and set(labels) != set(model_classes):
            raise ConfigError(f"{path}: label profile '{name}' must map exactly the model classes {model_classes}, "
                              f"got {sorted(labels)}")
        if len(set(labels.values())) != len(labels):
            raise ConfigError(f"{path}: label profile '{name}' maps two model classes to the same label")
        out[name] = LabelProfile(name, str(p.get("title", name)), " ".join(str(p.get("description", "")).split()),
                                 {str(k): str(v) for k, v in labels.items()}, str(p.get("unknown_label", "Unknown")))
    active = data.get("active")
    if active not in out:
        raise ConfigError(f"{path}: active label profile {active!r} is not defined")
    return LabelProfiles(active, out)


# --------------------------------------------------------------------------- materials
@dataclass(frozen=True)
class MaterialHint:
    detector_label: Optional[str]
    waste_item: bool
    material: str = "Unknown"
    certainty: str = "unknown"
    food: Optional[bool] = None
    expected_model_class: Optional[str] = None
    special_stream: Optional[str] = None
    note: str = ""

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass(frozen=True)
class MaterialTable:
    version: str
    classes: dict[str, MaterialHint]

    def hint(self, detector_label: Optional[str]) -> MaterialHint:
        if detector_label is None:          # whole-image fallback: no detector information
            return MaterialHint(None, True, note="no detector label (whole image classified)")
        h = self.classes.get(detector_label)
        return h if h is not None else MaterialHint(detector_label, False, note="not a waste item (not in the material table)")


def load_material_table(path: Path | str, model_classes: Optional[list[str]] = None,
                        detector_classes: Optional[tuple[str, ...]] = None) -> MaterialTable:
    data = load_yaml(path)
    raw = data.get("classes")
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: 'classes' must be a mapping")
    out = {}
    for label, d in raw.items():
        if detector_classes is not None and label not in detector_classes:
            raise ConfigError(f"{path}: '{label}' is not a detector class")
        if not isinstance(d, dict) or d.get("material") not in MATERIALS:
            raise ConfigError(f"{path}: '{label}': material must be one of {MATERIALS}")
        if d.get("certainty", "unknown") not in CERTAINTIES:
            raise ConfigError(f"{path}: '{label}': certainty must be one of {CERTAINTIES}")
        emc = d.get("expected_model_class")
        if emc is not None and model_classes is not None and emc not in model_classes:
            raise ConfigError(f"{path}: '{label}': expected_model_class {emc!r} is not a model class {model_classes}")
        out[label] = MaterialHint(label, True, d["material"], d.get("certainty", "unknown"), bool(d.get("food", False)),
                                  emc, d.get("special_stream"), str(d.get("note", "")))
    return MaterialTable(str(data.get("version", "unversioned")), out)


# --------------------------------------------------------------------------- recommendations
@dataclass
class Recommendation:
    stream: str
    stream_title: dict[str, str]
    guidance: dict[str, str]
    warnings: list[str] = field(default_factory=list)
    basis: str = ""

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class RecommendationEngine:
    def __init__(self, rules: dict, source: str = "<dict>"):
        self.version = str(rules.get("version", "unversioned"))
        self.streams = rules.get("streams") or {}
        self.materials = rules.get("materials") or {}
        self.model_classes = rules.get("model_classes") or {}
        self.review = rules.get("review")
        if not self.review:
            raise ConfigError(f"{source}: a 'review' rule is required")
        for group in (self.materials, self.model_classes, {"review": self.review}):
            for key, r in group.items():
                if not isinstance(r, dict) or r.get("stream") not in self.streams:
                    raise ConfigError(f"{source}: rule '{key}' needs a stream defined under 'streams'")
                if not r.get("en"):
                    raise ConfigError(f"{source}: rule '{key}' needs English guidance 'en'")

    @classmethod
    def load(cls, path: Path | str = CONFIG_DIR / "disposal_rules.yaml") -> "RecommendationEngine":
        return cls(load_yaml(path), str(path))

    def _make(self, r: dict, basis: str, extra: Optional[list[str]] = None) -> Recommendation:
        s = r["stream"]
        return Recommendation(s, dict(self.streams[s]), {"en": r["en"], "ta": r.get("ta", "")},
                              list(r.get("warnings") or []) + list(extra or []), basis)

    def recommend(self, hint: MaterialHint, model_class: Optional[str], reliable: bool,
                  conflict: bool = False) -> Recommendation:
        if hint.special_stream == "E_WASTE" and "E-Waste" in self.materials:
            # Electronics are routed to e-waste even when the classifier is unsure: never into a household bin.
            extra = [] if reliable else ["The waste model was not confident; the detector identified an electronic device."]
            if model_class == "Recyclable":
                extra.append("The waste model says 'Recyclable', but electronics do not belong in the dry-recycling bin.")
            return self._make(self.materials["E-Waste"], "special_stream:E_WASTE", extra)
        if not reliable or conflict:
            extra = ["Detector and classifier disagree about this item."] if conflict else []
            return self._make(self.review, "review", extra)
        if hint.certainty == "likely" and hint.material in self.materials:
            return self._make(self.materials[hint.material], f"material:{hint.material}")
        if model_class in self.model_classes:
            extra = [f"Detected as '{hint.detector_label}': {hint.note}."] if hint.detector_label and hint.note else []
            return self._make(self.model_classes[model_class], f"model_class:{model_class}", extra)
        return self._make(self.review, "review")
