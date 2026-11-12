"""MaterialWise: configurable ImageNet prediction -> organiser category -> disposal action.

No extra model is trained. ``config/mapping.json`` lists which ImageNet class
indices belong to which organiser category. Category scores are aggregated
from the softmax probabilities of every mapped class (method ``sum_softmax``),
so the best waste category is not necessarily the category of the single
highest-scoring ImageNet class.

Three separate outputs are always kept apart:
    raw ImageNet prediction  |  predicted organiser category  |  disposal action
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .imagenet_labels import NUM_IMAGENET_CLASSES
from .schemas import MANUAL_REVIEW, TOP1_MAPPED, TOP1_UNMAPPED, UNSUPPORTED

MAPPING_STATUSES = ("EXAMPLE", "ORGANISER")
SUPPORTED_AGGREGATIONS = ("sum_softmax",)
ACTION_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{1,31}$")
RESERVED_ACTIONS = frozenset({MANUAL_REVIEW})


class MappingError(ValueError):
    def __init__(self, problems: list[str], source: str = "mapping"):
        self.problems = problems
        super().__init__(f"Invalid {source}:\n  - " + "\n  - ".join(problems))


@dataclass(frozen=True)
class ClassRule:
    index: int
    label: str
    category: str
    material: Optional[str] = None
    rule: Optional[str] = None


@dataclass(frozen=True)
class WasteMapping:
    status: str
    aggregation: str
    categories: dict[str, str]            # organiser category -> disposal action
    aliases: dict[str, str]               # lower-cased alias -> canonical category
    rules: dict[int, ClassRule]           # ImageNet index -> rule
    source: str = ""
    note: str = ""
    _index_array: np.ndarray = field(default=None, repr=False, compare=False)  # type: ignore[assignment]
    _category_ids: np.ndarray = field(default=None, repr=False, compare=False)  # type: ignore[assignment]

    @property
    def is_organiser(self) -> bool:
        return self.status == "ORGANISER"

    @property
    def category_names(self) -> list[str]:
        return list(self.categories)

    @property
    def actions(self) -> list[str]:
        return sorted(set(self.categories.values()))

    def canonical_category(self, name: str) -> Optional[str]:
        """Exact name first, then case-insensitive name/alias match. None if unknown."""
        if name in self.categories:
            return name
        return self.aliases.get(name.strip().lower())


def validate_mapping_dict(data: object, labels: list[str]) -> list[str]:
    """Return a list of human-readable problems (empty list == valid)."""
    problems: list[str] = []
    if not isinstance(data, dict):
        return ["top level must be a JSON object"]
    allowed_top = {"schema_version", "status", "note", "aggregation", "aggregation_note", "categories", "classes"}
    for key in data:
        if key not in allowed_top:
            problems.append(f"unknown top-level key '{key}' (no implicit defaults are supported)")
    missing = [k for k in ("status", "aggregation", "categories", "classes") if k not in data]
    if missing:
        return problems + [f"missing required key '{k}'" for k in missing]

    if data["status"] not in MAPPING_STATUSES:
        problems.append(f"status must be one of {MAPPING_STATUSES}, got {data['status']!r}")
    if data["aggregation"] not in SUPPORTED_AGGREGATIONS:
        problems.append(f"aggregation must be one of {SUPPORTED_AGGREGATIONS}, got {data['aggregation']!r}")

    cats = data["categories"]
    seen_alias: dict[str, str] = {}
    if not isinstance(cats, dict) or len(cats) < 2:
        problems.append("categories must be an object with at least two organiser categories")
        cats = cats if isinstance(cats, dict) else {}
    for name, spec in cats.items():
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            problems.append(f"category name {name!r} must be a non-empty string without surrounding spaces")
            continue
        if not isinstance(spec, dict) or "action" not in spec:
            problems.append(f"category '{name}' must be an object with an 'action'")
            continue
        unknown = set(spec) - {"action", "aliases", "description"}
        if unknown:
            problems.append(f"category '{name}' has unknown keys {sorted(unknown)}")
        action = spec["action"]
        if not isinstance(action, str) or not ACTION_PATTERN.match(action):
            problems.append(f"category '{name}': action {action!r} must match {ACTION_PATTERN.pattern}")
        elif action in RESERVED_ACTIONS:
            problems.append(f"category '{name}': action '{action}' is reserved for EcoGuard REVIEW")
        for alias in [name, *spec.get("aliases", [])]:
            if not isinstance(alias, str) or not alias.strip():
                problems.append(f"category '{name}': invalid alias {alias!r}")
                continue
            key = alias.strip().lower()
            if key in seen_alias and seen_alias[key] != name:
                problems.append(f"alias '{alias}' is used by both '{seen_alias[key]}' and '{name}'")
            seen_alias[key] = name

    classes = data["classes"]
    if not isinstance(classes, list) or not classes:
        problems.append("classes must be a non-empty list")
        return problems
    seen_idx: dict[int, str] = {}
    used_categories: set[str] = set()
    for n, entry in enumerate(classes):
        where = f"classes[{n}]"
        if not isinstance(entry, dict):
            problems.append(f"{where} must be an object")
            continue
        unknown = set(entry) - {"index", "label", "category", "material", "rule"}
        if unknown:
            problems.append(f"{where} has unknown keys {sorted(unknown)}")
        idx, label, cat = entry.get("index"), entry.get("label"), entry.get("category")
        if not isinstance(idx, int) or isinstance(idx, bool) or not 0 <= idx < NUM_IMAGENET_CLASSES:
            problems.append(f"{where}: index {idx!r} must be an integer in [0, {NUM_IMAGENET_CLASSES - 1}]")
            continue
        if label != labels[idx]:
            problems.append(f"{where}: index {idx} is '{labels[idx]}' in ImageNet order, not {label!r}")
        if cat not in cats:
            problems.append(f"{where}: category {cat!r} is not a declared organiser category")
        else:
            used_categories.add(cat)
        if idx in seen_idx:
            kind = "conflicting" if seen_idx[idx] != cat else "duplicate"
            problems.append(f"{where}: {kind} mapping for index {idx} ('{labels[idx]}'): '{seen_idx[idx]}' vs '{cat}'")
        else:
            seen_idx[idx] = cat
    for name in cats:
        if name not in used_categories:
            problems.append(f"category '{name}' has no ImageNet classes mapped to it; it could never be predicted")
    return problems


def load_mapping(path: str | Path, labels: list[str]) -> WasteMapping:
    p = Path(path)
    if not p.is_file():
        raise MappingError([f"file not found: {p}"], "mapping")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MappingError([f"not valid JSON: {exc}"], str(p)) from exc
    problems = validate_mapping_dict(data, labels)
    if problems:
        raise MappingError(problems, str(p))
    return mapping_from_dict(data, source=str(p))


def mapping_from_dict(data: dict, source: str = "") -> WasteMapping:
    categories = {name: spec["action"] for name, spec in data["categories"].items()}
    aliases: dict[str, str] = {}
    for name, spec in data["categories"].items():
        for alias in [name, *spec.get("aliases", [])]:
            aliases[alias.strip().lower()] = name
    rules = {
        e["index"]: ClassRule(e["index"], e["label"], e["category"], e.get("material"), e.get("rule"))
        for e in data["classes"]
    }
    names = list(categories)
    idx = np.array(sorted(rules), dtype=np.int64)
    cat_ids = np.array([names.index(rules[i].category) for i in idx], dtype=np.int64)
    return WasteMapping(
        status=data["status"],
        aggregation=data["aggregation"],
        categories=categories,
        aliases=aliases,
        rules=rules,
        source=source,
        note=data.get("note", ""),
        _index_array=idx,
        _category_ids=cat_ids,
    )


@dataclass
class CategoryResult:
    predicted_category: Optional[str]
    predicted_category_score: Optional[float]
    category_scores: dict[str, float]
    category_margin: Optional[float]
    mapped_mass: float
    mapping_status: str
    top1_rule: Optional[ClassRule]
    category_action: Optional[str]


def map_prediction(probs: np.ndarray, mapping: WasteMapping) -> CategoryResult:
    """Aggregate softmax probabilities into organiser-category scores."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.shape != (NUM_IMAGENET_CLASSES,):
        raise ValueError(f"expected {NUM_IMAGENET_CLASSES} probabilities, got shape {probs.shape}")
    names = mapping.category_names
    scores = np.zeros(len(names), dtype=np.float64)
    np.add.at(scores, mapping._category_ids, probs[mapping._index_array])
    mapped_mass = float(scores.sum())
    category_scores = {n: float(s) for n, s in zip(names, scores)}

    top1 = int(np.argmax(probs))
    rule = mapping.rules.get(top1)

    if not np.isfinite(mapped_mass) or mapped_mass <= 0.0:
        return CategoryResult(None, None, category_scores, None, mapped_mass, UNSUPPORTED, rule, None)

    order = np.argsort(-scores, kind="stable")  # ties -> config order (deterministic)
    best = names[int(order[0])]
    margin = float(scores[order[0]] - scores[order[1]]) if len(names) > 1 else float(scores[order[0]])
    return CategoryResult(
        predicted_category=best,
        predicted_category_score=float(scores[order[0]]),
        category_scores=category_scores,
        category_margin=margin,
        mapped_mass=mapped_mass,
        mapping_status=TOP1_MAPPED if rule is not None else TOP1_UNMAPPED,
        top1_rule=rule,
        category_action=mapping.categories[best],
    )
