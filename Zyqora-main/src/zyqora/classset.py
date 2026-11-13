"""Class set for a model that predicts organiser-style categories directly (no ImageNet mapping).

``config/classes_waste.json`` lists the fine-tuned model's classes in output-index
order (0 = Organic, 1 = Recyclable) with their disposal actions. Categories the
model was never trained on (e.g. Residual) can be declared under
``unsupported_ground_truth``: they are accepted as ground-truth labels during
evaluation, are counted in the denominator, and can never be predicted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .materialwise import ACTION_PATTERN, RESERVED_ACTIONS, MappingError

CLASS_SET_STATUSES = ("MODEL_CLASSES",)


@dataclass(frozen=True)
class ClassSet:
    status: str
    class_names: tuple[str, ...]                 # index order == model output order
    categories: dict[str, str]                   # class name -> disposal action
    aliases: dict[str, str]                      # lower-case alias -> canonical name (incl. unsupported)
    unsupported_ground_truth: frozenset[str] = field(default_factory=frozenset)
    source: str = ""
    note: str = ""
    aggregation: str = "direct_softmax"
    rules: dict = field(default_factory=dict)    # no ImageNet rules for a direct model

    @property
    def is_organiser(self) -> bool:
        return False

    @property
    def category_names(self) -> list[str]:
        return list(self.class_names)

    @property
    def actions(self) -> list[str]:
        return sorted(set(self.categories.values()))

    def canonical_category(self, name: str) -> Optional[str]:
        """A category the model can predict (exact name, then case-insensitive alias)."""
        if name in self.categories:
            return name
        canon = self.aliases.get(name.strip().lower())
        return canon if canon in self.categories else None

    def ground_truth_category(self, name: str) -> Optional[str]:
        """Like canonical_category, but also accepts declared unsupported ground-truth categories."""
        if name in self.categories or name in self.unsupported_ground_truth:
            return name
        return self.aliases.get(name.strip().lower())


def validate_class_set_dict(data: object, num_classes: Optional[int] = None) -> list[str]:
    problems: list[str] = []
    if not isinstance(data, dict):
        return ["top level must be a JSON object"]
    unknown = set(data) - {"schema_version", "status", "note", "classes", "unsupported_ground_truth"}
    if unknown:
        problems.append(f"unknown top-level keys {sorted(unknown)}")
    if data.get("status") not in CLASS_SET_STATUSES:
        problems.append(f"status must be one of {CLASS_SET_STATUSES}, got {data.get('status')!r}")
    classes = data.get("classes")
    if not isinstance(classes, list) or len(classes) < 2:
        return problems + ["classes must be a list with at least two entries"]
    seen_alias: dict[str, str] = {}
    names: list[str] = []
    for n, entry in enumerate(classes):
        where = f"classes[{n}]"
        if not isinstance(entry, dict):
            problems.append(f"{where} must be an object")
            continue
        extra = set(entry) - {"index", "name", "action", "aliases"}
        if extra:
            problems.append(f"{where} has unknown keys {sorted(extra)}")
        if entry.get("index") != n:
            problems.append(f"{where}: index must be {n} (classes must be listed in model output order)")
        name, action = entry.get("name"), entry.get("action")
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            problems.append(f"{where}: invalid name {name!r}")
            continue
        if name in names:
            problems.append(f"{where}: duplicate class name '{name}'")
        names.append(name)
        if not isinstance(action, str) or not ACTION_PATTERN.match(action):
            problems.append(f"{where}: action {action!r} must match {ACTION_PATTERN.pattern}")
        elif action in RESERVED_ACTIONS:
            problems.append(f"{where}: action '{action}' is reserved for EcoGuard REVIEW")
        for alias in [name, *entry.get("aliases", [])]:
            key = str(alias).strip().lower()
            if key in seen_alias and seen_alias[key] != name:
                problems.append(f"alias '{alias}' used by both '{seen_alias[key]}' and '{name}'")
            seen_alias[key] = name
    unsupported = data.get("unsupported_ground_truth", {}) or {}
    if not isinstance(unsupported, dict):
        problems.append("unsupported_ground_truth must be an object")
        unsupported = {}
    for name, spec in unsupported.items():
        if name in names:
            problems.append(f"'{name}' cannot be both a model class and unsupported")
        for alias in [name, *((spec or {}).get("aliases", []))]:
            key = str(alias).strip().lower()
            if key in seen_alias and seen_alias[key] != name:
                problems.append(f"alias '{alias}' used by both '{seen_alias[key]}' and '{name}'")
            seen_alias[key] = name
    if num_classes is not None and len(classes) != num_classes:
        problems.append(f"{len(classes)} classes listed but the model contract has {num_classes} outputs")
    return problems


def load_class_set(path: str | Path, num_classes: Optional[int] = None) -> ClassSet:
    p = Path(path)
    if not p.is_file():
        raise MappingError([f"file not found: {p}"], "class set")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MappingError([f"not valid JSON: {exc}"], str(p)) from exc
    problems = validate_class_set_dict(data, num_classes)
    if problems:
        raise MappingError(problems, str(p))
    return class_set_from_dict(data, source=str(p))


def class_set_from_dict(data: dict, source: str = "") -> ClassSet:
    classes = sorted(data["classes"], key=lambda c: c["index"])
    aliases: dict[str, str] = {}
    for c in classes:
        for a in [c["name"], *c.get("aliases", [])]:
            aliases[a.strip().lower()] = c["name"]
    unsupported = data.get("unsupported_ground_truth", {}) or {}
    for name, spec in unsupported.items():
        for a in [name, *((spec or {}).get("aliases", []))]:
            aliases[a.strip().lower()] = name
    return ClassSet(
        status=data["status"],
        class_names=tuple(c["name"] for c in classes),
        categories={c["name"]: c["action"] for c in classes},
        aliases=aliases,
        unsupported_ground_truth=frozenset(unsupported),
        source=source,
        note=data.get("note", ""),
    )
