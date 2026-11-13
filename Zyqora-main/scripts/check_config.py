#!/usr/bin/env python3
"""Validate Zyqora configuration offline (no model or torch needed).

Checks config/app.yaml, config/mapping.json (structure, class indices and
labels against the bundled ImageNet order, duplicate/conflicting mappings,
categories, actions), config/model.json and config/voice.yaml (messages and
clips for every action).

    python scripts/check_config.py
    python scripts/check_config.py --mapping config/mapping.json
Exit code 0 = valid (warnings allowed), 1 = errors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from zyqora.config import ConfigError, load_app_config, load_yaml
from zyqora.imagenet_labels import LabelsError, load_labels
from zyqora.materialwise import MappingError, load_mapping
from zyqora.paths import AUDIO_DIR, DEFAULT_APP_CONFIG
from zyqora.preprocessing import PreprocessSpec
from zyqora.voicesort import REQUIRED_SYSTEM_KEYS, clip_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_APP_CONFIG, help="config/app.yaml")
    ap.add_argument("--mapping", type=Path, default=None, help="override mapping file")
    args = ap.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    try:
        app = load_app_config(args.config)
    except ConfigError as exc:
        print(f"ERROR app config: {exc}")
        return 1
    print(f"OK   app config {args.config}")

    try:
        labels = load_labels(app.labels)
        print(f"OK   ImageNet labels: {len(labels)} classes")
    except LabelsError as exc:
        print(f"ERROR {exc}")
        return 1

    try:
        spec = PreprocessSpec.from_file(app.contract)
        print(f"OK   model contract {app.contract.name}: input {spec.input_shape}, resize {spec.resize}, crop {spec.center_crop}")
    except (OSError, KeyError, ValueError) as exc:
        errors.append(f"model contract {app.contract}: {exc}")

    mapping_path = args.mapping or app.mapping
    mapping = None
    try:
        mapping = load_mapping(mapping_path, labels)
        print(f"OK   mapping {mapping_path}: status={mapping.status}, {len(mapping.rules)} class rules, "
              f"categories={mapping.categories}, aggregation={mapping.aggregation}")
        if not mapping.is_organiser:
            warnings.append("mapping status is EXAMPLE: categories are NOT the organiser's labels; "
                            "final evaluation is blocked until real categories are configured")
    except MappingError as exc:
        errors.append(str(exc))

    try:
        from zyqora.classset import load_class_set
        from zyqora.profiles import load_models_config

        models = load_models_config(app)
        print(f"OK   models.yaml: active={models.active} fallback={models.fallback} profiles={sorted(models.profiles)}")
        for prof in models.profiles.values():
            contract = json.loads(prof.contract.read_text(encoding="utf-8"))
            pspec = PreprocessSpec.from_contract(contract)
            n_out = int(contract.get("output", {}).get("shape", [1, 1000])[-1])
            if prof.decoder == "direct_classes":
                cs = load_class_set(prof.classes, n_out)
                extra = f"classes={list(cs.class_names)} actions={cs.categories} unsupported={sorted(cs.unsupported_ground_truth)}"
                if mapping is not None:
                    for a in cs.actions:
                        if a not in mapping.actions:
                            warnings.append(f"profile {prof.name}: action {a} not used by the ImageNet mapping")
            else:
                extra = f"mapping={prof.mapping.name}"
            print(f"OK   profile {prof.name}: {prof.decoder}, output {n_out}, resize {pspec.resize}, "
                  f"crop {pspec.center_crop}, EcoGuard {prof.ecoguard.threshold} (calibrated={prof.ecoguard.calibrated}), {extra}")
            if not prof.pte.is_file():
                warnings.append(f"profile {prof.name}: model artifact not present: {prof.pte}")
    except (ConfigError, MappingError, OSError, KeyError, ValueError) as exc:
        errors.append(f"models.yaml / profiles: {exc}")

    g = app.ecoguard
    print(f"OK   ecoguard threshold={g.threshold} min_margin={g.min_margin} "
          f"require_top1_mapped={g.require_top1_mapped} calibrated={g.calibrated} frozen={g.frozen}")
    if not g.calibrated:
        warnings.append("ecoguard threshold is provisional (calibrated: false)")
    if not g.frozen:
        warnings.append("ecoguard threshold not frozen; set frozen: true before the final organiser run")

    if not app.pte.is_file():
        warnings.append(f"model artifact not present yet: {app.pte}")

    try:
        voice = load_yaml(app.voice)
        messages = voice.get("messages", {})
        langs = list(voice.get("languages", {}))
        needed = list(REQUIRED_SYSTEM_KEYS) + (mapping.actions if mapping else [])
        for key in needed:
            for lang in langs:
                if not (messages.get(key) or {}).get(lang):
                    errors.append(f"voice.yaml: missing '{lang}' text for '{key}'")
                elif not clip_path(AUDIO_DIR, lang, key).is_file():
                    warnings.append(f"audio clip missing: {clip_path(AUDIO_DIR, lang, key)} (text fallback will be used)")
        print(f"OK   voice config {app.voice.name}: languages={langs}")
    except ConfigError as exc:
        errors.append(str(exc))

    for w in warnings:
        print(f"WARN {w}")
    for e in errors:
        print(f"ERROR {e}")
    print("RESULT:", "INVALID" if errors else "VALID")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
