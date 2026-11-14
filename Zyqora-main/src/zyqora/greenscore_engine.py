"""Green Score: a rule-based, project-defined sustainability indicator (0-100).

    Green Score = recycling_potential x 0.40 + environmental_impact x 0.35 + disposal_suitability x 0.25

Component ratings come from config/greenscore_rules.yaml (ordinal team ratings with a
written rationale; NOT measurements). The model's confidence is never an input: an item
is either reliably identified (ACCEPT, no conflict) and scored, or it is not scored.

This is separate from zyqora.greenscore, which awards *user reward points* for
supervisor-verified disposals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import ConfigError, load_yaml
from .paths import CONFIG_DIR

DEFAULT_RULES = CONFIG_DIR / "greenscore_rules.yaml"
COMPONENTS = ("recycling_potential", "environmental_impact", "disposal_suitability")
COMPONENT_TITLES = {
    "recycling_potential": "Recycling / composting potential",
    "environmental_impact": "Environmental impact rating",
    "disposal_suitability": "Disposal suitability",
}

# reasons an item is not scored
NS_NOT_WASTE = "NOT_A_WASTE_ITEM"
NS_REVIEW = "CLASSIFICATION_NOT_RELIABLE"
NS_CONFLICT = "DETECTOR_CLASSIFIER_CONFLICT"
NS_NO_RULE = "NO_RULE_FOR_CATEGORY"
NS_MISSING = "MISSING_INFORMATION"


@dataclass(frozen=True)
class Component:
    key: str
    value: float
    weight: float
    why: str = ""

    @property
    def weighted(self) -> float:
        return self.value * self.weight

    def as_dict(self) -> dict:
        return {"key": self.key, "title": COMPONENT_TITLES.get(self.key, self.key), "value": self.value,
                "weight": self.weight, "weighted": round(self.weighted, 4), "rationale": self.why}


@dataclass
class GreenScoreResult:
    scored: bool
    score: Optional[float] = None
    band: Optional[str] = None
    components: list[Component] = field(default_factory=list)
    basis: Optional[str] = None             # "material:Metal" | "model_class:Recyclable" | "custom"
    not_scored_reason: Optional[str] = None
    explanation: str = ""
    scoring_version: str = ""

    def as_dict(self) -> dict:
        return {"scored": self.scored, "score": self.score, "band": self.band, "basis": self.basis,
                "components": [c.as_dict() for c in self.components], "not_scored_reason": self.not_scored_reason,
                "explanation": self.explanation, "scoring_version": self.scoring_version,
                "formula": "recycling_potential x 0.40 + environmental_impact x 0.35 + disposal_suitability x 0.25"}


def _rating(d, where: str) -> tuple[float, str]:
    if isinstance(d, dict):
        v, why = d.get("value"), str(d.get("why", ""))
    else:
        v, why = d, ""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 100:
        raise ConfigError(f"{where}: rating must be a number in [0, 100], got {v!r}")
    return float(v), why


class GreenScoreEngine:
    def __init__(self, rules: dict, source: Path | str = "<dict>"):
        self.source = str(source)
        self.version = str(rules.get("version") or "unversioned")
        w = rules.get("weights") or {}
        if set(w) != set(COMPONENTS):
            raise ConfigError(f"{source}: weights must define exactly {COMPONENTS}")
        self.weights = {k: float(w[k]) for k in COMPONENTS}
        if any(v < 0 for v in self.weights.values()) or abs(sum(self.weights.values()) - 1.0) > 1e-9:
            raise ConfigError(f"{source}: weights must be non-negative and sum to 1.0, got {self.weights}")
        bands = rules.get("bands") or []
        try:
            self.bands = sorted(((float(b["min"]), str(b["label"])) for b in bands), reverse=True)
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"{source}: each band needs min and label") from exc
        if not self.bands or self.bands[-1][0] > 0:
            raise ConfigError(f"{source}: bands must include a band starting at 0")
        self.materials = self._table(rules.get("materials") or {}, "materials")
        self.model_classes = self._table(rules.get("model_classes") or {}, "model_classes")

    def _table(self, raw: dict, name: str) -> dict[str, dict[str, tuple[float, str]]]:
        out = {}
        for key, comps in raw.items():
            if not isinstance(comps, dict) or set(comps) != set(COMPONENTS):
                raise ConfigError(f"{self.source}: {name}.{key} must rate exactly {COMPONENTS}")
            out[key] = {c: _rating(comps[c], f"{self.source}: {name}.{key}.{c}") for c in COMPONENTS}
        return out

    @classmethod
    def load(cls, path: Path | str = DEFAULT_RULES) -> "GreenScoreEngine":
        return cls(load_yaml(path), path)

    # ---------------------------------------------------------------- core formula
    def band(self, score: float) -> str:
        for lo, label in self.bands:
            if score >= lo:
                return label
        return self.bands[-1][1]

    def score_components(self, recycling_potential: float, environmental_impact: float,
                         disposal_suitability: float, *, basis: str = "custom",
                         why: Optional[dict[str, str]] = None) -> GreenScoreResult:
        vals = {"recycling_potential": recycling_potential, "environmental_impact": environmental_impact,
                "disposal_suitability": disposal_suitability}
        for k, v in vals.items():
            if v is None:
                return GreenScoreResult(False, not_scored_reason=NS_MISSING, explanation=f"{COMPONENT_TITLES[k]} is missing.",
                                        scoring_version=self.version)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 100:
                raise ValueError(f"{k} must be a number in [0, 100], got {v!r}")
        comps = [Component(k, float(vals[k]), self.weights[k], (why or {}).get(k, "")) for k in COMPONENTS]
        score = round(sum(c.weighted for c in comps), 2)
        band = self.band(score)
        parts = " + ".join(f"{c.value:g} x {c.weight:.2f}" for c in comps)
        return GreenScoreResult(True, score, band, comps, basis,
                                explanation=f"{parts} = {score:g}/100 ({band}).", scoring_version=self.version)

    # ---------------------------------------------------------------- items
    def score_item(self, *, material: Optional[str], material_certainty: Optional[str], model_class: Optional[str],
                   decision: Optional[str], waste_item: bool = True, conflict: bool = False) -> GreenScoreResult:
        if not waste_item:
            return self._ns(NS_NOT_WASTE, "Not a waste item; not scored.")
        if decision != "ACCEPT":
            return self._ns(NS_REVIEW, "The item was not classified reliably (REVIEW), so it is not scored.")
        if conflict:
            return self._ns(NS_CONFLICT, "Detector and classifier disagree about this item; not scored until reviewed.")
        if material and material_certainty == "likely" and material in self.materials:
            table, basis = self.materials[material], f"material:{material}"
        elif model_class and model_class in self.model_classes:
            table, basis = self.model_classes[model_class], f"model_class:{model_class}"
        else:
            return self._ns(NS_NO_RULE, f"No Green Score rule for material {material!r} / class {model_class!r}.")
        return self.score_components(*(table[c][0] for c in COMPONENTS), basis=basis,
                                     why={c: table[c][1] for c in COMPONENTS})

    def _ns(self, reason: str, text: str) -> GreenScoreResult:
        return GreenScoreResult(False, not_scored_reason=reason, explanation=text, scoring_version=self.version)

    # ---------------------------------------------------------------- image aggregate
    def aggregate(self, results: list[GreenScoreResult], *, waste_items: Optional[int] = None) -> dict:
        """Image-level score = mean of scored items. Valid only when at least one item is scored."""
        scored = [r for r in results if r.scored and r.score is not None]
        total = len(results) if waste_items is None else waste_items
        if not scored:
            return {"valid": False, "score": None, "band": None, "scored_items": 0, "waste_items": total,
                    "coverage": 0.0 if total else None, "scoring_version": self.version,
                    "explanation": "No item was reliably identified, so no image Green Score is given."}
        mean = round(sum(r.score for r in scored) / len(scored), 2)
        cov = len(scored) / total if total else None
        note = "" if len(scored) == total else f" {total - len(scored)} item(s) were not scored (see each item)."
        return {"valid": True, "score": mean, "band": self.band(mean), "scored_items": len(scored), "waste_items": total,
                "coverage": cov, "partial": len(scored) < total, "scoring_version": self.version,
                "min": min(r.score for r in scored), "max": max(r.score for r in scored),
                "explanation": f"Mean of {len(scored)} scored item(s) = {mean:g}/100.{note}"}

    def describe(self) -> dict:
        return {"version": self.version, "source": self.source, "weights": self.weights,
                "bands": [{"min": lo, "label": label} for lo, label in sorted(self.bands)],
                "materials": {m: {c: {"value": v, "why": w} for c, (v, w) in t.items()} for m, t in self.materials.items()},
                "model_classes": {m: {c: {"value": v, "why": w} for c, (v, w) in t.items()} for m, t in self.model_classes.items()},
                "disclaimer": "Project-defined indicator from qualitative team ratings; not a measured or certified "
                              "environmental assessment."}
