# Green Score (Zyqora AI 4.0)

**Green Score** is a rule-based, project-defined indicator of how environmentally favourable the *potential handling* of a waste item is. It considers recycling or composting potential, environmental impact and disposal suitability. It is **not** a certified environmental assessment or a lifecycle assessment (LCA), and it is not a measurement.

It is separate from **Green Points** (`/rewards`, `zyqora.greenscore`). Green Points reward *people* for supervisor-verified disposals.

## Formula

```
Green Score = recycling_potential × 0.40 + environmental_impact × 0.35 + disposal_suitability × 0.25
```

Each component is a 0–100 rating, and higher is more favourable:

| Component | Meaning |
|---|---|
| Recycling / composting potential (40 %) | how much of the material can realistically be recovered if it reaches the right stream |
| Environmental impact rating (35 %) | how favourable the environmental outcome of handling it is (low = hazardous, persistent or hard to process) |
| Disposal suitability (25 %) | how specific and widely available the recommended stream is |

Example (also a unit test): 90 × 0.40 + 80 × 0.35 + 100 × 0.25 = **89 / 100**, which is "High sustainability potential".

Bands (configurable):

| Score | Band |
|---|---|
| 80–100 | High |
| 60–79 | Moderate-to-high |
| 40–59 | Moderate |
| 20–39 | Low |
| 0–19 | Very low |

## Rules (`config/greenscore_rules.yaml`, version `gs-rules-1`)

Ratings are chosen **per material** when the detector gives a likely material. Otherwise they are chosen per waste-model class (material unknown). Each rating has a one-line rationale in the YAML file, shown in the dashboard.

| Basis | RP | EI | DS | Score |
|---|---|---|---|---|
| Metal | 95 | 80 | 85 | 87.25 |
| Organic | 85 | 75 | 90 | 82.75 |
| Cardboard | 85 | 70 | 85 | 79.75 |
| Paper | 80 | 70 | 85 | 77.75 |
| Glass | 85 | 70 | 75 | 77.25 |
| Organic (material unknown) | 80 | 70 | 80 | 76.50 |
| Recyclable (material unknown) | 65 | 55 | 60 | 60.25 |
| E-Waste | 60 | 30 | 70 | 52.00 |
| Plastic | 50 | 35 | 60 | 47.25 |
| Other (textiles, composites) | 20 | 30 | 40 | 28.50 |

## When an item is NOT scored

| Reason code | Meaning |
|---|---|
| `CLASSIFICATION_NOT_RELIABLE` | EcoGuard REVIEW (score below the threshold) |
| `DETECTOR_CLASSIFIER_CONFLICT` | e.g. the detector saw a banana, the classifier said Recyclable |
| `NOT_A_WASTE_ITEM` | detector class not in the material table (person, chair, …) |
| `NO_RULE_FOR_CATEGORY` | no material or class rule |
| `MISSING_INFORMATION` | a component is missing (API calculator) |

The **image score** is the mean of the scored items. It is reported with its coverage ("2 of 3 items scored"); with no scored item there is no image score. Unknown items are never scored as zero.

## What it deliberately does not do

- It does **not** use the model's confidence. The engine has no confidence input; this is tested.
- It does **not** fabricate impact measurements. All values are ordinal team ratings with written reasons.
- It does **not** treat recyclable as harmless. Plastic scores 47 and e-waste 52, despite being "recyclable" streams.
- It does **not** know the item's condition: dirty or greasy items, mixed materials and local acceptance rules are not visible. The disposal guidance warns about these.

## Limitations

- The ratings are assumptions chosen by the team, not peer-reviewed values. Different weights or ratings would give different scores. Change them only in `config/greenscore_rules.yaml`, which bumps `version`; the version is stored with each score.
- Material comes from the COCO detector hint, which is often `Unknown`. A waste-trained detector (DETECTION.md, TACO) would make the material-level rules apply more often.
- Not validated against any LCA database or expert panel.

API: `POST /api/greenscore` (calculator), `GET /api/greenscore/rules`. Code: `src/zyqora/greenscore_engine.py`. Tests: `tests/test_mixed.py::GreenScoreTests`.
