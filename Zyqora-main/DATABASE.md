# Database (SQLite)

One file per device: `outputs/web/zyqora_web.sqlite3`, in WAL mode with `synchronous=NORMAL` and foreign keys on. Not committed. It is created and migrated automatically when the server starts, or by hand:

```bash
python scripts/init_db.py
```

Green Points (rewards) keep their own database, `outputs/greenscore.sqlite3` (`zyqora.greenscore`).

## Migrations

Migrations are tracked with `PRAGMA user_version`. Each numbered migration in `src/zyqora/api/detection_store.py::MIGRATIONS` runs once, in order. Current version: **1** (Zyqora AI 4.0 tables). The 3.0 `classifications` table is created with `CREATE TABLE IF NOT EXISTS` and is untouched, so existing 3.0 databases upgrade in place.

## Tables

| Table | Key columns |
|---|---|
| `classifications` (3.0, single-item) | predicted_category, confidence, ecoguard_status, disposal_action, inference_ms, model_profile, ground_truth, has_thumbnail |
| `detections` | created_at, source, filename, image_path, image_width/height, total_objects, waste_items, accepted_items, review_items, ignored_objects, whole_image_fallback, image_green_score, green_score_coverage, detector_inference_ms, classification_ms, inference_time_ms, pipeline_ms, detector_profile, classifier_profile, **model_version** (`classifier@sha12+detector`), label_profile, result_json (full API response, for exact re-display) |
| `detected_objects` | detection_id → detections **ON DELETE CASCADE**, obj_index, source (`detector`/`whole_image`), bounding_box (JSON xyxy, original pixels), detection_label, detection_confidence, classification_label (model label), classification_confidence, ecoguard_status, decision (ACCEPT/REVIEW/IGNORED/SKIPPED), material_category, material_certainty, food, organiser_category, recommended_disposal, conflict, review_reasons |
| `green_scores` | object_id → detected_objects **ON DELETE CASCADE**, recycling_potential, environmental_impact, disposal_suitability, final_score, band, basis, scoring_version (only scored items have a row) |
| `system_metrics` | timestamp, source (`detection`/`poll`), cpu_percent, memory_percent, process_rss_mb, inference_time_ms, temperature_c, measured_power_watts (NULL unless an external meter is added; never estimated) |
| `settings` | key, value (JSON), updated_at; dashboard overrides of config defaults |

Indexes:
- `detections(created_at)`
- `detected_objects(detection_id)` and `detected_objects(classification_label)`
- `green_scores(object_id)`
- `system_metrics(timestamp)`
- `classifications(created_at)` and `classifications(batch_id)`

## Retention

There are three ways to remove old data:
- **Settings → "Keep analyses for (days)":** stored as `retention_days`; the purge runs on save and every 50th analysis.
- **`POST /api/settings/purge?days=N`:** purges on demand.
- **`DELETE /api/detections/{id}`:** deletes one analysis.

Deleting an analysis removes its objects, scores and stored image. System samples are written at most every `metrics_sample_seconds` (default 30 s) while monitoring, plus one per analysis.

Stored images (`outputs/web/detections/<id>.jpg`) are downscaled to at most 1600 px. Annotated images are drawn on request and not stored.
