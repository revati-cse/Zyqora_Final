# Zyqora AI API

FastAPI app in `src/zyqora/api/`, started with `serve_zyqora.py`. Interactive docs are served at `/docs` on the running server.

- The model is loaded **once per process**. Inference is serialized behind a lock, and a request that would be the fifth one waiting gets `SERVER_BUSY`.
- The API calls the same code as the CLI: `zyqora.profiles` → `zyqora.pipeline` → decoder → EcoGuard → `zyqora.sorter`, `zyqora.greenscore`, `zyqora.voicesort`. There is no second inference implementation.
- If the model cannot be loaded, every inference returns an explicit error code. **No other model or mock result is substituted** unless the server was started with `--allow-fallback`, and then every response has `model.fallback_used: true`.
- There is no authentication. Only run it on a trusted local network. CORS is off unless origins are listed (`--cors-origin` or `ZYQORA_CORS_ORIGINS`).

## Errors
Every error has the same shape and HTTP status:
```json
{"error": {"code": "MODEL_NOT_FOUND", "message": "Model file not found: models/mobilenetv2_waste_fp32_xnnpack.pte", "details": {"expected_pte": "..."}}}
```
| Code | HTTP | Meaning |
|---|---|---|
| `MODEL_NOT_FOUND` | 503 | The profile's `.pte` file is missing |
| `MODEL_CONTRACT_MISMATCH` | 503 | The `.pte` output/input shape doesn't match its profile (e.g. the ImageNet model under the waste profile) |
| `MODEL_HASH_MISMATCH` | 503 | The file differs from the SHA-256 pinned in `config/models.yaml` |
| `MODEL_LOAD_FAILED`, `RUNTIME_UNAVAILABLE`, `CONFIG_ERROR` | 503 | ExecuTorch could not load the program / is not installed / bad config |
| `INVALID_IMAGE`, `EMPTY_FILE` | 422 | Not a decodable JPEG/PNG/WebP/BMP |
| `UNSUPPORTED_MEDIA_TYPE` | 415 | Upload content type is not an image |
| `FILE_TOO_LARGE`, `TOO_MANY_FILES` | 413 | Over `--max-upload-mb` (default 10) / over 50 files per batch |
| `SERVER_BUSY` | 503 | Too many classifications waiting; retry |
| `INVALID_USER_ID`, `INVALID_GROUND_TRUTH`, `VALIDATION_ERROR` | 422 | Bad form or JSON fields |
| `CAMERA_UNAVAILABLE` | 503 | No camera backend could capture a frame |
| `ALREADY_VERIFIED` | 409 | GreenScore event was already verified |
| `GREENSCORE_ERROR` | 409/422 | Other GreenScore rule violation (e.g. REVIEW item verified without a correct bin) |
| `NOT_FOUND` | 404 | Unknown route or id |
| `DETECTOR_NOT_FOUND`, `DETECTOR_HASH_MISMATCH`, `DETECTOR_CONTRACT_MISMATCH`, `DETECTOR_LOAD_FAILED`, `DETECTOR_DISABLED` | 503 | YOLO detector missing / differs from its pin in `config/detector.yaml` / wrong output shape / runtime error / started with `--no-detector`. Single-item endpoints keep working. |
| `DETECTOR_INFERENCE_FAILED`, `INFERENCE_FAILED` | 500 | A model ran but produced an unusable output (never replaced by a fake result) |
| `UNSUPPORTED_CLASSIFIER` | 503 | Mixed-waste mode needs the waste classifier profile (`direct_classes`) |

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Server, model readiness (`model_ready`, `model_code`), device model, `is_raspberry_pi` |
| GET | `/api/model/status` | Loaded profile, `.pte` name/size/SHA-256, input/output shape, preprocessing, classes, ExecuTorch version, XNNPACK, threads, EcoGuard settings |
| POST | `/api/model/reload` | Retry loading (e.g. after copying the `.pte`) |
| GET | `/api/model/metrics` | `validation` (team-reported, `config/model_metrics.json`), `test` (from `outputs/test_waste/summary.json` if it exists), `benchmark` (from `outputs/benchmark/` if it exists), `live` (this server's history). Each has `scope` and `source`. |
| POST | `/api/inference` | multipart: `file` (required), `user_id`, `batch_id`, `ground_truth`, `source` (`upload`/`browser-camera`/`batch`) → classification |
| POST | `/api/inference/batch` | multipart: `files[]`, optional `ground_truths` (JSON `{filename: category}`), `user_id` → `{batch_id, results, errors, summary}` |
| GET | `/api/batches/{batch_id}` | Items + summary (labelled metrics only if labels were given) |
| GET | `/api/camera/status` · POST `/api/camera/capture` | Device camera (`config/hardware.yaml` or `--camera`); capture classifies one frame |
| GET | `/api/history` | `limit`, `offset`, `category`, `status` (ACCEPT/REVIEW), `date_from`, `date_to` (ISO), `batch_id`, `source` |
| GET | `/api/history/{id}` · `/api/history/{id}/image` | One record / its stored 320 px thumbnail |
| GET | `/api/history/export.csv` | CSV export with the same filters |
| GET | `/api/analytics/summary` | Totals, ACCEPT/REVIEW, by category/day, latency stats and series, batch throughput, labelled accuracy (only when labels exist) |
| GET | `/api/device/metrics` | CPU % (from `/proc/stat` deltas), RAM, CPU temperature, uptime, load average, process RSS/peak, runtime/backend/threads, last inference time, camera/sorter state. Unavailable values are `null`. |
| GET | `/api/greenscore` · `/api/greenscore/{user_id}` | Participants / one participant's events, counts and points by bin |
| POST | `/api/greenscore/users` | `{user_id, display_name}` |
| POST | `/api/greenscore/verify` | `{event_id, observed_action, correct_action?, supervisor_id?}`. Points only if observed bin == correct bin; once per event. |
| GET | `/api/voice/messages` · `/api/voice/{en|ta}/{KEY}.wav` | Instruction text / offline WAV clip |
| POST | `/api/voice/play` | `{lang, key}`: play the clip on the device speaker (`aplay`) |

### Zyqora AI 4.0 (mixed-waste detection)
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/models/status` | `classifier` (as `/api/model/status`), `detector` (profile, file, size, runtime, imgsz, load + warm-up ms, licence, `trained_on`), `label_profiles`, `greenscore_rules`, `pipeline.ready` |
| POST | `/api/models/reload` | Reload classifier and detector |
| POST | `/api/detect` | multipart: `file`, `user_id?`, `source` (`upload`/`browser-camera`/`camera-frame`), `persist` (default true) → **detection result**: `objects[]` (box in original pixels, `detection_label` + confidence, `classification.model_label` + score + EcoGuard, `material`, `conflict`, `organiser_category`, `organiser_labels` for every profile, `recommendation`, `green_score` with components), `summary` (counts by organiser label / profile / material / stream), `image_green_score`, `timing` (detector pre/inference/post, classification, pipeline ms), `notes`, `image_url`, `annotated_url`. Never actuates a bin. |
| POST | `/api/classify` | Alias of `/api/inference` (single item) |
| POST | `/api/greenscore` | JSON: `{components: {recycling_potential, environmental_impact, disposal_suitability}}` or `{material}` or `{model_class, decision}` → Green Score with breakdown (GREEN_SCORE.md) |
| GET | `/api/greenscore/rules` | Weights, bands, every rating with its rationale, disclaimer |
| GET | `/api/detections` | `limit`, `offset`, `q` (file / object / label / material), `material`, `model_label`, `has_review`, `min_score`, `date_from`, `date_to` |
| GET · DELETE | `/api/detections/{id}` | Full stored result / delete it (objects, scores and image cascade) |
| GET | `/api/detections/{id}/image` · `/annotated.jpg` | Stored image (≤1600 px) / boxes drawn server-side |
| GET | `/api/detections/export.csv` | One row per object |
| DELETE | `/api/history/{id}` | Delete one single-item classification |
| GET | `/api/history` · `/api/history/{id}` | (3.0) single-item history; mixed-waste history is `/api/detections` |
| GET | `/api/analytics` | `days?` → `{classifications, detections}`; detection analytics: counts, per-profile distributions, material, streams, daily, Green Score series, latency series, stored system samples |
| GET | `/api/system/metrics` | Device telemetry + classifier/detector status and load times + `pmic` (Pi 5 rail readings) + `measured_power_watts` (`null` unless metered). Stores a sample at most every `metrics_sample_seconds`. |
| GET | `/api/system/history` | Stored system samples (`days`) |
| GET | `/api/benchmarks` | All benchmark / evaluation JSON (committed `reports/` + local `outputs/`), each with its `platform` |
| GET · PUT | `/api/settings` | Effective settings, overrides, defaults, options. PUT a partial object (`detector_profile`, `label_profile`, `conf_threshold`, `iou_threshold`, `agnostic_nms`, `max_detections`, `max_objects_classified`, `crop_padding`, `min_box_area_fraction`, `whole_image_fallback`, `retention_days`, `metrics_sample_seconds`; `null` resets a key). Validated; 422 on bad values. |
| POST | `/api/settings/purge?days=N` | Delete analyses and system samples older than N days |

### Classification response (abridged)
```json
{
  "id": 12, "filename": "bottle.jpg", "source": "upload",
  "predicted_category": "Recyclable", "confidence": 0.991,
  "class_scores": {"Organic": 0.009, "Recyclable": 0.991},
  "score_note": "softmax score of the model output; not a calibrated probability",
  "ecoguard": {"status": "ACCEPT", "threshold": 0.95, "reasons": [], "explanation": {"en": "", "ta": ""}},
  "disposal_action": "RECYCLE",
  "guidance": {"en": "Place this item in the recycling bin.", "ta": "இந்தப் பொருளை மறுசுழற்சித் தொட்டியில் போடவும்."},
  "sort": {"action": "RECYCLE", "actuate": true, "message": "[SIMULATION] would open the RECYCLE bin", "hardware": false},
  "timing": {"preprocess_ms": 4.6, "inference_ms": 3.7, "total_ms": 9.0},
  "model": {"profile": "waste_mobilenetv2", "pte": "mobilenetv2_waste_fp32_xnnpack.pte", "sha256": "...", "runtime": "ExecuTorch",
            "backend": "XNNPACK", "fallback_used": false, "device": {"device_model": "Raspberry Pi 5 ...", "is_raspberry_pi": true}},
  "greenscore_event_id": null, "image_url": "/api/history/12/image"
}
```
A REVIEW result keeps `predicted_category` (for evaluation), has `disposal_action: "MANUAL_REVIEW"` and `sort.actuate: false`, and includes the reasons in English and Tamil.

## Examples

Mixed-waste analysis:
```bash
curl -F file=@mixed.jpg http://<pi-ip>:8000/api/detect
```
Green Score calculator:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"components":{"recycling_potential":90,"environmental_impact":80,"disposal_suitability":100}}' http://<pi-ip>:8000/api/greenscore
```

```bash
curl -s http://<pi-ip>:8000/api/model/status
```
```bash
curl -s -F "file=@bottle.jpg;type=image/jpeg" http://<pi-ip>:8000/api/inference
```
```bash
curl -s -X POST -H 'Content-Type: application/json' -d '{"event_id": 3, "observed_action": "RECYCLE"}' http://<pi-ip>:8000/api/greenscore/verify
```

## Storage
- History: `outputs/web/zyqora_web.sqlite3` (SQLite, WAL), thumbnails in `outputs/web/thumbs/`, server log `outputs/web/server.log`.
- GreenScore: `outputs/greenscore.sqlite3`, the same database as the CLI.

All of these are local, git-ignored, and never sent anywhere.
