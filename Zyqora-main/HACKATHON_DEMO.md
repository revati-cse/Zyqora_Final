# Zyqora AI 4.0: judge demonstration (4–6 minutes)

**Ground rules.**
- Every number shown must come from the device during or before the session: the dashboard, `reports/`, `summary.json` or the benchmark JSON.
- If something hasn't been measured, say "not measured yet". The dashboard does this itself and never fills gaps with sample data.
- Say "simulation" whenever the sorter or camera is simulated.
- Say clearly that **the detector is a pretrained COCO model, not trained on waste**.

## Setup (before judges arrive)
On the Pi (README_PI §17), or on a Windows laptop (WINDOWS_SETUP.md):
```bash
cd /home/pi/Zyqora_Final/Zyqora-main && source .venv/bin/activate && python scripts/check_config.py
```
```bash
bash scripts/start_pi_demo.sh
```
Open `http://<pi-ip>:8000`. The top bar must show **PI CONNECTED** and **MODEL READY**; Overview must show **Detector READY** and **Classifier READY**.

Checklist:
- [ ] **Model Comparison**: the classifier profile is `waste_mobilenetv2` with the pinned SHA-256; detector `yolo11n_640`; the VOC evaluation table is visible. If you ran `scripts/benchmark_detector.py` on the Pi, its rows say "Raspberry Pi 5".
- [ ] **Settings → Mixed-waste detection**: label profile chosen. Use *Organic / Recyclable* (organiser labels) unless the organisers asked for Recyclable/Non-Recyclable or Food/Waste.
- [ ] Demo photos on the laptop:
  - one **mixed** photo you have already analysed on this device (several separate, clearly visible items, e.g. a bottle, a glass, a banana, a phone);
  - one single organic item;
  - one item that **you have already seen give REVIEW / Unknown** (do not stage one).
- [ ] Avoid relying on cans, plastic bags or cardboard as separate boxes: COCO has no class for them. If one is missed, say why (DETECTION.md).

## Script
| Time | Page | Show | Say |
|---|---|---|---|
| 0:00–0:30 | **Hackathon Showcase** | Problem / limitation / solution cards, architecture strip | "A one-label classifier fails on a mixed photo. Zyqora finds every object, classifies each one on the Pi, and refuses to guess when unsure." |
| 0:30–2:00 | **AI Detection** | Upload the mixed photo → **Analyse waste** | Point at the boxes, then at each object card: model label, organiser label, material, disposal stream, Green Score. Read **Pipeline on device** (measured). "The detector is pretrained on COCO, so its labels are only material hints; the waste label comes from our fine-tuned MobileNetV2." |
| 2:00–2:40 | **AI Detection** | Click an Unknown / review item, or a non-waste box | "Low confidence, or a disagreement between the detector and the classifier, gives Unknown and no score: ask a supervisor. A person in the frame is drawn but never classified." If a TV or phone is in the photo: "The classifier alone would say Recyclable; electronics go to e-waste." |
| 2:40–3:30 | **Green Score** | Gauge, item list, the weighted table; calculator with 90/80/100 → 89 | "Project-defined indicator, 40/35/25 weights, every rating has a written reason. It never uses model confidence, and it is not a lifecycle assessment." |
| 3:30–4:20 | **Model Comparison** | Comparison table, VOC chart | "YOLO11n at 640 was chosen by held-out accuracy: VOC mAP@0.5 0.832 vs 0.813 for YOLOv8n. Classifier: 96.6% validation, reproduced on this Pi at 12.7 ms. Waste-detection mAP is not measured; we had no annotated waste test set." |
| 4:20–5:00 | **System Monitoring / Analytics** | CPU, RAM, temperature, PMIC rails; charts of what was just analysed | "Everything is stored locally in SQLite, nothing leaves the device. We don't convert CPU load into watts; real power needs a meter." |
| 5:00–5:30 | **Live Camera** (only if opened on the Pi itself or over https) | Continuous mode | Read the measured FPS and round-trip time; they are real, not targets. |

## If something fails
| Problem | What to do |
|---|---|
| DETECTOR UNAVAILABLE | Read the code (README_PI §17). Single-item classification still works: use **Single-item Classification**. |
| MODEL UNAVAILABLE | README_PI §0.4, then **Settings → Reload model on device**. Do not demo with another model. |
| An object is missed | It is probably not a COCO class. Say so; the TACO fine-tuning pipeline exists but was not run (approval + GPU needed). |
| A prediction is wrong | Say so. EcoGuard, Unknown and supervisor verification exist for this. |
| Camera blocked | Expected on plain http from a laptop; use a photo. |

## Talking points (all implemented and visible)
- Two-stage edge AI, YOLO → crop → MobileNetV2, both on ExecuTorch XNNPACK on the Pi's CPU. Offline, with no cloud.
- Organiser labels kept intact, with display profiles that rename and never invent.
- A transparent Green Score with per-component explanation, stored per item with its rules version.
- Honest metrics: every figure says where it was measured (Pi, host, validation, held-out VOC). Missing evaluations are labelled "not measured".
- Safety: Unknown instead of a confident wrong bin; mixed-waste mode never moves a bin; electronics go to e-waste.
