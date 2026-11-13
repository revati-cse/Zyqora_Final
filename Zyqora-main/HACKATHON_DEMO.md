# Zyqora AI 3.0: judge demonstration (3–5 minutes)

**Ground rules.** Every number shown must come from the Pi during or before the session: the dashboard, `summary.json`, or the benchmark JSON. If something hasn't been measured yet, say "not measured yet". The dashboard does this itself and never fills gaps with sample data. Say "simulation" whenever the sorter or camera is simulated; the dashboard labels it.

## Setup (before judges arrive)
On the Pi:
```bash
cd /home/pi/Zyqora_Final/Zyqora-main && source .venv/bin/activate
```
```bash
python scripts/check_config.py
```
That should end with `RESULT: VALID`.
```bash
python scripts/verify_waste_model.py --images <VAL sample with O/ R/> --expected-sha256 <Mac hash>
```
That should end with `RESULT: PASSED`. Use `--activate` once to make the waste model the default.
```bash
bash scripts/start_pi_demo.sh
```
This prints `Open on a laptop in the same network: http://<pi-ip>:8000`.

On the laptop, open `http://<pi-ip>:8000`. The top bar must show **PI CONNECTED** and **MODEL READY**.

Checks:
- [ ] **Model Performance → Loaded model** shows profile `waste_mobilenetv2`, output `[1,2]`, classes Organic/Recyclable, and a SHA-256 equal to the Mac export (proof that the trained model, not the ImageNet one, is running).
- [ ] **Device Monitoring** shows a Raspberry Pi 5 model string and a CPU temperature.
- [ ] Settings → GreenScore participant set (e.g. `DEMO01`; add it on the GreenScore page first). Voice = "Play in this browser" if the laptop has speakers, or "Play on the device speaker".
- [ ] Demo images ready on the laptop: two clear organic items, two clear recyclable items, and one item **you have already seen produce REVIEW on this Pi** (do not stage one).
- [ ] Offline: if you want to show it, put the Pi and laptop on a local hotspot/router with no internet uplink (see README_PI §14). Disabling the Pi's Wi-Fi would also cut the laptop off, unless they're connected by Ethernet.
- [ ] Fallback: a USB stick or folder with the same images, in case the browser camera isn't allowed (plain http).

## Script
| Time | Page | Show | Say |
|---|---|---|---|
| 0:00–0:30 | **Overview** | Status pills, live counters | "Zyqora sorts waste at the bin, offline. A MobileNetV2 we fine-tuned runs on this Raspberry Pi 5. The laptop is only the screen; the model runs on the Pi." |
| 0:30–1:10 | **Model Performance** | Pipeline strip, loaded model card, validation card | "MobileNetV2 from ImageNet weights, fine-tuned on two classes, exported to ExecuTorch with XNNPACK CPU kernels: an 8.47 MiB file. Validation accuracy 96.65% on 2,000 images. That's validation, not test." Then read the **TEST** card: real numbers if the evaluation was run, otherwise "not measured yet". |
| 1:10–2:10 | **Live Classification** | Upload an organic image → **Classify Waste**; then a recyclable one | Read the card: category, score, ACCEPT, and the **inference time measured on the Pi**. Point at "ExecuTorch + XNNPACK · Raspberry Pi 5" in the footer. Optionally switch to தமிழ் and press **Play instruction**. |
| 2:10–2:50 | **Live Classification** | The image you know gives REVIEW | "Below 95% confidence, EcoGuard sends the item to a supervisor: no bin opens. It is a decision, not a third class." Be honest: "The model only knows Organic and Recyclable. A residual item can still be forced into one of them, which is our next step: a Residual class." |
| 2:50–3:30 | **GreenScore** | The pending event from the classification → **Verify** (bin placed, correct bin) | "Points come only after a supervisor checks the real bin. Verifying again is refused." |
| 3:30–4:00 | **Device Monitoring** | CPU, RAM, temperature from the Pi | "Telemetry comes straight from the Pi's /proc; the model is loaded once and inference is serialized, so the Pi stays cool." |
| 4:00–4:30 | **History / Analytics** | The items just classified | "Everything is stored locally in SQLite. Nothing leaves the device." |
| 4:30–5:00 | Wrap-up | — | What is new is the combination, all running on the Pi: offline edge AI, a small model, confidence-aware review, bilingual guidance, verified rewards, and transparent reporting of what is measured and what isn't. |

## If something fails
| Problem | What to do |
|---|---|
| MODEL UNAVAILABLE | Read the code on the Live page (`MODEL_NOT_FOUND` / `MODEL_CONTRACT_MISMATCH` / `MODEL_HASH_MISMATCH`). Fix it with README_PI §0.4, then **Settings → Reload model on device** (or `curl -X POST http://<pi-ip>:8000/api/model/reload`). Do not demo with another model. |
| Browser camera blocked | Expected on plain http. Use **Capture with device camera** (Pi camera) or upload a photo. |
| Laptop can't reach the Pi | Same network? `hostname -I` on the Pi; `curl http://<pi-ip>:8000/api/health` from the laptop; check `sudo ufw status` if a firewall is enabled. |
| No sound | Switch Voice in Settings (browser vs device speaker); the text guidance is always shown. |
| A prediction is wrong | Say so. EcoGuard and supervisor verification exist for this; GreenScore never awards points for an unverified prediction. |

## Talking points (all implemented)
- **Offline-first edge AI.** Inference, history, voice clips and fonts are all local; no cloud calls.
- **Small and efficient.** One 8.47 MiB FP32 `.pte`, loaded once, with XNNPACK kernels and a configurable thread count. The latency, CPU and memory figures quoted must come from `scripts/benchmark_pi.py` on the Pi.
- **Confidence-aware review.** A threshold of 0.95, provisional and chosen from the validation trade-off table on the Model Performance page.
- **Safe sorting.** REVIEW never actuates a bin. The actuator stays in simulation unless explicitly enabled.
- **Bilingual.** English and Tamil guidance in the UI and offline audio clips. The Tamil wording still awaits review by a native speaker.
- **Verify-to-earn.** GreenScore points only after supervisor verification, once per event.
- **Transparent reporting.** Every metric shows its source: validation (team-reported), TEST, Pi benchmark, or live.
