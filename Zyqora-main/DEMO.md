# Two-minute demonstration (offline)

Rules:
- Show only real output from the Pi.
- Read the numbers from the `summary.json` files produced on the Pi (TEST evaluation, benchmark). If a run hasn't happened yet, say "not measured yet"; never quote prepared figures.
- Say "simulation" whenever the camera or sorter is simulated. The station prints this itself.

## Preparation (before judges arrive)
```bash
cd /home/pi/Zyqora_Final/Zyqora-main && source .venv/bin/activate
```
```bash
python scripts/check_config.py
```
`check_config` should show `active=waste_mobilenetv2` (after `verify_waste_model.py --activate`) and end with `RESULT: VALID`.
```bash
PYTHONPATH=src python -m zyqora.greenscore add-user --user-id DEMO01 --name "Demo user"
```
```bash
sudo nmcli radio wifi off && python scripts/check_offline.py
```
That should print `RESULT: READY | OFFLINE`.

Have ready:
- **Real items**: an organic item (banana peel, leaves) and a recyclable one (bottle, paper), plus one residual item (chip packet, tissue) to show the limitation honestly.
- **Without a camera**: a folder `data/demo/` with 3–5 images in that order.
- The TEST evaluation summary `outputs/test_waste/summary.json` and the benchmark JSON, both from the Pi.

## Script
| Time | Show | Say |
|---|---|---|
| **0:00–0:15** | `nmcli radio wifi` → disabled; `check_offline.py` → READY, OFFLINE | "Zyqora sorts waste on a Raspberry Pi 5 with no internet. Organic or recyclable, in Tamil or English." |
| **0:15–0:50** | `python zyqora_station.py --camera auto --voice ta --user DEMO01` (or `--camera folder:data/demo`). Press Enter for the organic item, then the recyclable one. | "Our MobileNetV2, fine-tuned on Organic vs Recyclable, runs through ExecuTorch with the XNNPACK CPU backend. The card shows both class probabilities, the EcoGuard threshold and the decision." The Tamil instruction plays. |
| **0:50–1:15** | Press Enter for the residual or an ambiguous item | If REVIEW: "Below 95% confidence: no bin opens and a supervisor is asked." If it is wrongly ACCEPTed: "The model only knows two classes, so a residual item can be forced into one. That is our known limitation, and the next step is a Residual class." Say only what actually happened. |
| **1:15–1:40** | `outputs/test_waste/summary.json` key lines (`overall_accuracy`, per-class F1, `review_rate`, `accepted_only_accuracy`) and the benchmark table | Read the real TEST and Pi numbers. If they don't exist yet, say so. Mention that validation accuracy was 96.65% and that validation is not test. |
| **1:40–2:00** | `PYTHONPATH=src python -m zyqora.greenscore verify --event-id <id> --voice en`, then `... points --user-id DEMO01` | "Points only after a supervisor checks the real bin. Verifying again is refused." |

Find an ACCEPT event id with:
```bash
PYTHONPATH=src python -m zyqora.greenscore events --status PENDING
```

## Fallbacks
| Problem | Do |
|---|---|
| Camera fails | Restart the station with `--camera folder:data/demo` and say "simulated camera". |
| No speaker | The instruction text is printed. Say "the clip is played with aplay; no speaker is connected". |
| Waste model refused at load | The station uses the ImageNet fallback and prints `FALLBACK`. Say so, or stop and show the error. |
| Time is short | Skip the residual item; keep GreenScore. |

## After the demo
```bash
sudo nmcli radio wifi on
```
