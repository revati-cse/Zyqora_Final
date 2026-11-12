# Two-minute demonstration

Rules: show only **real** outputs from the run on the Pi. Read the numbers from `summary.json` as they come out; do not prepare them in advance.
Before the session, follow README_PI.md sections 6–9 and 14. Have two terminal tabs open in `~/zyqora/Zyqora-main` with the venv active.

## Preparation (before judges arrive)
```bash
python scripts/check_config.py                   # RESULT: VALID, mapping status ORGANISER, threshold frozen
PYTHONPATH=src python -m zyqora.greenscore add-user --user-id DEMO01 --name "Demo user"
sudo nmcli radio wifi off && python scripts/check_offline.py   # RESULT: READY | OFFLINE
clear
```

## Script

| Time | Show | Say (key points) |
|---|---|---|
| **0:00–0:15** | Pi 5 on the table. Terminal: `nmcli radio wifi; python scripts/check_offline.py` | "Mixed waste ruins recycling. Zyqora tells a person which bin to use, fully offline, on a Raspberry Pi 5. Wi-Fi is off." |
| **0:15–0:50** | `python run_zyqora.py --images data/organiser --labels data/organiser/labels.csv --greenscore-user DEMO01 --out outputs/final` with per-image lines scrolling | "A pretrained MobileNetV2, exported with ExecuTorch and accelerated by the XNNPACK CPU backend; the whole network runs in one XNNPACK delegate. MaterialWise maps ImageNet classes to the organiser's categories by adding up class probabilities." |
| **0:50–1:15** | Summary block printed at the end (images, correct, accuracy, ACCEPT/REVIEW, median ms, model MB, memory) | Read the real numbers aloud. "Every image is in the accuracy denominator, including the ones EcoGuard sent to review." |
| **1:15–1:35** | Open `outputs/final/annotated/` (two images: ideally one green ACCEPT, one orange REVIEW) | "EcoGuard only recommends a bin when the category is supported and confident. Otherwise it asks a supervisor, and says why." If both samples are ACCEPT (or both REVIEW), say so; do not stage one. |
| **1:35–1:50** | `PYTHONPATH=src python -m zyqora.voicesort --lang ta --say RECYCLE` (then `--lang en`) | "VoiceSort gives the instruction in Tamil or English from local audio clips, with no internet." |
| **1:50–2:00** | `PYTHONPATH=src python -m zyqora.greenscore verify --event-id <id of an ACCEPT event> --observed RECYCLE --supervisor S1 --voice en` then `... points --user-id DEMO01` | "Points come only after a supervisor checks the real bin. Verifying again is refused." (Optional: re-run verify to show `already verified`.) |

To find an ACCEPT event id quickly: `PYTHONPATH=src python -m zyqora.greenscore events --status PENDING | grep ACCEPT | head -1`.

## Fallbacks
- No speaker: VoiceSort prints the Tamil/English text. Say "the clip is played with aplay; the speaker isn't connected".
- Run too long: add `--quiet` (summary only), or demo on a 10-image subset folder.
- Model error: run `python scripts/check_offline.py` and show the clear error rather than switching to any other runtime.

## After the demo
```bash
sudo nmcli radio wifi on
```
