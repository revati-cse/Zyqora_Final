# Zyqora AI on Raspberry Pi 5

> Status: these steps follow the Arm ExecuTorch course environment (Module 2, commit `1045d51`) and the wheels verified to exist for aarch64/Python 3.12. **Nobody has run them end-to-end on a Pi yet.** Record what actually happens in IMPLEMENTATION_STATUS.md.

## 1. Hardware
- Raspberry Pi 5 (any RAM size; 4 GB or more recommended because torch + ExecuTorch use several hundred MB of RAM)
- Official 27 W USB-C PSU, microSD (32 GB+) or NVMe
- Active Cooler recommended (check throttling with `vcgencmd get_throttled`)
- Optional: speaker or headphones (3.5 mm via a USB/HAT adapter, HDMI audio, or a USB speaker) for VoiceSort
- **No camera is required.** Zyqora classifies saved image files.

Record the actual RAM, cooling, storage, audio and camera with `bash scripts/pi_collect_facts.sh`.

## 2. Operating system
Use **Raspberry Pi OS (64-bit) Trixie**, the version the course targets. Lite is enough and saves RAM, since Zyqora is terminal-only. `uname -m` must print `aarch64`.

## 3. Python 3.12
Trixie ships a newer system Python. The pinned wheels target **Python 3.12**, so install 3.12.14 with pyenv as the course does:
```bash
sudo apt update
sudo apt install -y git build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libffi-dev liblzma-dev tk-dev curl alsa-utils espeak-ng
curl -fsSL https://pyenv.run | bash      # then add the printed lines to ~/.bashrc and open a new shell
pyenv install 3.12.14
pyenv shell 3.12.14
python --version                          # Python 3.12.14
```
`alsa-utils` provides `aplay` for VoiceSort. `espeak-ng` is only needed if you want to regenerate clips on the Pi.

## 4. Virtual environment
Reuse the course venv if it already exists. Otherwise create it:
```bash
mkdir -p ~/executorch_on_arm
python -m venv ~/executorch_on_arm/.cpu_lab_venv
source ~/executorch_on_arm/.cpu_lab_venv/bin/activate
python -m pip install --upgrade pip
```

## 5. ExecuTorch and dependencies
```bash
git clone <your fork of this repo> ~/zyqora && cd ~/zyqora/Zyqora-main
pip install -r requirements-pi.txt       # torch 2.12.1+cpu, executorch 1.3.1, torchao 0.17.0, pillow, numpy, pyyaml
python -c "from executorch.extension.pybindings import portable_lib as p; print(p._get_registered_backend_names())"
```
The last line must list `XnnpackBackend`. If the import fails, the venv is not Python 3.12 or the wheels are mismatched: delete the venv and redo steps 3–5.

## 6. Model artifact
Export it on the development machine (WSL2/Linux/macOS) or directly on the Pi (the same venv contains the export toolchain):
```bash
python scripts/prepare_model.py          # internet needed once: official TorchVision weights, hash-checked
python scripts/export_fp32.py            # models/mobilenetv2_fp32_xnnpack.pte (~14 MB) + .json
# optional INT8 (see MODEL_COMPARISON.md before using it)
python scripts/export_int8.py --calibration <calibration folder> --exclude data/organiser
```
To copy artifacts from the dev machine instead:
```bash
scp models/mobilenetv2_*_xnnpack.pte models/mobilenetv2_*_xnnpack.json pi@<pi-host>:~/zyqora/Zyqora-main/models/
```
`.pte` files are architecture-independent; the course also exports on a host and runs the result on the Pi. Check integrity on the Pi with `sha256sum models/*.pte` against `pte_sha256` in the `.json`.

## 7. Configuration checks
```bash
python scripts/check_config.py           # must end with RESULT: VALID
python -m unittest discover -s tests -t . -v
python scripts/check_parity.py --images data/dev     # FP32 gate on the Pi itself (needs torchvision)
```

## 8. Prepare images offline
```
data/organiser/   organiser images (jpg/png/bmp/webp) + labels.csv
data/dev/         separate development images + labels.csv (for the threshold sweep), never the test images
```
`labels.csv` format (see `data/organiser/labels.csv.template`):
```
image_filename,ground_truth_category
img01.jpg,<exact organiser category>
```
Before the final run:
1. Put the **organiser's exact category names** into `config/mapping.json → categories`, choose an action for each (e.g. `RECYCLE`), review every class rule, and set `"status": "ORGANISER"`.
2. Add a `voice.yaml` message (en + ta) and audio clips for any new action, then run `python scripts/generate_audio.py`.
3. Run `python scripts/check_config.py`.
4. Sweep and freeze the threshold on **dev** images only (section 9).

## 9. Run the classifier
```bash
# development run + threshold sweep (dev images only)
python run_zyqora.py --images data/dev --labels data/dev/labels.csv --out outputs/dev --allow-example-mapping
python scripts/sweep_threshold.py --results outputs/dev/results.csv
#   -> edit config/app.yaml: ecoguard.threshold, calibrated: true, frozen: true

# FINAL mandatory evaluation (organiser set, mapping status ORGANISER)
python run_zyqora.py --images data/organiser --labels data/organiser/labels.csv --out outputs/final
```
Outputs: `outputs/final/results.csv`, `summary.json`, `annotated/`.

## 10. Benchmark
```bash
python scripts/benchmark_pi.py --model models/mobilenetv2_fp32_xnnpack.pte --threads 1 2 4 --images data/organiser
python scripts/benchmark_pi.py --model models/mobilenetv2_int8_xnnpack.pte --threads 1 2 4 --images data/organiser
vcgencmd measure_temp; vcgencmd get_throttled
```
Results go to `outputs/benchmark/*.json`, which records `device_model` (must say "Raspberry Pi 5"). Put the chosen thread count into `config/app.yaml → model.num_threads`.

## 11. Voice demonstration
```bash
aplay -l                                            # list output devices
PYTHONPATH=src python -m zyqora.voicesort --lang ta --list
PYTHONPATH=src python -m zyqora.voicesort --lang ta --say RECYCLE
python run_zyqora.py --images data/demo --voice ta --allow-example-mapping --out outputs/tamil_demo
```
If there is no sound, set the default output device (`sudo raspi-config` → System → Audio) or create `~/.asoundrc`. VoiceSort prints the text instead of failing.

## 12. GreenScore
```bash
PYTHONPATH=src python -m zyqora.greenscore add-user --user-id U01 --name "Team A"
python run_zyqora.py --images data/demo --greenscore-user U01 --allow-example-mapping --out outputs/gs_demo
PYTHONPATH=src python -m zyqora.greenscore events --status PENDING
PYTHONPATH=src python -m zyqora.greenscore verify --event-id 1 --voice ta     # interactive supervisor check
PYTHONPATH=src python -m zyqora.greenscore points --user-id U01
```
The database is `outputs/greenscore.sqlite3` (local file, not committed).

## 13. Tests
```bash
python -m unittest discover -s tests -t . -v
```
On the Pi with the `.pte` present, the real-runtime integration test also runs; it is skipped elsewhere.

## 14. Disable networking for the demo
```bash
sudo nmcli radio wifi off               # Wi-Fi off (NetworkManager, default on Trixie)
sudo ip link set eth0 down              # if Ethernet is plugged in
nmcli radio wifi; ip -brief addr        # show it is off
python scripts/check_offline.py         # all assets local + "network: not reachable"
# after the demo
sudo nmcli radio wifi on; sudo ip link set eth0 up
```
Offline checklist:
- [ ] `check_offline.py` prints `RESULT: READY | OFFLINE`
- [ ] `run_zyqora.py` completes with Wi-Fi off
- [ ] VoiceSort clip plays with Wi-Fi off
- [ ] GreenScore verify works with Wi-Fi off
- [ ] Nothing in the run prints a download (weights are not fetched at runtime: the `.pte` and labels are local)

## 15. Troubleshooting
| Symptom | Fix |
|---|---|
| `Could not import ExecuTorch pybindings` | Not Python 3.12 or wrong wheels. Recreate the venv with `pyenv shell 3.12.14`. |
| `Required ExecuTorch backend 'XnnpackBackend' is not registered` | Reinstall `executorch==1.3.1` in the venv. |
| `Model file not found` | Run section 6 or copy the `.pte` to `models/`. |
| `ExecuTorch failed to load ... incompatible` | The `.pte` was exported with another ExecuTorch version. Re-export with the pinned toolchain. |
| `expects input (...)` | Wrong model for this contract; use the MobileNetV2 export. |
| `mapping.json has status EXAMPLE` | Configure the organiser categories (section 8) or add `--allow-example-mapping` for dev runs. |
| `unknown category names` in labels | The label spelling must match a category name or alias in `mapping.json`. |
| `image(s) have no ground-truth label` | Every image needs a row in `labels.csv` and vice versa. |
| No sound | `aplay -l`; set the output device in `raspi-config`; check `audio/<lang>/*.wav` exist. |
| `flatc` not found (export) | Call the venv's python, or activate the venv; the scripts add `venv/bin` to `PATH`. |
| Slow or variable latency | Check `vcgencmd get_throttled` (cooling); close other programs; use Lite OS. |
