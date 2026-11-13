# Hardware integration

> Nothing in this file has been tested on physical hardware by the Zyqora authors. Camera and GPIO code paths are unit-tested with fakes only. The station runs fully in **simulation** (folder camera + simulated sorter) without any hardware.

## Station
```bash
python zyqora_station.py --camera folder:data/demo --voice ta --user DEMO01        # simulation, no hardware
python zyqora_station.py --camera auto --voice en --user DEMO01                     # Pi camera, simulated sorter
python zyqora_station.py --camera auto --enable-actuator --user DEMO01              # ONLY after the wiring checks below
```
Press Enter to capture one item and `q` to quit. Each item goes through: capture → classify → explanation card → voice → sorter → GreenScore PENDING event (`--verify-now` asks the supervisor right away). The station loads the model once, does no continuous inference, and logs to `outputs/station/<run>/station.log`, with one JSON line per item in `events.jsonl`.

Safety rules in code:
- **REVIEW items never actuate a bin** (`zyqora.sorter.decide_sort`); unit-tested.
- Camera failures are reported, nothing is classified, and the station stops after 3 consecutive failures.
- The GPIO backend needs `actuator.backend: gpio` in `config/hardware.yaml` **and** `--enable-actuator`. Without the flag it falls back to simulation and logs a warning.
- Every run prints whether the camera and sorter are `SIMULATION` or hardware, and `summary.json` records `camera_simulated` / `sorter_hardware`.

## Camera (`config/hardware.yaml → camera`)
| Backend | Notes |
|---|---|
| `picamera2` | Fastest: the camera stays running and frames stay in memory. Needs `sudo apt install python3-picamera2`, and the venv must see system packages (`python -m venv --system-site-packages .venv`). |
| `rpicam` | Runs `rpicam-still` per capture (shipped with Raspberry Pi OS). No Python dependency, but the camera restarts each time, which is slower. |
| `auto` | picamera2, then rpicam |
| `folder:<dir>` | Simulation from saved images |

Check the camera before using the station:
```bash
rpicam-hello --list-cameras
```
```bash
rpicam-still -n -t 300 --width 640 --height 480 -o /tmp/test.jpg
```
640×480 is enough, because the model resizes to 224×224; larger captures cost decode time only. A USB webcam is not supported (it would need OpenCV); save its frames to a folder and use `folder:` instead.

## Actuator (`config/hardware.yaml → actuator`)
`gpio` pulses one output per bin action for `pulse_s` seconds, using `gpiozero.OutputDevice` with BCM pin numbers. Typical wiring is a GPIO pin driving a relay module, or the signal input of a servo/solenoid driver board. **Never drive a motor or solenoid directly from a GPIO pin.** The example pins (COMPOST 17, RECYCLE 27) are placeholders; set them to match your wiring.

Before `--enable-actuator`:
1. Check that each pin drives the correct bin, using a multimeter or an LED first, not the mechanism.
2. Make sure the mechanism has its own power supply with a common ground, and a manual stop.
3. Test with `--camera folder:<dir> --auto 1` and a single known item while someone watches the bin.

## Audio (VoiceSort)
```bash
aplay -l
```
```bash
PYTHONPATH=src python -m zyqora.voicesort --lang ta --say RECYCLE
```
Choose the output with `sudo raspi-config` → System → Audio. Playback failures never stop the station; the text is printed instead.

## Dashboard hardware checks (Zyqora AI 3.0), none done yet
- [ ] `bash scripts/start_pi_demo.sh` on the Pi; laptop opens `http://<pi-ip>:8000` and shows PI CONNECTED + MODEL READY.
- [ ] Live Classification on a real organic and recyclable item photo: the result footer says Raspberry Pi 5 and shows ExecuTorch + XNNPACK.
- [ ] **Capture with device camera** returns a classification (Pi camera via `/api/camera/capture`); unplugging the camera gives `CAMERA_UNAVAILABLE`, not a fake result.
- [ ] Voice: "Play on the device speaker" plays through the Pi's audio output; "Play in this browser" plays on the laptop.
- [ ] Device Monitoring shows the Pi's CPU temperature and RAM, and the CPU % rises briefly during a batch.
- [ ] Batch of 50 images completes without `SERVER_BUSY`; record images/s on Analytics.
- [ ] With `--enable-actuator` and the gpio backend (HARDWARE.md checks first): ACCEPT pulses the right pin, REVIEW pulses nothing.
