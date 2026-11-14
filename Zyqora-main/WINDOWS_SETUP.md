# Running Zyqora AI 4.0 on Windows (VS Code)

> **Verification status: UNVERIFIED on Windows.** The scripts and commands below were written for Windows 10/11 x64, but no Windows machine was available to the developers. What *was* checked:
> - every Python package has a Windows wheel for Python 3.12 on PyPI (torch 2.12.1, executorch 1.3.1, onnxruntime, pytorch-tokenizers, …);
> - the ExecuTorch Windows wheel's `_portable_lib` contains the XNNPACK backend (static inspection of the binary);
> - the same Python code, tests and dashboard build pass on Linux;
> - `scripts/init_db.py` works with a path containing spaces.
>
> If a step fails, the scripts stop with the failing command named; please report it.

## Prerequisites (one time)

1. **Python 3.12, 64-bit** from python.org. In the installer, tick *"Add python.exe to PATH"*. The `py` launcher is installed by default.
2. **Node.js 20 LTS or newer** from nodejs.org (includes npm).
3. **Git for Windows**, and **VS Code** (optional).

## Get the code

In PowerShell:
```powershell
git clone https://github.com/revati-cse/Zyqora_Final.git
```
```powershell
cd Zyqora_Final\Zyqora-main
```
```powershell
git switch claude/compassionate-mccarthy-6ec5a6
```
The model files (`models/*.pte`, `models/*.onnx`) are in the repository, so nothing is downloaded at runtime.

## Set up (creates `.venv`, installs packages, builds the dashboard, initialises the database)

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```
`-ExecutionPolicy Bypass` applies to this one command only; it does not change system policy. The script checks that ExecuTorch reports `XnnpackBackend`, and stops if it does not.

## Start

```powershell
powershell -ExecutionPolicy Bypass -File .\start_all.ps1
```
This opens the backend in a second window, waits for `/api/health`, prints whether the classifier and detector are ready, and opens **http://127.0.0.1:8000** in the browser.

Other ways to start:

| Goal | Command |
|---|---|
| Backend only (serves the built dashboard) | `powershell -ExecutionPolicy Bypass -File .\start_backend.ps1` |
| Frontend dev server with hot reload (port 5173), against the local backend | `powershell -ExecutionPolicy Bypass -File .\start_all.ps1 -Dev` |
| Frontend dev server against the Raspberry Pi | `powershell -ExecutionPolicy Bypass -File .\start_frontend.ps1 -ApiTarget http://<pi-ip>:8000` |
| Single-item mode only (no YOLO) | `powershell -ExecutionPolicy Bypass -File .\start_backend.ps1 -NoDetector` |

## Manual commands (what the scripts do)

```powershell
py -3.12 -m venv .venv
```
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.txt
```
```powershell
.\.venv\Scripts\python.exe scripts\init_db.py
```
```powershell
cd frontend; npm ci; npm run build; cd ..
```
```powershell
.\.venv\Scripts\python.exe serve_zyqora.py --host 127.0.0.1 --port 8000
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -t .
```
```powershell
cd frontend; npm test; cd ..
```

## Notes

- **Camera.** The *Live Camera* page works at `http://127.0.0.1:8000` or `http://localhost:5173`, because browsers treat localhost as secure. The Pi camera options (`picamera2`) are Raspberry Pi only.
- **Speed.** A laptop CPU is usually faster than the Pi 5. Latency shown on Windows is **not** Pi performance; the dashboard labels every figure with the device it was measured on.
- **Power** readings (`vcgencmd`) only exist on the Pi; Windows shows "no meter connected".
- **If ExecuTorch fails to import:** use Python 3.10–3.13, 64-bit. The Microsoft Store Python and 32-bit Python are not supported by the wheels.
