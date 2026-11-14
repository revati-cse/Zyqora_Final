<#
.SYNOPSIS
  One-time setup of Zyqora AI 4.0 on Windows 10/11 (x64): Python venv, packages, dashboard build, database.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
.NOTES
  Needs Python 3.12 (64-bit, from python.org; the "py" launcher is used when present) and Node.js 20+.
  Works from folders whose path contains spaces. Re-running it is safe.
#>
param(
  [string]$PythonVersion = "3.12",
  [switch]$SkipFrontend
)
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location -LiteralPath $Root

function Invoke-Checked([string]$What, [scriptblock]$Cmd) {
  Write-Host "==> $What" -ForegroundColor Cyan
  & $Cmd
  if ($LASTEXITCODE -ne 0) { throw "$What failed (exit code $LASTEXITCODE)" }
}

$Venv = Join-Path $Root ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
  if (Get-Command py -ErrorAction SilentlyContinue) {
    Invoke-Checked "Create virtual environment (Python $PythonVersion)" { py "-$PythonVersion" -m venv "$Venv" }
  } elseif (Get-Command python -ErrorAction SilentlyContinue) {
    Invoke-Checked "Create virtual environment" { python -m venv "$Venv" }
  } else {
    throw "Python not found. Install Python $PythonVersion (64-bit) from python.org and tick 'Add python.exe to PATH'."
  }
}
$ver = & $Python -c "import sys, struct; print('%d.%d %d-bit' % (sys.version_info[0], sys.version_info[1], struct.calcsize('P') * 8))"
Write-Host "Python in venv: $ver"
if ($ver -notmatch "^3\.(10|11|12|13) 64-bit$") { throw "ExecuTorch needs 64-bit Python 3.10-3.13; the venv has $ver. Delete .venv and install Python $PythonVersion (64-bit)." }

Invoke-Checked "Upgrade pip" { & $Python -m pip install --upgrade pip }
Invoke-Checked "Install Python packages (requirements-windows.txt)" { & $Python -m pip install -r (Join-Path $Root "requirements-windows.txt") }
Invoke-Checked "Check ExecuTorch + XNNPACK" {
  & $Python -c "from executorch.extension.pybindings import portable_lib as p; b=list(p._get_registered_backend_names()); print('ExecuTorch backends:', b); raise SystemExit(0 if 'XnnpackBackend' in b else 3)"
}
Invoke-Checked "Initialise the database" { & $Python (Join-Path $Root "scripts\init_db.py") }

if (-not $SkipFrontend) {
  if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { throw "npm not found. Install Node.js 20 LTS or newer from nodejs.org, then re-run." }
  Push-Location -LiteralPath (Join-Path $Root "frontend")
  try {
    Invoke-Checked "Install dashboard packages (npm ci)" { npm ci }
    Invoke-Checked "Build the dashboard (npm run build)" { npm run build }
  } finally { Pop-Location }
}
Write-Host ""
Write-Host "Setup complete. Start everything with:  powershell -ExecutionPolicy Bypass -File .\start_all.ps1" -ForegroundColor Green
