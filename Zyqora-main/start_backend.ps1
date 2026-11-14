<#
.SYNOPSIS
  Start the Zyqora AI API (and the built dashboard) on this Windows PC.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\start_backend.ps1                 # http://127.0.0.1:8000
  powershell -ExecutionPolicy Bypass -File .\start_backend.ps1 -Dev            # also allow the Vite dev server (port 5173)
  powershell -ExecutionPolicy Bypass -File .\start_backend.ps1 -NoDetector     # single-item classification only
#>
param(
  [int]$Port = 8000,
  [string]$BindHost = "127.0.0.1",
  [int]$Threads = 0,
  [switch]$Dev,
  [switch]$NoDetector
)
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location -LiteralPath $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { throw "No .venv found. Run setup_windows.ps1 first." }

$argsList = @((Join-Path $Root "serve_zyqora.py"), "--host", $BindHost, "--port", "$Port")
if ($Threads -gt 0) { $argsList += @("--threads", "$Threads") }
if ($Dev) { $argsList += @("--cors-origin", "http://localhost:5173", "--cors-origin", "http://127.0.0.1:5173") }
if ($NoDetector) { $argsList += "--no-detector" }
if (-not (Test-Path -LiteralPath (Join-Path $Root "frontend\dist\index.html"))) {
  Write-Warning "frontend\dist is missing: API only. Build it with setup_windows.ps1 or start_frontend.ps1 (dev server)."
}
Write-Host "Starting Zyqora AI API on http://${BindHost}:$Port  (Ctrl+C to stop)" -ForegroundColor Green
& $Python @argsList
exit $LASTEXITCODE
