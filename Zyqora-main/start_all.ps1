<#
.SYNOPSIS
  Start Zyqora AI 4.0 on Windows and open the dashboard.
  Default: one process - the API serves the built dashboard at http://127.0.0.1:8000.
  -Dev:    backend in a second window + Vite dev server (hot reload) at http://localhost:5173.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\start_all.ps1
  powershell -ExecutionPolicy Bypass -File .\start_all.ps1 -Dev
#>
param([int]$Port = 8000, [switch]$Dev, [switch]$NoBrowser)
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Backend = Join-Path $Root "start_backend.ps1"
$shell = if (Get-Command pwsh -ErrorAction SilentlyContinue) { "pwsh" } else { "powershell" }
$backendArgs = @("-NoExit", "-ExecutionPolicy", "Bypass", "-File", "`"$Backend`"", "-Port", "$Port")
if ($Dev) { $backendArgs += "-Dev" }
Start-Process -FilePath $shell -ArgumentList $backendArgs -WorkingDirectory $Root | Out-Null

Write-Host "Waiting for the API (model loading can take ~10-30 s)..."
$health = "http://127.0.0.1:$Port/api/health"
$ok = $false
for ($i = 0; $i -lt 120; $i++) {
  try { Invoke-RestMethod -Uri $health -TimeoutSec 2 | Out-Null; $ok = $true; break } catch { Start-Sleep -Seconds 1 }
}
if (-not $ok) { throw "The API did not answer at $health. Check the backend window for the error." }
$status = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/models/status"
Write-Host ("Classifier ready: {0}   Detector ready: {1}" -f $status.classifier.ready, $status.detector.ready)

if ($Dev) {
  $Frontend = Join-Path $Root "start_frontend.ps1"
  Start-Process -FilePath $shell -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-File", "`"$Frontend`"", "-ApiTarget", "http://127.0.0.1:$Port") -WorkingDirectory $Root | Out-Null
  $url = "http://localhost:5173"
  Start-Sleep -Seconds 4
} else {
  $url = "http://127.0.0.1:$Port"
}
Write-Host "Dashboard: $url" -ForegroundColor Green
if (-not $NoBrowser) { Start-Process $url }
