<#
.SYNOPSIS
  Start the dashboard in development mode (Vite, hot reload) on http://localhost:5173.
  API calls under /api are proxied to the backend (default http://127.0.0.1:8000).
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\start_frontend.ps1
  powershell -ExecutionPolicy Bypass -File .\start_frontend.ps1 -ApiTarget http://192.168.1.50:8000   # backend on the Pi
#>
param([string]$ApiTarget = "http://127.0.0.1:8000")
$ErrorActionPreference = "Stop"
$Frontend = Join-Path $PSScriptRoot "frontend"
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { throw "npm not found. Install Node.js 20 LTS or newer." }
Push-Location -LiteralPath $Frontend
try {
  if (-not (Test-Path -LiteralPath (Join-Path $Frontend "node_modules"))) {
    npm ci
    if ($LASTEXITCODE -ne 0) { throw "npm ci failed" }
  }
  $env:ZYQORA_API_TARGET = $ApiTarget
  Write-Host "Dashboard (dev): http://localhost:5173   API: $ApiTarget" -ForegroundColor Green
  npm run dev
} finally { Pop-Location }
