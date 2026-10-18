# Windows: create venv outside eDisk (no symlink support) and sync deps.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_venv.ps1
$ErrorActionPreference = "Stop"
$venv = Join-Path $env:USERPROFILE ".cache\ctts-venv"
# 系统 TEMP 若指向 D:\temo 等不可写目录，uv 解压 wheel 会报 os error 5
$tmp = Join-Path $env:USERPROFILE ".cache\ctts-tmp"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$env:TMP = $tmp
$env:TEMP = $tmp
$env:UV_CACHE_DIR = Join-Path $env:USERPROFILE ".cache\uv"

Write-Host "uv: $(uv --version)"
Write-Host "TEMP redirected to: $tmp"
try { Write-Host "python: $(python --version 2>&1)" } catch { Write-Host "python: (not on PATH)" }

if (-not (Test-Path $venv)) {
  Write-Host "Creating venv at $venv ..."
  uv venv $venv
} else {
  Write-Host "Venv already exists: $venv"
}

Set-Location $PSScriptRoot
# 不调用 Activate.ps1，避免 Restricted 执行策略再次拦截
$env:VIRTUAL_ENV = $venv
$env:PATH = "$(Join-Path $venv 'Scripts');$env:PATH"
uv sync --active
if ($LASTEXITCODE -ne 0) { throw "uv sync --active failed with exit $LASTEXITCODE" }

$py = Join-Path $venv "Scripts\python.exe"
Write-Host "venv python: $(& $py --version)"
& $py -c "import cv2, numpy, httpx, label_studio_sdk, PySide6, ultralytics; print('imports OK')"
Write-Host "Done. Activate with:"
Write-Host "  & `"$venv\Scripts\Activate.ps1`""
