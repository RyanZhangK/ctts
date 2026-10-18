# 兼容旧入口：实际逻辑在跨平台的 setup_env.py 中。
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_venv.ps1
$ErrorActionPreference = "Stop"
if (Get-Command py -ErrorAction SilentlyContinue) {
  & py -3 "$PSScriptRoot\setup_env.py" @args
} else {
  & python "$PSScriptRoot\setup_env.py" @args
}
exit $LASTEXITCODE
