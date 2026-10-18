@echo off
set "VENV_PY=%USERPROFILE%\.cache\ctts-venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
  echo venv not found: %VENV_PY%
  echo Run setup_venv.cmd first.
  exit /b 1
)
"%VENV_PY%" "%~dp0ctts.py" --gui
exit /b %ERRORLEVEL%
