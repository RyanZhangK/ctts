@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_venv.ps1"
exit /b %ERRORLEVEL%
