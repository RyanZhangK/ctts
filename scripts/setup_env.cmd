@echo off
rem CTTS 跨平台环境初始化（Windows 入口）
set "PY=python"
where py >nul 2>nul && set "PY=py -3"
%PY% "%~dp0setup_env.py" %*
exit /b %ERRORLEVEL%
