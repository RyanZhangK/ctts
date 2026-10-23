@echo off
rem 环境缺失时自动创建并安装依赖，然后启动 GUI。
set "PY=python"
where py >nul 2>nul && set "PY=py -3"
%PY% "%~dp0setup_env.py" --run %*
exit /b %ERRORLEVEL%
