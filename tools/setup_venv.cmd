@echo off
rem 兼容旧入口：实际逻辑在跨平台的 setup_env.py 中。
call "%~dp0setup_env.cmd" %*
exit /b %ERRORLEVEL%
