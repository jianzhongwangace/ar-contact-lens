@echo off
setlocal
cd /d "%~dp0"
set "ARLENS_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "runtime-python.txt" set /p ARLENS_PYTHON=<"runtime-python.txt"
"%ARLENS_PYTHON%" image_app.py %*
if errorlevel 1 pause
