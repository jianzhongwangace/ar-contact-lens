@echo off
setlocal
cd /d "%~dp0"
set "ELLSEG_PYTHON=D:\ARLensRuntime\venv\Scripts\python.exe"
if exist "runtime-python.txt" set /p ELLSEG_PYTHON=<"runtime-python.txt"
if not exist "%ELLSEG_PYTHON%" (
  echo Missing runtime. Run setup.ps1 first.
  pause
  exit /b 1
)
"%ELLSEG_PYTHON%" ellseg_app.py %*
if errorlevel 1 pause
