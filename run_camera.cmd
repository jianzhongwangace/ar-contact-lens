@echo off
setlocal
cd /d "%~dp0"
set "ARLENS_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "runtime-python.txt" set /p ARLENS_PYTHON=<"runtime-python.txt"
if not exist "%ARLENS_PYTHON%" (
  echo Missing environment. Please read README_USB_CAMERA.md.
  pause
  exit /b 1
)
"%ARLENS_PYTHON%" camera_app.py %*
if errorlevel 1 pause
