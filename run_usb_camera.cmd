@echo off
setlocal
cd /d "%~dp0"
call run_camera.cmd --camera 0 --backend dshow --width 1920 --height 1080 %*
