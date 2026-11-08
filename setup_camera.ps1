param([switch]$Cpu, [switch]$Cuda, [string]$RuntimeRoot = '')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $RuntimeRoot) {
    $RuntimeRoot = if (Test-Path -LiteralPath 'D:/') { 'D:/ARLensRuntime' } else { Join-Path $PSScriptRoot '.runtime' }
}
$cameraEnv = Join-Path $RuntimeRoot 'venv'
$cameraPython = Join-Path $cameraEnv 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $cameraPython)) {
    & python -m venv $cameraEnv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create Python environment' }
}
& $cameraPython -m pip install --cache-dir (Join-Path $RuntimeRoot 'pip-cache') -r requirements-camera.txt
if ($LASTEXITCODE -ne 0) { throw 'Camera dependencies failed to install' }
if ($Cuda) {
    & $cameraPython -m pip install --cache-dir (Join-Path $RuntimeRoot 'pip-cache') --timeout 120 'torch==2.5.1+cu118' 'torchvision==0.20.1+cu118' --index-url https://download.pytorch.org/whl/cu118
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed' }
} else {
    & $cameraPython -c 'import torch, torchvision'
    if ($LASTEXITCODE -ne 0) {
        & $cameraPython -m pip install --cache-dir (Join-Path $RuntimeRoot 'pip-cache') --timeout 120 'torch==2.5.1+cpu' 'torchvision==0.20.1+cpu' --index-url https://download.pytorch.org/whl/cpu
        if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed' }
    }
}
if (-not (Test-Path -LiteralPath 'models/lightiris-2x160x256.onnx')) {
    & $cameraPython export_onnx.py
    if ($LASTEXITCODE -ne 0) { throw 'ONNX export failed' }
}
if ($Cpu) {
    & $cameraPython -c 'from eye_pipeline import EyePipeline; p=EyePipeline(device="cpu"); print("Model ready:",p.predictor.device); p.close()'
} else {
    & $cameraPython -c 'from eye_pipeline import EyePipeline; p=EyePipeline(); print("Model ready:",p.predictor.device); p.close()'
}
if ($LASTEXITCODE -ne 0) { throw 'Model validation failed' }
Set-Content -LiteralPath 'runtime-python.txt' -Value $cameraPython -Encoding ASCII
Write-Output 'Setup complete. Start run_usb_camera.cmd or run_image.cmd. Use --device cpu for CPU-only execution.'
