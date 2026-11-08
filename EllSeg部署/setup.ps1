param([string]$RuntimeRoot='D:/EllSegRuntime')
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$ellsegPython=Join-Path $RuntimeRoot 'venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $ellsegPython)) {
    python -m venv (Join-Path $RuntimeRoot 'venv')
    if ($LASTEXITCODE -ne 0) { throw 'Cannot create EllSeg environment' }
}
& $ellsegPython -m pip install --cache-dir (Join-Path $RuntimeRoot 'pip-cache') -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& $ellsegPython -m pip install --cache-dir (Join-Path $RuntimeRoot 'pip-cache') --timeout 120 'torch==2.5.1+cpu' --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
& $ellsegPython -c 'from ellseg_core import EllSegPredictor; p=EllSegPredictor("cpu"); print("Strict weight loading OK")'
if ($LASTEXITCODE -ne 0) { throw 'Checkpoint validation failed' }
Set-Content -LiteralPath runtime-python.txt -Value $ellsegPython -Encoding ASCII
Write-Output 'Ready. Double click run.cmd.'
