param([switch]$SkipSetup)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
if (-not $SkipSetup) { & "$PSScriptRoot/setup-windows.ps1" }
if (-not (Test-Path 'ui/dist/index.html')) { throw 'Build the UI before packaging.' }
& '.venv/Scripts/python.exe' scripts/bundle-models.py
if ($LASTEXITCODE -ne 0) { throw 'Preparing verified offline model files failed.' }
& '.venv/Scripts/python.exe' -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed; packaging stopped.' }
& '.venv/Scripts/python.exe' -m PyInstaller --noconfirm --clean AtlasScout.spec
if ($LASTEXITCODE -ne 0) { throw 'Windows packaging failed.' }
if (-not (Test-Path 'dist/AtlasScout/AtlasScout.exe')) { throw 'Expected executable was not produced.' }
Compress-Archive -Path 'dist/AtlasScout' -DestinationPath 'dist/AtlasScout-Offline-Windows-x64.zip' -Force
Write-Host 'Created dist/AtlasScout-Offline-Windows-x64.zip. The models are included; keep the entire folder.' -ForegroundColor Green
