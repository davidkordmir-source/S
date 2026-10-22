param([switch]$SkipSetup)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
if (-not $SkipSetup) { & "$PSScriptRoot/setup-windows.ps1" }
if (-not (Test-Path 'ui/dist/index.html')) { throw 'Build the UI before packaging.' }
& '.venv/Scripts/python.exe' -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed; packaging stopped.' }
& '.venv/Scripts/python.exe' -m PyInstaller --noconfirm --clean AtlasScout.spec
if ($LASTEXITCODE -ne 0) { throw 'Windows packaging failed.' }
if (-not (Test-Path 'dist/AtlasScout/AtlasScout.exe')) { throw 'Expected executable was not produced.' }
Compress-Archive -Path 'dist/AtlasScout' -DestinationPath 'dist/AtlasScout-Windows-x64.zip' -Force
Write-Host 'Created dist/AtlasScout-Windows-x64.zip. Keep all files beside AtlasScout.exe.' -ForegroundColor Green
