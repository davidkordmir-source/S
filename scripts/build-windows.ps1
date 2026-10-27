param([switch]$SkipSetup)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
if (-not $SkipSetup) { & "$PSScriptRoot/setup-windows.ps1" }
& cargo test --manifest-path vendor/ecw2tiff/Cargo.toml --no-default-features --locked
if ($LASTEXITCODE -ne 0) { throw 'Vendored ECW codec regression tests failed.' }
if (-not (Test-Path 'ui/dist/index.html')) { throw 'Build the UI before packaging.' }
& '.venv/Scripts/python.exe' scripts/bundle-raster.py
if ($LASTEXITCODE -ne 0) { throw 'Preparing the verified native ECW/raster decoder failed.' }
& '.venv/Scripts/python.exe' -c "import logging; from pathlib import Path; from atlas.validation import validate_packaged_formats; logging.basicConfig(level=logging.INFO); validate_packaged_formats(Path('.raster-cache/validation'))"
if ($LASTEXITCODE -ne 0) { throw 'Actual input image format validation failed before packaging.' }
& '.venv/Scripts/python.exe' scripts/bundle-models.py
if ($LASTEXITCODE -ne 0) { throw 'Preparing verified offline model files failed.' }
& '.venv/Scripts/python.exe' scripts/bundle-enhanced-models.py
if ($LASTEXITCODE -ne 0) { throw 'Preparing aerial, segmentation and accelerated model files failed.' }
& '.venv/Scripts/python.exe' -m pytest -q --junitxml=pytest-results.xml 2>&1 | Tee-Object -FilePath pytest.log
if ($LASTEXITCODE -ne 0) {
    if (Test-Path 'pytest-results.xml') {
        [xml]$TestReport = Get-Content 'pytest-results.xml' -Raw
        foreach ($Failure in $TestReport.SelectNodes('//failure | //error')) {
            $Details = $Failure.InnerText.Replace('%', '%25').Replace("`r", '%0D').Replace("`n", '%0A')
            Write-Host "::error title=$($Failure.ParentNode.name)::$Details"
        }
    }
    throw 'Tests failed; packaging stopped.'
}
& '.venv/Scripts/python.exe' -m PyInstaller --noconfirm --clean AtlasScout.spec
if ($LASTEXITCODE -ne 0) { throw 'Windows packaging failed.' }
if (-not (Test-Path 'dist/AtlasScout/AtlasScout.exe')) { throw 'Expected executable was not produced.' }
Compress-Archive -Path 'dist/AtlasScout' -DestinationPath 'dist/AtlasScout-Offline-Windows-x64.zip' -Force
Write-Host 'Created dist/AtlasScout-Offline-Windows-x64.zip. The models are included; keep the entire folder.' -ForegroundColor Green
