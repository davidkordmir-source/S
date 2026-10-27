param([switch]$SkipAI)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}

if (-not (Test-Path '.venv/Scripts/python.exe')) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        Invoke-Checked 'py' @('-3.12', '-m', 'venv', '.venv')
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        Invoke-Checked 'python' @('-m', 'venv', '.venv')
    } else { throw 'Install Python 3.12 x64 from python.org first.' }
}
$PythonExecutable = Join-Path $ProjectRoot '.venv/Scripts/python.exe'
Invoke-Checked $PythonExecutable @('-c', "import sys,struct; assert sys.version_info[:2] == (3,12) and struct.calcsize('P')==8, 'Python 3.12 x64 is required'")
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) { throw 'Source builds require Rust 1.99.1 (rustup.rs); the portable EXE does not need Rust.' }
if ($SkipAI) {
    Invoke-Checked $PythonExecutable @('-m', 'pip', 'install', './vendor/ecw2tiff')
    Invoke-Checked $PythonExecutable @('-m', 'pip', 'install', '-e', '.[dev]')
} else {
    Invoke-Checked $PythonExecutable @('-m', 'pip', 'install', '-r', 'requirements-lock.txt')
    Invoke-Checked $PythonExecutable @('-m', 'pip', 'install', '--no-deps', '-e', '.')
}
if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) { throw 'Install Node.js 22 or newer, then run this script again.' }
Invoke-Checked 'node' @('-e', "const [major,minor]=process.versions.node.split('.').map(Number); if(major<22 || (major===22 && minor<12)) process.exit(1)")
Push-Location ui
try {
    Invoke-Checked 'npm.cmd' @('ci', '--no-fund', '--no-audit')
    Invoke-Checked 'npm.cmd' @('run', 'build')
} finally { Pop-Location }
Write-Host 'Setup finished. Start with .venv\Scripts\python.exe -m atlas' -ForegroundColor Green
