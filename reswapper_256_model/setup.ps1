<#
.SYNOPSIS
    One-shot environment setup for the AI Live Face Swap prototype (Windows).

.DESCRIPTION
    1. Verifies Python 3.11 is on PATH
    2. Creates a .venv virtual environment
    3. Upgrades pip
    4. Installs GPU or CPU requirements based on --Cpu flag
    5. Creates the models/ directory
    6. Prompts to run download_models.py

.EXAMPLE
    # GPU (CUDA) build — default
    .\setup.ps1

    # CPU-only build (no CUDA required, much slower)
    .\setup.ps1 -Cpu
#>
param(
    [switch]$Cpu
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ── 1. Python version check ──────────────────────────────────────────────────
Write-Host "`n[1/5] Checking Python version..." -ForegroundColor Cyan
$pyVersion = & python --version 2>&1
if ($pyVersion -notmatch '3\.11') {
    Write-Host "[warn] Python 3.11 not detected (found: $pyVersion)." -ForegroundColor Yellow
    Write-Host "       InsightFace and onnxruntime-gpu are tested against Python 3.11."
    Write-Host "       Download from: https://www.python.org/downloads/release/python-3110/"
    $continue = Read-Host "Continue anyway? [y/N]"
    if ($continue -ne 'y') { exit 1 }
} else {
    Write-Host "  OK — $pyVersion" -ForegroundColor Green
}

# ── 2. Virtual environment ────────────────────────────────────────────────────
Write-Host "`n[2/5] Creating virtual environment (.venv)..." -ForegroundColor Cyan
if (Test-Path ".venv") {
    Write-Host "  .venv already exists — skipping creation."
} else {
    python -m venv .venv
    Write-Host "  Created .venv" -ForegroundColor Green
}

$pip   = ".\.venv\Scripts\pip.exe"
$python = ".\.venv\Scripts\python.exe"

# ── 3. Upgrade pip ────────────────────────────────────────────────────────────
Write-Host "`n[3/5] Upgrading pip..." -ForegroundColor Cyan
& $pip install --quiet --upgrade pip

# ── 4. Install dependencies ───────────────────────────────────────────────────
if ($Cpu) {
    $reqFile = "requirements-cpu.txt"
    Write-Host "`n[4/5] Installing CPU-only requirements ($reqFile)..." -ForegroundColor Cyan
    & $pip install -r $reqFile
} else {
    $reqFile = "requirements.txt"
    Write-Host "`n[4/5] Installing GPU requirements ($reqFile)..." -ForegroundColor Cyan
    Write-Host "  Using Microsoft CUDA 12 pip index — standard PyPI ships a CUDA 11 build"
    Write-Host "  which will not work on systems with CUDA 12.x installed."
    # The extra-index-url provides onnxruntime-gpu compiled against CUDA 12.
    # Without it, pip pulls the CUDA 11 wheel from PyPI and the CUDA provider
    # fails at runtime with LoadLibrary error 126.
    $cudaIndex = "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/"
    & $pip install -r $reqFile --extra-index-url $cudaIndex
}

# ── 5. Models directory ───────────────────────────────────────────────────────
Write-Host "`n[5/5] Checking models/ directory..." -ForegroundColor Cyan
if (-not (Test-Path "models")) {
    New-Item -ItemType Directory -Path "models" | Out-Null
}
Write-Host "  models/ ready." -ForegroundColor Green

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host "`n Setup complete." -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Activate the environment : .\.venv\Scripts\Activate.ps1"
Write-Host "  2. Download swap model      : python download_models.py"
Write-Host "  3. Run the prototype        : python run.py --source <your_face.jpg>"
Write-Host "  4. Run the benchmark        : python benchmark.py --source <your_face.jpg> --duration 60"
Write-Host ""
Write-Host "For virtual camera output, install OBS Studio and activate OBS Virtual Camera,"
Write-Host "then add --virtual-cam to the run.py command."
Write-Host ""

$dl = Read-Host "Download models now? [y/N]"
if ($dl -eq 'y') {
    & $python download_models.py
}
