param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPath = Join-Path $repoRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
$stampPath = Join-Path $venvPath ".surface-impedance-installed"
$shouldInstall = $false

Set-Location $repoRoot

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment in .venv..." -ForegroundColor Cyan
    py -m venv $venvPath
    $shouldInstall = $true
}

if (-not $SkipInstall -and -not (Test-Path $stampPath)) {
    $shouldInstall = $true
}

if ($shouldInstall) {
    Write-Host "Installing project into the virtual environment..." -ForegroundColor Cyan
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -e ".[plot]"
    Set-Content -Path $stampPath -Value "installed"
}

. $activateScript

Write-Host ""
Write-Host "SurfaceImpedance environment is ready." -ForegroundColor Green
Write-Host "Project folder: $repoRoot"
Write-Host "Python: $(Get-Command python | Select-Object -ExpandProperty Source)"
Write-Host ""
Write-Host "Example command:" -ForegroundColor Yellow
Write-Host "python -m surface_impedance.cli --list-models"
Write-Host ""

Start-Process powershell.exe -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-Command", "& { Set-Location '$repoRoot'; . '$activateScript' }"
) | Out-Null
