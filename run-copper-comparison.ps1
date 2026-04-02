param(
    [double]$FMin = 1e9,
    [double]$FMax = 100e9,
    [int]$Points = 300,
    [double]$ProfileFrequency = 10e9
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtual environment not found at .venv. Start the project with 'Launch SurfaceImpedance.cmd' first."
}

& $python -m surface_impedance.cli `
    --f-min $FMin `
    --f-max $FMax `
    --points $Points `
    --case "label=half-space-copper,model=half-space,sigma=5.8e7,tau=0,epsr_real=1.0,epsr_imag=0.0,mur_real=1.0,mur_imag=0.0" `
    --case "label=rough-copper-1um,model=rough-single,sigma_metal=5.8e7,rq=1e-6,mu_r=1.0" `
    --export "results\halfspace-vs-rough-1um-1GHz-100GHz.csv" `
    --plot "results\halfspace-vs-rough-1um-1GHz-100GHz.png" `
    --profile-frequency $ProfileFrequency `
    --profile-plot "results\halfspace-vs-rough-1um-profile-10GHz.png" `
    --show-plot
