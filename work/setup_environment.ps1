param([switch]$CheckOnly, [switch]$WithReports)
$ErrorActionPreference = 'Stop'
$packageWork = Split-Path -Parent $MyInvocation.MyCommand.Path
$packageRoot = Split-Path -Parent $packageWork
$packageVenv = Join-Path $packageWork '.venv_madqn_ablation'
$packagePython = Join-Path $packageVenv 'Scripts/python.exe'

if (-not (Test-Path -LiteralPath $packagePython)) {
    if ($CheckOnly) { throw 'The project environment has not been installed. Run setup_windows.cmd first.' }
    $packageCandidates = @(
        @{ Name = 'py'; Prefix = @('-3.10') },
        @{ Name = 'py'; Prefix = @('-3.11') },
        @{ Name = 'py'; Prefix = @('-3.12') },
        @{ Name = 'python'; Prefix = @() }
    )
    $packageSelected = $null
    foreach ($packageCandidate in $packageCandidates) {
        $packageCommand = Get-Command $packageCandidate.Name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $packageCommand) { continue }
        $packagePrefix = $packageCandidate.Prefix
        try {
            & $packageCommand.Source @packagePrefix -c 'import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)' 2>$null
            if ($LASTEXITCODE -eq 0) {
                $packageSelected = @{ Exe = $packageCommand.Source; Prefix = $packagePrefix }
                break
            }
        } catch {}
    }
    if (-not $packageSelected) { throw 'Install Python 3.10, 3.11 or 3.12 and enable its PATH option, then run setup_windows.cmd again. Python 3.10 matches the original experiment environment.' }
    $packagePrefix = $packageSelected.Prefix
    & $packageSelected.Exe @packagePrefix -m venv $packageVenv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the project Python environment.' }
}

if (-not $CheckOnly) {
    $packageRequirements = Join-Path $packageRoot 'requirements.txt'
    if ($WithReports) { $packageRequirements = Join-Path $packageRoot 'requirements-reports.txt' }
    & $packagePython -m pip install -r $packageRequirements
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check Internet access and the pip output above.' }
}

& $packagePython -c 'import sys,numpy,scipy,simpy,gymnasium,pettingzoo,torch; print("Python:",sys.version.split()[0]); print("SimPy:",simpy.__version__); print("PyTorch:",torch.__version__); print("Core dependencies are ready.")'
if ($LASTEXITCODE -ne 0) { throw 'The dependency check failed.' }
