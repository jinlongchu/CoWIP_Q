param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$dashboardWork = Split-Path -Parent $MyInvocation.MyCommand.Path
$dashboardRoot = Split-Path -Parent $dashboardWork
$dashboardPython = Join-Path $dashboardWork '.venv_madqn_ablation/Scripts/python.exe'
$dashboardServer = Join-Path $dashboardWork 'run_des_dashboard.py'
if (-not (Test-Path -LiteralPath $dashboardPython)) {
    throw 'The project Python environment is missing: work/.venv_madqn_ablation'
}
$dashboardLogDir = Join-Path $dashboardRoot 'outputs/des_dashboard'
New-Item -ItemType Directory -Path $dashboardLogDir -Force | Out-Null
$dashboardUrl = $null
foreach ($dashboardPort in 8767..8777) {
    $candidateUrl = "http://127.0.0.1:$dashboardPort"
    try {
        $dashboardExisting = Invoke-RestMethod -Uri "$candidateUrl/api/state" -TimeoutSec 2
        if ($dashboardExisting.application -eq 'conwip-des-dashboard') {
            $dashboardUrl = $candidateUrl
            break
        }
        continue
    } catch {
        if (Get-NetTCPConnection -LocalPort $dashboardPort -State Listen -ErrorAction SilentlyContinue) {
            continue
        }
    }
    $dashboardProcess = Start-Process -FilePath $dashboardPython -ArgumentList @('-u', ('"{0}"' -f $dashboardServer), '--port', [string]$dashboardPort) -WorkingDirectory $dashboardRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $dashboardLogDir "server-$dashboardPort.log") -RedirectStandardError (Join-Path $dashboardLogDir "server-$dashboardPort.error.log") -PassThru
    for ($dashboardAttempt = 0; $dashboardAttempt -lt 90; $dashboardAttempt++) {
        Start-Sleep -Milliseconds 500
        try {
            $dashboardReady = Invoke-RestMethod -Uri "$candidateUrl/api/state" -TimeoutSec 2
            if ($dashboardReady.application -eq 'conwip-des-dashboard') {
                $dashboardUrl = $candidateUrl
                break
            }
        } catch {}
        if ($dashboardProcess.HasExited) {
            throw "The simulator could not start. See outputs/des_dashboard/server-$dashboardPort.error.log"
        }
    }
    if ($dashboardUrl) { break }
    throw 'The simulator did not become ready within the startup timeout.'
}
if (-not $dashboardUrl) { throw 'No available port between 8767 and 8777.' }
Write-Output "DES dashboard: $dashboardUrl"
if (-not $NoBrowser) { Start-Process $dashboardUrl }
