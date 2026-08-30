$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspace = (Resolve-Path (Join-Path $scriptDir '..\..\..')).Path

Push-Location $workspace
try {
    python (Join-Path $scriptDir 'run_occurrence_fallback.py')
    python (Join-Path $scriptDir 'build_unified_audit.py')
}
finally {
    Pop-Location
}
