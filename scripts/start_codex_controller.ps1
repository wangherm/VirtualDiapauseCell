param(
    [Parameter(Mandatory=$true)][string]$RemoteRun,
    [string]$SshTarget = 'autodl-vdc',
    [int]$Port = 22,
    [string]$IdentityFile = '',
    [string]$Workspace = '',
    [int]$Turns = 24,
    [string]$Model = '',
    [switch]$DoctorOnly
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Use the project Windows .venv (python -m venv .venv; .venv/Scripts/python -m pip install numpy), then rerun.'
}
if (-not $Workspace) { $Workspace = Join-Path $repo 'work/codex_controller' }
$Workspace = [System.IO.Path]::GetFullPath($Workspace)
New-Item -ItemType Directory -Force -Path $Workspace | Out-Null
$connection = @{
    transport = 'ssh'; ssh_target = $SshTarget; ssh_port = $Port
    python = '/root/autodl-tmp/VirtualDiapauseCell/.venv-autodl-vdc/bin/python'
    worker_script = "$RemoteRun/release/scripts/research_worker.py"; run = $RemoteRun
}
if ($IdentityFile) { $connection.identity_file = $IdentityFile }
$connectionPath = Join-Path $Workspace 'connection.local.json'
$serialized = $connection | ConvertTo-Json
if (Test-Path -LiteralPath $connectionPath) {
    $old = Get-Content -Raw -LiteralPath $connectionPath | ConvertFrom-Json
    if ($old.run -ne $RemoteRun -or $old.ssh_target -ne $SshTarget -or $old.ssh_port -ne $Port) {
        throw 'Workspace belongs to another connection/run. Choose a new -Workspace.'
    }
}
[System.IO.File]::WriteAllText($connectionPath, $serialized, [System.Text.UTF8Encoding]::new($false))
& $python "$repo/scripts/run_codex_research.py" doctor --connection $connectionPath
if ($LASTEXITCODE -ne 0) { throw 'SSH worker check failed; establish host trust and key login first.' }
if ($DoctorOnly) { exit 0 }
$cliArgs = @("$repo/scripts/run_codex_research.py", 'run', '--connection', $connectionPath,
             '--workspace', $Workspace, '--turns', "$Turns")
if ($Model) { $cliArgs += @('--model', $Model) }
& $python @cliArgs
exit $LASTEXITCODE
