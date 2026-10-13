param(
    [Parameter(Mandatory=$true)][string]$RemoteRun,
    [string]$SshTarget = 'autodl-vdc',
    [int]$Port = 22,
    [string]$IdentityFile = '',
    [string]$Workspace = '',
    [int]$Turns = 24,
    [string]$Model = '',
    [switch]$Continuous,
    [switch]$AcceptControllerUpdate,
    [string]$Brief = '',
    [int]$CheckpointExperiments = 5,
    [int]$CheckpointTurns = 24,
    [switch]$DoctorOnly
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Use the project Windows .venv (python -m venv .venv; .venv/Scripts/python -m pip install numpy), then rerun.'
}
if (-not $Workspace) { $Workspace = Join-Path $repo 'work/codex_controller' }
# PowerShell location can differ from .NET process cwd. Respect the shell location.
$Workspace = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Workspace)
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
if (-not (Get-Command codex -ErrorAction SilentlyContinue)) {
    $bundled = Get-ChildItem -Path "$env:LOCALAPPDATA/OpenAI/Codex/bin/*/codex.exe" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $bundled) { throw 'Codex CLI not found. Install or add its directory to PATH.' }
    $env:Path = "$($bundled.DirectoryName);$env:Path"
}
& codex login status
if ($LASTEXITCODE -ne 0) { throw 'Codex login check failed; no research started.' }
$cliArgs = @("$repo/scripts/run_codex_research.py", 'run', '--connection', $connectionPath,
             '--workspace', $Workspace, '--turns', "$Turns",
             '--checkpoint-experiments', "$CheckpointExperiments", '--checkpoint-turns', "$CheckpointTurns")
if ($Model) { $cliArgs += @('--model', $Model) }
if ($Continuous) { $cliArgs += '--continuous' }
if ($AcceptControllerUpdate) { $cliArgs += '--accept-controller-update' }
if ($Brief) { $cliArgs += @('--brief', $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Brief)) }
& $python @cliArgs
exit $LASTEXITCODE
