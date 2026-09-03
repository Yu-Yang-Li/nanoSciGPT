[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Destination
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$sourceRoot = Join-Path $repoRoot "skills"
$destinationRoot = [System.IO.Path]::GetFullPath($Destination)
$skillNames = @(
    "nanoscigpt-research-baseline-builder",
    "nanogpt-pretraining",
    "nanoscigpt-scientific-language",
    "autoresearch-model-iteration",
    "ai-scientist-v1-workflow",
    "ai-scientist-v2-tree-search"
)

foreach ($name in $skillNames) {
    $source = Join-Path $sourceRoot $name
    $target = Join-Path $destinationRoot $name
    if (-not (Test-Path -LiteralPath (Join-Path $source "SKILL.md") -PathType Leaf)) {
        throw "source skill is incomplete: $source"
    }
    if (Test-Path -LiteralPath $target) {
        throw "skill already exists: $target"
    }
}

New-Item -ItemType Directory -Force -Path $destinationRoot | Out-Null
foreach ($name in $skillNames) {
    Copy-Item -LiteralPath (Join-Path $sourceRoot $name) -Destination $destinationRoot -Recurse
}

Write-Output "installed 6 nanoSciGPT course skills -> $destinationRoot"
