param(
    [string[]]$RequiredModules = @("numpy", "pandas", "sklearn")
)

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$profileRoot = if ($env:USERPROFILE) {
    $env:USERPROFILE
} else {
    [Environment]::GetFolderPath("UserProfile")
}
$candidates = [System.Collections.Generic.List[string]]::new()

foreach ($name in @("python", "python3")) {
    $command = Get-Command $name -ErrorAction SilentlyContinue
    if ($command -and $command.Source) {
        $candidates.Add($command.Source)
    }
}

foreach ($path in @(
    (Join-Path $repoRoot ".venv\Scripts\python.exe"),
    (Join-Path $repoRoot "venv\Scripts\python.exe")
)) {
    if (Test-Path -LiteralPath $path) {
        $candidates.Add($path)
    }
}

foreach ($pattern in @(
    (Join-Path $profileRoot ".conda\envs\*\python.exe"),
    (Join-Path $profileRoot "AppData\Local\Programs\Python\Python*\python.exe")
)) {
    Get-ChildItem -Path $pattern -File -ErrorAction SilentlyContinue |
        ForEach-Object { $candidates.Add($_.FullName) }
}

$imports = $RequiredModules -join ", "
$seen = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)

foreach ($candidate in $candidates) {
    if (-not $seen.Add($candidate)) {
        continue
    }
    & $candidate -c "import $imports" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Output $candidate
        exit 0
    }
}

Write-Error "没有找到能够导入 $imports 的 Python。请先创建课程环境并安装相应依赖。"
exit 1
