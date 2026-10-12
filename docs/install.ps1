# EIS Studio installer for Windows (PowerShell).
#
#   powershell -ExecutionPolicy ByPass -c "irm https://zacharyslate.github.io/eis-studio/install.ps1 | iex"
#
# Installs uv (Astral's Python package manager) if needed, then EIS Studio with a private
# Python 3.12 in an isolated environment. No administrator rights needed. Creates Desktop and
# Start Menu shortcuts. Uninstall:  uv tool uninstall eis-studio   (and delete the shortcuts)

$ErrorActionPreference = "Stop"
$Source = if ($env:EIS_STUDIO_SOURCE) { $env:EIS_STUDIO_SOURCE } else { "eis-studio @ https://github.com/zacharyslate/eis-studio/archive/refs/heads/main.zip" }
$PyVer  = if ($env:EIS_STUDIO_PYTHON) { $env:EIS_STUDIO_PYTHON } else { "3.12" }

Write-Host "`nInstalling EIS Studio" -ForegroundColor Cyan

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    $uvHome = Join-Path $env:USERPROFILE ".local\bin"
    if (-not (Test-Path (Join-Path $uvHome "uv.exe"))) {
        Write-Host "Installing uv (Python package manager, https://docs.astral.sh/uv/)" -ForegroundColor Cyan
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    }
    $env:Path = "$uvHome;$env:Path"
}

Write-Host "Downloading EIS Studio and its scientific libraries (first install: a few minutes)" -ForegroundColor Cyan
$noBuild = @("cvxopt","numpy","scipy","scikit-learn","matplotlib","pandas","pyarrow","contourpy","kiwisolver") |
    ForEach-Object { "--no-build-package"; $_ }
uv tool install --force --refresh-package eis-studio --python $PyVer @noBuild $Source
if ($LASTEXITCODE -ne 0) { throw "uv tool install failed (exit code $LASTEXITCODE)." }
uv tool update-shell | Out-Null

$bin = Join-Path (uv tool dir --bin) "eis-studio.exe"
$shell = New-Object -ComObject WScript.Shell
$targets = @([Environment]::GetFolderPath("Desktop"),
             (Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"))
foreach ($dir in $targets) {
    if (-not (Test-Path $dir)) { continue }
    $lnk = $shell.CreateShortcut((Join-Path $dir "EIS Studio.lnk"))
    $lnk.TargetPath = $bin
    $lnk.WorkingDirectory = $env:USERPROFILE
    $lnk.Description = "EIS Studio: impedance analysis (ECM and DRT)"
    $lnk.Save()
}

Write-Host "`nDone." -ForegroundColor Green
Write-Host "Start EIS Studio from the 'EIS Studio' shortcut on your Desktop or in the Start Menu,"
Write-Host "or type  eis-studio  in a new terminal window."
Write-Host "It opens in your web browser and runs only on this computer; your data is not uploaded anywhere."
