#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# UNSLOTH_ROCM_TORCH picks the Windows multi-arch torch trio (unslothai/unsloth#9932). Get-RocmWindowsTorchSet is cut
# out of install.ps1 and studio/setup.ps1, checked to be the same text, and run: the default stays rocm7.14.0, rocm10.1
# installs the exact 10.1 trio from whl-next, an installed 10.1 trio is kept with the variable unset, and "default"
# switches back. ASCII only: Windows PowerShell 5.1 reads a BOM-less script in the ANSI code page.
# Run: pwsh -NoProfile -File tests/studio/test_rocm_windows_torch_set.ps1

$ErrorActionPreference = "Stop"
$root = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path

function Cut-Function([string]$Path) {
    $text = Get-Content -LiteralPath $Path -Raw -Encoding UTF8
    $start = $text.IndexOf("function Get-RocmWindowsTorchSet")
    if ($start -lt 0) { throw "$Path no longer defines Get-RocmWindowsTorchSet" }
    $end = $text.IndexOf("return `$default", $text.IndexOf("foreach (`$s in `$optIn) { if (`$installedTag", $start))
    $end = $text.IndexOf("}", $end) + 1
    # Normalise indentation so the install.ps1 (nested) and setup.ps1 (top-level) copies compare.
    return ((($text.Substring($start, $end - $start)) -split "`r?`n") | ForEach-Object { $_.TrimStart() }) -join "`n"
}
$fromInstall = Cut-Function (Join-Path $root "install.ps1")
$fromSetup = Cut-Function (Join-Path (Join-Path $root "studio") "setup.ps1")

$failures = 0
function Check([string]$Name, [bool]$Ok) {
    if ($Ok) { Write-Host "  PASS  $Name" } else { Write-Host "  FAIL  $Name" -ForegroundColor Red; $script:failures++ }
}

Check "install.ps1 and setup.ps1 carry the same Get-RocmWindowsTorchSet" ($fromInstall -eq $fromSetup)
Invoke-Expression $fromSetup

$base = "https://repo.amd.com/rocm/whl-multi-arch"
function Trio($s) { "$($s.IndexBase) $($s.Torch)+$($s.Tag) $($s.Vision)+$($s.Tag) $($s.Audio)+$($s.Tag)" }
$default = "$base 2.11.0+rocm7.14.0 0.26.0+rocm7.14.0 2.11.0+rocm7.14.0"
$rocm101 = "https://stable.repo.amd.com/rocm/whl-next 2.14.0+rocm10.1.0 0.29.0a0+rocm10.1.0 2.11.0.3+rocm10.1.0"

$s = Get-RocmWindowsTorchSet -Requested "" -InstalledTorchVersion "" -DefaultIndexBase $base -MultiArchMirror ""
Check "unset on a fresh venv is the default trio" ((Trio $s) -eq $default -and -not $s.OptIn)
$s = Get-RocmWindowsTorchSet -Requested "" -InstalledTorchVersion "2.11.0+rocm7.14.0" -DefaultIndexBase $base -MultiArchMirror ""
Check "unset on a default venv stays default" ((Trio $s) -eq $default)
$s = Get-RocmWindowsTorchSet -Requested "" -InstalledTorchVersion "2.10.0+rocm7.13.0" -DefaultIndexBase $base -MultiArchMirror ""
Check "an older default tag is not treated as an opt-in" ((Trio $s) -eq $default)
foreach ($v in @("rocm10.1", "ROCM10.1", " rocm10.1.0 ")) {
    $s = Get-RocmWindowsTorchSet -Requested $v -InstalledTorchVersion "2.11.0+rocm7.14.0" -DefaultIndexBase $base -MultiArchMirror ""
    Check "'$v' selects the 10.1 trio from whl-next" ((Trio $s) -eq $rocm101 -and $s.OptIn)
}
$s = Get-RocmWindowsTorchSet -Requested "" -InstalledTorchVersion "2.14.0+rocm10.1.0" -DefaultIndexBase $base -MultiArchMirror ""
Check "unset keeps an installed 10.1 trio" ((Trio $s) -eq $rocm101)
foreach ($v in @("default", "rocm7.14", "rocm7.14.0")) {
    $s = Get-RocmWindowsTorchSet -Requested $v -InstalledTorchVersion "2.14.0+rocm10.1.0" -DefaultIndexBase $base -MultiArchMirror ""
    Check "'$v' switches an opted-in venv back" ((Trio $s) -eq $default -and -not $s.Unknown)
}
$s = Get-RocmWindowsTorchSet -Requested "rocm10.2" -InstalledTorchVersion "" -DefaultIndexBase $base -MultiArchMirror ""
Check "an unknown value is the default, flagged" ((Trio $s) -eq $default -and $s.Unknown)
$s = Get-RocmWindowsTorchSet -Requested "rocm10.1" -InstalledTorchVersion "" -DefaultIndexBase "https://m.example/amd" -MultiArchMirror "https://m.example/amd/"
Check "a multi-arch mirror serves the 10.1 trio too" ($s.IndexBase -eq "https://m.example/amd")

if ($failures) { Write-Host "$failures failed" -ForegroundColor Red; exit 1 }
Write-Host "all passed"
