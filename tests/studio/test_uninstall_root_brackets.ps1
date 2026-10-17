#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Install roots may contain [ ] (the installer creates them through .NET for that reason), so
# "is this path under a root" must be a literal prefix test, not a -like pattern.
# Run: pwsh -NoProfile -File tests/studio/test_uninstall_root_brackets.ps1

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path
$ps1Path = [System.IO.Path]::Combine($repoRoot, "scripts", "uninstall.ps1")

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($ps1Path, [ref]$tokens, [ref]$errors)
Check "uninstall.ps1 parses" ($null -eq $errors -or $errors.Count -eq 0)

$ps1Text = Get-Content -LiteralPath $ps1Path -Raw
Check "no root is matched as a -like pattern" `
    (-not ($ps1Text -match '-i?like "\$(r|rNorm)\\\*"'))

$fn = $ast.FindAll({
        param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq "_PathUnderRoot"
    }, $true) | Select-Object -First 1
if (-not $fn) {
    Write-Host "  FAIL  _PathUnderRoot not found in uninstall.ps1" -ForegroundColor Red
    exit 1
}
. ([scriptblock]::Create($fn.Extent.Text))

$r = "D:\AI [x]\studio"
Check "a path under a bracketed root matches it" `
    (_PathUnderRoot "D:\AI [x]\studio\unsloth_studio\Scripts\python.exe" $r)
Check "the root itself matches" (_PathUnderRoot "d:\ai [X]\STUDIO" $r)
Check "a trailing separator on the root is ignored" (_PathUnderRoot "D:\AI [x]\studio\bin" "$r\")
Check "the sibling the pattern would have matched does not" (-not (_PathUnderRoot "D:\AI x\studio\bin" $r))
Check "a directory sharing the prefix does not" (-not (_PathUnderRoot "D:\AI [x]\studio2\bin" $r))
$threw = $false
try { $unbalanced = _PathUnderRoot "D:\AI [old\studio\bin" "D:\AI [old\studio" } catch { $threw = $true }
Check "an unbalanced [ neither throws nor misses" ((-not $threw) -and $unbalanced)
Check "an empty root matches nothing" (-not (_PathUnderRoot "D:\anything" ""))
Check "an empty path matches nothing" (-not (_PathUnderRoot "" $r))

if ($failures -gt 0) { Write-Host "$failures check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "All bracketed-root checks passed"
