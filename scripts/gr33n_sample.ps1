param([string]$Phase, [int]$N = 20)
$tok = (Get-Content "$env:USERPROFILE\lmr-token.txt" -Raw).Trim()
$h = @{ Authorization = "Bearer $tok" }
$out = "$env:USERPROFILE\lmr-acc"; New-Item -ItemType Directory -Force $out | Out-Null
$rows = @()
$serverPid = (Get-NetTCPConnection -State Listen -LocalPort 8893).OwningProcess | Select-Object -First 1
for ($i = 0; $i -lt $N; $i++) {
  $t0 = Get-Date
  $sys = Invoke-RestMethod "http://127.0.0.1:8893/api/system?gpu_max_age=1" -Headers $h
  $inf = Invoke-RestMethod "http://127.0.0.1:8893/api/inference/status" -Headers $h
  $os = Get-CimInstance Win32_OperatingSystem
  $ramUsedTruth = ($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1MB
  $adapters = (Get-Counter "\GPU Adapter Memory(*)\Dedicated Usage").CounterSamples | Sort-Object CookedValue -Descending
  $procs = (Get-Counter "\GPU Process Memory(*)\Dedicated Usage").CounterSamples
  # Server process tree
  $all = Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId, Name, WorkingSetSize
  $tree = @($serverPid); $added = $true
  while ($added) { $added = $false; foreach ($p in $all) { if ($tree -contains $p.ParentProcessId -and -not ($tree -contains $p.ProcessId)) { $tree += $p.ProcessId; $added = $true } } }
  $treeWs = ($all | Where-Object { $tree -contains $_.ProcessId } | Measure-Object WorkingSetSize -Sum).Sum / 1GB
  $treePrivate = 0
  foreach ($id in $tree) { try { $treePrivate += (Get-Process -Id $id).PrivateMemorySize64 } catch {} }
  $treeGpu = ($procs | Where-Object { $m = [regex]::Match($_.InstanceName, 'pid_(\d+)_'); $m.Success -and ($tree -contains [int]$m.Groups[1].Value) } | Measure-Object CookedValue -Sum).Sum / 1GB
  $llama = $all | Where-Object { $_.Name -like 'llama-server*' -and ($tree -contains $_.ProcessId) } | Select-Object -First 1
  $llamaGpu = if ($llama) { ($procs | Where-Object { $_.InstanceName -like "pid_$($llama.ProcessId)_*" } | Measure-Object CookedValue -Sum).Sum / 1GB } else { 0 }
  $devs = @($sys.gpu.devices)
  $resident = 0; foreach ($k in @('resident_mib')) { if ($inf.$k) { $resident = $inf.$k / 1024 } }
  $rows += [pscustomobject]@{
    phase = $Phase; t = $t0.ToString('HH:mm:ss.fff')
    cpu_pct = $sys.cpu.usage_percent
    ram_used_monitor = [math]::Round($sys.memory.total_gb - $sys.memory.available_gb, 3)
    ram_used_truth = [math]::Round($ramUsedTruth, 3)
    unsloth_ram_monitor = [math]::Round($sys.memory.process_tree_used_mb / 1024, 3)
    unsloth_ram_truth_ws = [math]::Round($treeWs, 3)
    unsloth_ram_truth_private = [math]::Round($treePrivate / 1GB, 3)
    gpu0_used_monitor = $devs[0].vram_used_gb
    gpu1_used_monitor = if ($devs.Count -gt 1) { $devs[1].vram_used_gb } else { $null }
    adapter_dedicated_top2 = (($adapters | Select-Object -First 2 | ForEach-Object { [math]::Round($_.CookedValue / 1GB, 3) }) -join ';')
    unsloth_vram_truth = [math]::Round($treeGpu, 3)
    llama_vram_truth = [math]::Round($llamaGpu, 3)
    model_resident_monitor = [math]::Round($resident, 3)
  }
  Start-Sleep -Milliseconds ([math]::Max(0, 1000 - ((Get-Date) - $t0).TotalMilliseconds))
}
$rows | Export-Csv "$out\$Phase.csv" -NoTypeInformation
$sys | ConvertTo-Json -Depth 8 | Out-File "$out\$Phase-system.json"
$inf | ConvertTo-Json -Depth 8 | Out-File "$out\$Phase-inference-status.json"
($adapters | ForEach-Object { "$($_.InstanceName)`t$([math]::Round($_.CookedValue/1GB,3))" }) | Out-File "$out\$Phase-adapters.txt"
"done $Phase $($rows.Count)"
