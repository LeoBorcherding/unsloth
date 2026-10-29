$tok = (Get-Content "$env:USERPROFILE\lmr-token.txt" -Raw).Trim(); $h = @{ Authorization = "Bearer $tok" }
$r = @(); for ($i=0;$i -lt 30;$i++){ $t0=Get-Date; $sw=[Diagnostics.Stopwatch]::StartNew(); $s=Invoke-RestMethod "http://127.0.0.1:8893/api/system?gpu_max_age=1" -Headers $h; $ms=$sw.Elapsed.TotalMilliseconds
 $r += [pscustomobject]@{i=$i; t=$t0.ToString('HH:mm:ss.fff'); ms=[math]::Round($ms,1); gpu0=$s.gpu.devices[0].vram_used_gb; gpu1=$s.gpu.devices[1].vram_used_gb}
 Start-Sleep -Milliseconds ([math]::Max(0,1000-((Get-Date)-$t0).TotalMilliseconds)) }
$r | Export-Csv "$env:USERPROFILE\lmr-acc\latency-$($args[0]).csv" -NoTypeInformation
$m = $r.ms | Sort-Object; "median {0} ms  p95 {1} ms  max {2} ms  slow(>500ms) {3}/30" -f $m[15],$m[28],$m[29],(@($r|?{$_.ms -gt 500}).Count)
