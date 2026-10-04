# Runs several long videos through a running server one after another and records the GPU memory peak.
# Usage: powershell -File scripts\stress_test.ps1 <youtube-id> [<youtube-id> ...]
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Ids)
$env:PYTHONUTF8 = "1"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root "server\.venv\Scripts\python.exe"
$out = Join-Path $root "data"
Set-Location $out

$sampler = Start-Job -ArgumentList $out -ScriptBlock {
    param($out)
    $max = 0
    while ($true) {
        $v = [int]((nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -replace '\D', '')
        if ($v -gt $max) { $max = $v; Set-Content -Path (Join-Path $out "vram_peak.txt") -Value $max }
        Start-Sleep -Milliseconds 300
    }
}
$sw = [Diagnostics.Stopwatch]::StartNew()
foreach ($id in $Ids) {
    $log = Join-Path $out "stress_$id.txt"
    & $py (Join-Path $root "scripts\e2e_test.py") "https://www.youtube.com/watch?v=$id" 0 900 > $log 2>&1
    $last = Get-Content $log -Encoding UTF8 | Select-String "\] (done|error)|ERROR|timeout|ready chunks"
    Write-Output "$id :: $($last -join ' | ')"
}
Write-Output ("total wall time: {0:N0}s" -f $sw.Elapsed.TotalSeconds)
Stop-Job $sampler; Remove-Job $sampler
Write-Output "VRAM peak: $(Get-Content (Join-Path $out 'vram_peak.txt')) MiB"
