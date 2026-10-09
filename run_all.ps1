# Train crop + disease models for each backbone, then combine them.
# Resumable: runs that already have results.json are skipped.
#   powershell -ExecutionPolicy Bypass -File run_all.ps1
# Watch progress from another terminal:  Get-Content train_queue_log.txt -Wait -Tail 20
Set-Location $PSScriptRoot
$log = "train_queue_log.txt"
$env:PYTHONUNBUFFERED = "1"   # print each epoch as it happens

function Log($msg) { $msg | Tee-Object -FilePath $log -Append }

foreach ($bb in @("efficientnet_b0", "resnet50", "convnext_tiny", "efficientnet_b3")) {
    foreach ($task in @("crop", "disease")) {
        $done = Get-ChildItem runs/main -Directory -Filter "$($task)_$($bb)_*" -ErrorAction SilentlyContinue |
            Where-Object { Test-Path (Join-Path $_.FullName "results.json") }
        if ($done) { Log "=== skip $bb $task (already finished)"; continue }
        Log "=== $bb $task started $(Get-Date -Format 'HH:mm')"
        python train.py --task $task --backbone $bb --img_dir DPD_Small_Dataset --out runs/main 2>$null |
            ForEach-Object { Log $_ }
    }
    $c = (Get-ChildItem runs/main -Directory -Filter "crop_$($bb)_*" | Select-Object -First 1).FullName
    $d = (Get-ChildItem runs/main -Directory -Filter "disease_$($bb)_*" | Select-Object -First 1).FullName
    python combine.py --crop_run $c --disease_run $d --out combined/$bb 2>$null | ForEach-Object { Log $_ }
}
Log "=== ALL DONE $(Get-Date -Format 'HH:mm')"
