# Train crop + disease models for each backbone and seed, combine them, then score everything with evaluate.py.
# Resumable: runs that already have results.json are skipped.
#   powershell -ExecutionPolicy Bypass -File run_all.ps1
#   powershell -ExecutionPolicy Bypass -File run_all.ps1 -Backbones "efficientnet_b0,resnet50" -Seeds "123,2026"
# Watch progress from another terminal:  Get-Content train_queue_log.txt -Wait -Tail 20
param(
    [string]$Backbones = "efficientnet_b0,resnet50,convnext_tiny,efficientnet_b3",
    [string]$Seeds = "42,123,2026"
)
Set-Location $PSScriptRoot
$log = "train_queue_log.txt"
$env:PYTHONUNBUFFERED = "1"   # print each epoch as it happens

function Log($msg) { $msg | Tee-Object -FilePath $log -Append }

foreach ($seed in ($Seeds -split ",")) {
    foreach ($bb in ($Backbones -split ",")) {
        foreach ($task in @("crop", "disease")) {
            $done = Get-ChildItem runs/main -Directory -Filter "$($task)_$($bb)_*_s$($seed)" -ErrorAction SilentlyContinue |
                Where-Object { Test-Path (Join-Path $_.FullName "results.json") }
            if ($done) { Log "=== skip $bb $task seed $seed (already finished)"; continue }
            Log "=== $bb $task seed $seed started $(Get-Date -Format 'HH:mm')"
            python train.py --task $task --backbone $bb --seed $seed --img_dir DPD_Small_Dataset --out runs/main 2>$null |
                ForEach-Object { Log $_ }
        }
        $c = (Get-ChildItem runs/main -Directory -Filter "crop_$($bb)_*_s$($seed)" | Select-Object -First 1).FullName
        $d = (Get-ChildItem runs/main -Directory -Filter "disease_$($bb)_*_s$($seed)" | Select-Object -First 1).FullName
        python combine.py --crop_run $c --disease_run $d --out combined/$($bb)_s$($seed) 2>$null | ForEach-Object { Log $_ }
    }
}
# Score every finished run (independent + the teammate's multitask runs, if runs/multitask exists)
python evaluate.py --ind_dir runs/main --mt_dir runs/multitask 2>$null | ForEach-Object { Log $_ }
Log "=== ALL DONE $(Get-Date -Format 'HH:mm')"
