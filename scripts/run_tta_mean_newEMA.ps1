# scripts/run_today.ps1
# Automates all runs for today in sequence:
#   Step 1 - Baseline (200ep, no EMA) min-TTA WBM eval
#   Step 2 - Stage A (500ep, EMA-0.999) min-TTA WBM eval
#   Step 3 - F1 scoring for both min-TTA runs
#   Step 4 - Stage A EMA-0.99 training (500ep, seed=0)
#
# Outputs are kept separate from original mean-TTA results:
#   runs/default/min_tta/     <- baseline min-TTA
#   runs/stage_a/min_tta/     <- stage_a min-TTA
#   runs/stage_a_ema99/       <- new EMA-0.99 training
#
# Usage:
#   .\scripts\run_today.ps1
# ---------------------------------------------------------------------------

$ErrorActionPreference = "Stop"

function Print-Header($msg) {
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
}

function Check-Exit($step) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "ERROR: $step failed (exit code $LASTEXITCODE). Stopping." -ForegroundColor Red
        exit 1
    }
}

# ---------------------------------------------------------------------------
# Step 1 - Baseline min-TTA eval
# ---------------------------------------------------------------------------
Print-Header "Step 1/4 - Baseline (200ep) WBM eval with min-TTA"

python scripts/eval_wbm.py `
    --checkpoint runs/default/best.pt `
    --out-dir    runs/default/min_tta `
    --aggregator min
Check-Exit "Baseline min-TTA eval"

python scripts/f1_wbm.py `
    --predictions runs/default/min_tta/predictions_wbm.csv `
    --out-dir     runs/default/min_tta
Check-Exit "Baseline min-TTA F1"

Write-Host "Baseline min-TTA done. Results in runs/default/min_tta/" -ForegroundColor Green

# ---------------------------------------------------------------------------
# Step 2 - Stage A min-TTA eval
# ---------------------------------------------------------------------------
Print-Header "Step 2/4 - Stage A (500ep EMA-0.999) WBM eval with min-TTA"

python scripts/eval_wbm.py `
    --checkpoint runs/stage_a/best.pt `
    --out-dir    runs/stage_a/min_tta `
    --aggregator min
Check-Exit "Stage A min-TTA eval"

python scripts/f1_wbm.py `
    --predictions runs/stage_a/min_tta/predictions_wbm.csv `
    --out-dir     runs/stage_a/min_tta
Check-Exit "Stage A min-TTA F1"

Write-Host "Stage A min-TTA done. Results in runs/stage_a/min_tta/" -ForegroundColor Green

# ---------------------------------------------------------------------------
# Step 3 - Summary before training
# ---------------------------------------------------------------------------
Print-Header "Step 3/4 - Min-TTA eval complete. Summary:"

Write-Host ""
Write-Host "  Baseline mean-TTA F1 : 0.365  (runs/default/f1_wbm.json)"
Write-Host "  Stage A  mean-TTA F1 : 0.363  (runs/stage_a/f1_wbm.json)"
Write-Host "  Baseline min-TTA  F1 : check  runs/default/min_tta/f1_wbm.json"
Write-Host "  Stage A  min-TTA  F1 : check  runs/stage_a/min_tta/f1_wbm.json"
Write-Host ""
Write-Host "Starting EMA-0.99 training in 10 seconds. Ctrl+C to abort." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# ---------------------------------------------------------------------------
# Step 4 - Stage A EMA-0.99 training
# ---------------------------------------------------------------------------
Print-Header "Step 4/4 - Training Stage A EMA-0.99 (500ep, seed=0)"
Write-Host "Expected wall time: ~7.5h on RTX 4070 Ti" -ForegroundColor Yellow
Write-Host "Output: runs/stage_a_ema99/" -ForegroundColor Yellow

python scripts/train_full_500.py --config configs/stage_a_ema99.yaml
Check-Exit "Stage A EMA-0.99 training"

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  All done." -ForegroundColor Green
Write-Host "  Eval results : runs/default/min_tta/ and runs/stage_a/min_tta/" -ForegroundColor Green
Write-Host "  New training : runs/stage_a_ema99/" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
