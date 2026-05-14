# scripts/run_wbm_eval_stage_a.ps1
# Runs WBM inference then F1 scoring for Stage A checkpoint.

Write-Host "=== Stage A WBM Eval ===" -ForegroundColor Cyan

python scripts/eval_wbm.py `
    --checkpoint runs/stage_a/best.pt `
    --out-dir runs/stage_a

if ($LASTEXITCODE -ne 0) {
    Write-Host "eval_wbm.py failed. Stopping." -ForegroundColor Red
    exit 1
}

Write-Host "`n=== Stage A F1 Scoring ===" -ForegroundColor Cyan

python scripts/f1_wbm.py `
    --predictions runs/stage_a/predictions_wbm.csv `
    --out-dir runs/stage_a

if ($LASTEXITCODE -ne 0) {
    Write-Host "f1_wbm.py failed." -ForegroundColor Red
    exit 1
}

Write-Host "`nDone. Results in runs/stage_a/" -ForegroundColor Green