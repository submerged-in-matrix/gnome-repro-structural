# Serial 6-seed ensemble on a single GPU (Windows / PowerShell).
# Any seed whose best.pt already exists is skipped, so a reboot or crash
# resumes at the first unfinished seed instead of retraining earlier ones.

$ErrorActionPreference = "Stop"

# Repo root; edit this one line if the project is moved.
$repo    = ".\"
$dataDir = Join-Path $repo "data"
$runsDir = Join-Path $repo "runs"
$seeds   = 0..5
$epochs  = 3
# Physical batch size per forward pass.  Halved from the 128 default so the
# Adam foreach kernels fit in GPU memory; accum_steps is doubled to keep the
# effective batch at 256 (= 64 × 4).
$batchSize  = 64
$accumSteps = 4

foreach ($seed in $seeds) {
    # summary.json is written only after the full epoch loop completes, so it
    # is a true done-marker; best.pt appears mid-run and would falsely skip a
    # seed interrupted partway through.
    $done = Join-Path $runsDir "ensemble\seed_$seed\summary.json"
    if (Test-Path $done) {
        Write-Host "seed $seed : summary.json present (fully trained) -> skip" -ForegroundColor Yellow
        continue
    }
    Write-Host "=== training seed $seed ($epochs epochs) ===" -ForegroundColor Cyan
    python -m gnome.train_ensemble_seed `
        --seed $seed `
        --epochs $epochs `
        --data-dir $dataDir `
        --runs-dir $runsDir `
        --batch-size $batchSize `
        --accum-steps $accumSteps
    if ($LASTEXITCODE -ne 0) {
        Write-Host "seed $seed FAILED (exit $LASTEXITCODE) -> stop" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "seed $seed done." -ForegroundColor Green
}

Write-Host "All 6 seeds complete. Checkpoints under runs\ensemble\seed_*\best.pt" -ForegroundColor Green
