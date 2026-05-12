$ErrorActionPreference = "Stop"   # equivalent of set -euo pipefail

Write-Host "=== Stage B Ablations ==="
Write-Host "Started: $(Get-Date)"

python scripts/train_full.py --config configs/ablation_no_norm.yaml
Write-Host "B1 done: $(Get-Date)"

python scripts/train_full.py --config configs/ablation_small.yaml
Write-Host "B2 done: $(Get-Date)"

python scripts/train_full.py --config configs/ablation_shallow.yaml
Write-Host "B3 done: $(Get-Date)"

Write-Host "=== All ablations complete: $(Get-Date) ==="
