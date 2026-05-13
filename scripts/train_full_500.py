"""Launcher for Stage A — 500 epochs, EMA, effective batch=256."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from gnome.train_stage_a import StageAConfig, fit_stage_a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   required=True,
                        help="Path to YAML config (e.g. configs/stage_a.yaml)")
    parser.add_argument("--run-name", default=None,
                        help="Override run_name in config")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _REPO_ROOT / config_path

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if args.run_name:
        raw["run_name"] = args.run_name

    cfg = StageAConfig(**raw)
    print(f"Stage A config:\n{raw}\n")
    fit_stage_a(cfg)


if __name__ == "__main__":
    main()
