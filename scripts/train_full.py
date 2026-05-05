"""Launch a full training run from a YAML config."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from gnome.train import TrainConfig, fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                         help="Path to YAML config")
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

    cfg = TrainConfig(**raw)
    print(f"Config: {raw}\n")
    fit(cfg)


if __name__ == "__main__":
    main()