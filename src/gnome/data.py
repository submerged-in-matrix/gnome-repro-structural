"""Data loading and split utilities for GNoME reproduction.

Reads the two Matbench Discovery MP files from ${GNOME_DATA_DIR}/raw/
(downloaded by scripts/download_data.py) and assigns each material
to train or test based on a deterministic hash of its reduced formula.

We deliberately do NOT import matbench_discovery here. Its eager
top-level imports of WBM files are fragile and unnecessary for training.
We will import matbench_discovery only when packaging WBM predictions
for leaderboard submission (Phase 3).
"""
from __future__ import annotations

import bz2
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Iterator

import pandas as pd
from pymatgen.entries.computed_entries import ComputedStructureEntry
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

# Resolve GNOME_DATA_DIR robustly:
#   - Absolute path → used as-is
#   - Relative path → resolved against the repo root (parent of src/)
# Reason: notebook kernels and scripts launch with different CWDs;
#         relative paths against CWD silently break.
_REPO_ROOT = Path(__file__).resolve().parents[2]   # src/gnome/data.py → repo root
_raw_data_dir = os.environ.get("GNOME_DATA_DIR", "./data")
DATA_DIR = Path(_raw_data_dir)
if not DATA_DIR.is_absolute():
    DATA_DIR = (_REPO_ROOT / DATA_DIR).resolve()
RAW_DIR = DATA_DIR / "raw"
ENERGIES_CSV = RAW_DIR / "2023-01-10-mp-energies.csv"
STRUCTURES_JSON = RAW_DIR / "2023-02-07-mp-computed-structure-entries.json"


# -----------------------------------------------------------------------------
# Composition-based split
# -----------------------------------------------------------------------------

TRAIN_FRACTION_PCT = 85   # 85% train, 15% test


def composition_hash(reduced_formula: str) -> int:
    """MD5 of the reduced formula, mod 100. Deterministic across machines."""
    md5 = hashlib.md5(reduced_formula.encode("utf-8")).hexdigest()
    return int(md5, 16) % 100


def assign_split(reduced_formula: str) -> str:
    """Map a reduced formula to 'train' or 'test'."""
    return "train" if composition_hash(reduced_formula) < TRAIN_FRACTION_PCT else "test"


# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------

def _open_maybe_compressed(path: Path):
    """Open .json, .json.gz, or .json.bz2 transparently as text."""
    p = str(path)
    if p.endswith(".bz2"):
        return bz2.open(p, "rt")
    if p.endswith(".gz"):
        return gzip.open(p, "rt")
    return open(p, "r")


def _check_files_present():
    missing = [f for f in (ENERGIES_CSV, STRUCTURES_JSON) if not f.exists()]
    if missing:
        raise FileNotFoundError(
            "Required data files are missing:\n  "
            + "\n  ".join(str(m) for m in missing)
            + "\n\nRun: python scripts/download_data.py"
        )


def iter_mp_entries() -> Iterator[dict]:
    """Yield one dict per MP material with keys:
        material_id, structure, formula_pretty, e_form_per_atom.

    Uses the MP2020-corrected formation energy as ground truth (this is
    what the Matbench Discovery leaderboard treats as canonical).
    """
    _check_files_present()

    energies_df = pd.read_csv(ENERGIES_CSV).set_index("material_id")

    if "e_form_per_atom_mp2020_corrected" in energies_df.columns:
        e_col = "e_form_per_atom_mp2020_corrected"
    elif "formation_energy_per_atom" in energies_df.columns:
        e_col = "formation_energy_per_atom"
    else:
        raise RuntimeError(
            f"No formation-energy column in {ENERGIES_CSV}. "
            f"Available: {list(energies_df.columns)}"
        )

    with _open_maybe_compressed(STRUCTURES_JSON) as f:
        raw = json.load(f)

    # The Matbench Discovery v1 (which was manually downloaded though!!)  structures file is a column-oriented dataframe
    # dump: raw['material_id'] and raw['entry'] are parallel dicts keyed by row index strings ('0', '1', ...). We zip them.
    
    if not isinstance(raw, dict) or "material_id" not in raw or "entry" not in raw:
        raise RuntimeError(
            f"Unexpected JSON shape in {STRUCTURES_JSON}. "
            f"Top-level keys: {list(raw.keys()) if isinstance(raw, dict) else type(raw).__name__}"
        )

    id_by_idx = raw["material_id"]
    entry_by_idx = raw["entry"]

    for idx in id_by_idx:
        mid = id_by_idx[idx]
        if mid not in energies_df.index:
            continue
        try:
            entry = ComputedStructureEntry.from_dict(entry_by_idx[idx])
        except Exception:
            continue
        row = energies_df.loc[mid]
        formula = row["formula"]
        e_form = row[e_col]
        # Filter: NaN formula or NaN energy → unparseable, skip.
        # Reason: pandas reads missing CSV cells as float NaN.
        if not isinstance(formula, str) or pd.isna(e_form):
            continue
        yield {
            "material_id": mid,
            "structure": entry.structure,
            "formula_pretty": formula,
            "e_form_per_atom": float(e_form),
        }


def load_mp_data(verbose: bool = True) -> list[dict]:
    """Load all MP entries into memory. Returns a list of dicts."""
    it = iter_mp_entries()
    if verbose:
        it = tqdm(it, desc="loading MP entries", total=160_000)
    return list(it)