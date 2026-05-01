"""Data loading and split utilities for GNoME reproduction.

Reads the two Matbench Discovery MP files from ${GNOME_DATA_DIR}/raw/
(downloaded by scripts/download_data.py) and assigns each material
to train or test based on a deterministic hash of its reduced formula.

do NOT import matbench_discovery here. because Its eager
top-level imports of WBM files are fragile and unnecessary for training :xD.
import matbench_discovery only when packaging WBM predictions
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

DATA_DIR = Path(os.environ.get("GNOME_DATA_DIR", "./data"))
RAW_DIR = DATA_DIR / "raw"

ENERGIES_CSV = RAW_DIR / "2023-02-07-mp-energies.csv"
STRUCTURES_JSON = RAW_DIR / "2023-02-07-mp-computed-structure-entries.json.bz2"


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

    if isinstance(raw, dict):
        raw = raw.get("data", raw.get("entries", list(raw.values())))

    for d in raw:
        try:
            entry = ComputedStructureEntry.from_dict(d)
        except Exception:
            continue
        mid = entry.entry_id
        if mid not in energies_df.index:
            continue
        row = energies_df.loc[mid]
        yield {
            "material_id": mid,
            "structure": entry.structure,
            "formula_pretty": row["formula_pretty"],
            "e_form_per_atom": float(row[e_col]),
        }


def load_mp_data(verbose: bool = True) -> list[dict]:
    """Load all MP entries into memory. Returns a list of dicts."""
    it = iter_mp_entries()
    if verbose:
        it = tqdm(it, desc="loading MP entries", total=160_000)
    return list(it)