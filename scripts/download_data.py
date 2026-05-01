from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

DOWNLOADS = {
    "2025-02-01-mp-energies.csv.gz": (
        "https://figshare.com/ndownloader/files/40344436",
        5 * 1024 * 1024,
    ),
    "2023-02-07-mp-computed-structure-entries.json.gz": (
        "https://figshare.com/ndownloader/files/40344473",
        100 * 1024 * 1024,
    ),
}

def download_with_progress(url: str, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    with requests.get(url, stream=True, allow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        written = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
                    written += len(chunk)

    if written == 0:
        raise RuntimeError(f"Empty download from {url}")

    tmp.replace(dest)
    return written

def main() -> int:
    data_dir = Path(os.environ.get("GNOME_DATA_DIR", "./data"))
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {raw_dir.resolve()}\n")

    for fname, (url, min_bytes) in DOWNLOADS.items():
        dest = raw_dir / fname
        if dest.exists() and dest.stat().st_size > min_bytes:
            mb = dest.stat().st_size / (1024 * 1024)
            print(f"Skipping {fname} (already present, {mb:.1f} MB)")
            continue

        try:
            n = download_with_progress(url, dest)
        except Exception as e:
            print(f"ERROR downloading {fname}: {e}", file=sys.stderr)
            print(f"Check the current Figshare file ID for {fname}", file=sys.stderr)
            return 1

        if n < min_bytes:
            print(
                f"WARNING: downloaded {fname} is suspiciously small ({n} bytes, expected > {min_bytes}).",
                file=sys.stderr,
            )
            return 1

    print("\nAll files present.")
    return 0

if __name__ == "__main__":
    sys.exit(main())