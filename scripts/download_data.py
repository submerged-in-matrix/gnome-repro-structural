"""Download the two Matbench Discovery MP files we need, directly from Figshare.

We bypass the matbench_discovery package for data fetching because its
HTTP layer produces empty files against current Figshare URLs on Windows.
The package is still useful for evaluation utilities (Phase 3).

URLs are stable Figshare 'file' endpoints from the v1 release
(DOI 10.6084/m9.figshare.22715158). If any URL breaks, find the
replacement via the dataset page:
    https://figshare.com/articles/dataset/22715158
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


# Direct Figshare 'file' endpoints. Each tuple is (url, expected_min_bytes).
# expected_min_bytes is a sanity floor: a successful download must exceed it.
DOWNLOADS = {
    "2023-02-07-mp-energies.csv": (
        "https://figshare.com/ndownloader/files/40344436",
        5 * 1024 * 1024,            # ~10 MB; 5 MB floor
    ),
    "2023-02-07-mp-computed-structure-entries.json.bz2": (
        "https://figshare.com/ndownloader/files/40344473",
        100 * 1024 * 1024,          # ~250 MB; 100 MB floor
    ),
}


def download_with_progress(url: str, dest: Path) -> int:
    """Stream-download url to dest. Returns bytes written.

    Writes to a .part file, then renames atomically on success so a
    half-download can never masquerade as complete.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  Downloading {url}\n  -> {dest}")

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "gnome-repro/0.1")]

    with opener.open(url) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        total_mb = total / (1024 * 1024) if total else None
        written = 0
        chunk_size = 1 << 16  # 64 KB
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                written += len(chunk)
                if total_mb:
                    pct = 100 * written / total
                    print(f"\r  {written/(1024*1024):7.1f} MB / "
                          f"{total_mb:7.1f} MB ({pct:5.1f}%)", end="")
                else:
                    print(f"\r  {written/(1024*1024):7.1f} MB", end="")
        print()

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
            print(f"  Skipping {fname} (already present, {mb:.1f} MB)")
            continue

        try:
            n = download_with_progress(url, dest)
        except Exception as e:
            print(f"\nERROR downloading {fname}: {e}", file=sys.stderr)
            print(f"You can manually download from {url}", file=sys.stderr)
            print(f"and place the file at {dest}", file=sys.stderr)
            return 1

        if n < min_bytes:
            print(f"\nWARNING: downloaded {fname} is suspiciously small "
                  f"({n} bytes, expected > {min_bytes}).",
                  file=sys.stderr)
            print(f"The Figshare URL may have changed. Check {url}",
                  file=sys.stderr)
            return 1

    print("\nAll files present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())