"""Download Matbench Discovery files directly from confirmed Figshare URLs.

Browser-like headers + allow_redirects required; Figshare blocks bare urllib
and returns HTML/empty response without them.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

try:
    import requests
except ImportError:
    print("ERROR: pip install requests", file=sys.stderr)
    sys.exit(1)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/octet-stream,*/*",
}

# Confirmed working URLs from browser test.
DOWNLOADS = {
    "mp-energies.csv":                       "https://figshare.com/ndownloader/files/49083124",
    "mp-computed-structure-entries.json.gz": "https://figshare.com/ndownloader/files/40344436",
    "wbm-summary.csv.gz":                    "https://figshare.com/ndownloader/files/44225498",
}


def download(url: str, dest: Path) -> bool:
    """Stream to .part file, rename on completion; browser headers + redirects."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    try:
        with requests.get(
            url,
            headers=HEADERS,
            stream=True,
            allow_redirects=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()

            # Figshare sometimes returns HTML on a bad request; catch it early.
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" in content_type:
                print(f"\n  ERROR: got HTML instead of file (URL may have changed).",
                      file=sys.stderr)
                return False

            total = int(resp.headers.get("Content-Length") or 0)
            total_mb = total / 1024 / 1024 if total else None
            written = 0

            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
                    written += len(chunk)
                    mb = written / 1024 / 1024
                    if total_mb:
                        pct = 100 * written / total
                        print(f"\r  {mb:7.1f} / {total_mb:.1f} MB  ({pct:5.1f}%)",
                              end="", flush=True)
                    else:
                        print(f"\r  {mb:7.1f} MB", end="", flush=True)
        print()
    except Exception as e:
        print(f"\n  ERROR: {e}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False

    tmp.replace(dest)
    size_mb = dest.stat().st_size / 1024 / 1024
    print(f"  Saved: {dest.name}  ({size_mb:.1f} MB)")
    return True


def main() -> int:
    data_dir = Path(os.environ.get("GNOME_DATA_DIR", "./data"))
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target dir: {raw_dir.resolve()}\n")

    results = []
    for fname, url in DOWNLOADS.items():
        dest = raw_dir / fname
        print(f"[{fname}]\n  {url}")

        if dest.exists() and dest.stat().st_size > 1024:
            print(f"  Already present ({dest.stat().st_size/1024/1024:.1f} MB), skipping.\n")
            results.append((fname, "skipped"))
            continue

        ok = download(url, dest)
        results.append((fname, "ok" if ok else "FAILED"))
        print()

    # Sanity check on WBM summary row count.
    summary = raw_dir / "wbm-summary.csv.gz"
    if summary.exists():
        try:
            import pandas as pd
            df = pd.read_csv(summary)
            expected = 256_963
            status = "OK" if df.shape[0] == expected else "MISMATCH"
            print(f"WBM summary [{status}]: {df.shape[0]:,} rows × {df.shape[1]} cols\n")
        except Exception as e:
            print(f"WBM summary check FAILED: {e}\n", file=sys.stderr)

    print("Summary:")
    for fname, status in results:
        print(f"  {status:8s}  {fname}")

    return 1 if any(s == "FAILED" for _, s in results) else 0


if __name__ == "__main__":
    sys.exit(main())