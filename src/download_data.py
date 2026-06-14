"""
Download NVD CVE JSON feeds from the fkie-cad community mirror.
https://github.com/fkie-cad/nvd-json-data-feeds
"""
import lzma
import shutil
import sys
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data"
# Pinned mirror snapshot so results are reproducible. NVD records change over time.
RELEASE = "v2026.10.05-000016"
BASE_URL = f"https://github.com/fkie-cad/nvd-json-data-feeds/releases/download/{RELEASE}"


def download_year(year: int, force: bool = False) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / f"CVE-{year}.json"

    if out_path.exists() and not force:
        print(f"[skip] {out_path.name} already exists ({out_path.stat().st_size // 1_000_000} MB)")
        return out_path

    xz_path = DATA_DIR / f"CVE-{year}.json.xz"
    url = f"{BASE_URL}/CVE-{year}.json.xz"
    print(f"Downloading {url} ...")

    with requests.get(url, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(xz_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                pct = done / total * 100 if total else 0
                print(f"\r  {done // 1_000_000} / {total // 1_000_000} MB ({pct:.1f}%)", end="", flush=True)
    print()

    print(f"  Decompressing ...")
    with lzma.open(xz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    xz_path.unlink()
    print(f"  -> {out_path.name} ({out_path.stat().st_size // 1_000_000} MB)")
    return out_path


if __name__ == "__main__":
    years = [int(y) for y in sys.argv[1:]] if len(sys.argv) > 1 else [2022, 2023, 2024]
    for year in years:
        download_year(year)
    print("Done.")
