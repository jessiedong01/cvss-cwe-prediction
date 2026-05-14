"""
Download NVD CVE JSON feeds from the fkie-cad community mirror.
https://github.com/fkie-cad/nvd-json-data-feeds
"""
import gzip
import shutil
import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
BASE_URL = "https://github.com/fkie-cad/nvd-json-data-feeds/releases/latest/download"


def download_year(year: int, force: bool = False) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / f"CVE-{year}.json"

    if out_path.exists() and not force:
        print(f"[skip] {out_path.name} already exists ({out_path.stat().st_size // 1_000_000} MB)")
        return out_path

    gz_path = DATA_DIR / f"CVE-{year}.json.gz"
    url = f"{BASE_URL}/CVE-{year}.json.gz"
    print(f"Downloading {url} ...")

    def _progress(block, block_size, total):
        done = block * block_size
        pct = done / total * 100 if total > 0 else 0
        print(f"\r  {done // 1_000_000} / {total // 1_000_000} MB ({pct:.1f}%)", end="", flush=True)

    urllib.request.urlretrieve(url, gz_path, reporthook=_progress)
    print()

    print(f"  Decompressing ...")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()
    print(f"  -> {out_path.name} ({out_path.stat().st_size // 1_000_000} MB)")
    return out_path


if __name__ == "__main__":
    years = [int(y) for y in sys.argv[1:]] if len(sys.argv) > 1 else [2022, 2023, 2024]
    for year in years:
        download_year(year)
    print("Done.")
