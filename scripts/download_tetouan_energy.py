"""Download the UCI Tetouan archive and extract the source CSV.

Use this on a networked machine before manuscript submission. The script does
not silently fall back to a mirror; provenance remains explicit.
"""
from __future__ import annotations
import argparse, hashlib, urllib.request, zipfile
from pathlib import Path

UCI_URL = "https://archive.ics.uci.edu/static/public/849/power%2Bconsumption%2Bof%2Btetouan%2Bcity.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "tetouan_uci_849.zip"
    urllib.request.urlretrieve(UCI_URL, archive)
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        csv_names = [name for name in names if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise RuntimeError(f"Unexpected UCI archive contents: {names}")
        extracted = Path(handle.extract(csv_names[0], output))
    target = output / "tetouan_power_consumption.csv"
    extracted.replace(target)
    print(f"Downloaded: {target}")
    print(f"SHA-256: {sha256(target)}")


if __name__ == "__main__":
    main()
