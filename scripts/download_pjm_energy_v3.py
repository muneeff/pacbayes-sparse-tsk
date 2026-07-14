from __future__ import annotations
import hashlib
from pathlib import Path
from urllib.request import urlopen

FILES = {
    "AEP_hourly.csv": ("https://raw.githubusercontent.com/ping543f/ren-energy/master/Data/AEP_hourly.csv", "109d122f7b485555c609eecdec2cd5a03172e0a08acd358d863acc93eb452585"),
    "COMED_hourly.csv": ("https://raw.githubusercontent.com/ping543f/ren-energy/master/Data/COMED_hourly.csv", "2e79007e3f1be8765c92ff2b26314c0df0507bdd783110388a1e8d678d13fa1e"),
    "DAYTON_hourly.csv": ("https://raw.githubusercontent.com/ping543f/ren-energy/master/Data/DAYTON_hourly.csv", "84d3397231d5819b5cf086ef9e87cadf6dd7954085f2834be7e37257702ec768"),
    "PJME_hourly.csv": ("https://raw.githubusercontent.com/ping543f/ren-energy/master/Data/PJME_hourly.csv", "4eb2b16d42bf07ec41ab55cb842191594cb69452725a6d3c0991658a628fde84"),
}

def main() -> None:
    destination = Path("data/raw/pjm")
    destination.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in FILES.items():
        target = destination / name
        payload = urlopen(url, timeout=120).read()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Hash mismatch for {name}: {actual} != {expected}")
        target.write_bytes(payload)
        print(f"verified {name}: {actual}")

if __name__ == "__main__":
    main()
