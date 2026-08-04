#!/usr/bin/env python
"""One-shot official NESO 2025 download for the frozen V6 experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import urllib.request


SOURCE_URL = (
    "https://api.neso.energy/dataset/"
    "8f2fe0af-871c-488d-8bad-960426f24601/resource/"
    "b2bde559-3455-4021-b179-dfe60c0337b0/download/"
    "demanddata_2025.csv"
)
RESOURCE_ID = "b2bde559-3455-4021-b179-dfe60c0337b0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="data/raw/neso_demanddata_2025.csv",
    )
    parser.add_argument(
        "--audit",
        default="artifacts/neso_v6_download_audit.json",
    )
    parser.add_argument(
        "--execution-snapshot",
        default="artifacts/prospective_execution_snapshot_v6.json",
    )
    parser.add_argument(
        "--config",
        default="configs/v6/neso_demand_2025_v6.yaml",
    )
    parser.add_argument(
        "--protocol",
        default="protocol/EXPERIMENT_PROTOCOL_V6.md",
    )
    args = parser.parse_args()

    output = Path(args.output)
    audit = Path(args.audit)
    snapshot = Path(args.execution_snapshot)
    config = Path(args.config)
    protocol = Path(args.protocol)

    for required in (snapshot, config, protocol):
        if not required.is_file():
            raise FileNotFoundError(
                f"Required frozen input is missing: {required}"
            )
    for target in (output, audit):
        if target.exists():
            raise FileExistsError(
                f"Refusing to overwrite one-shot V6 artifact: {target}"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    audit.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".part")
    if partial.exists():
        raise FileExistsError(partial)

    request = urllib.request.Request(
        SOURCE_URL,
        headers={
            "User-Agent": (
                "pacbayes-tsk-v6-prospective-research/1.0"
            )
        },
    )
    digest = hashlib.sha256()
    size_bytes = 0
    downloaded_utc = datetime.now(timezone.utc).isoformat()
    try:
        with urllib.request.urlopen(
            request,
            timeout=180,
        ) as response, partial.open("xb") as handle:
            final_url = response.geturl()
            selected_headers = {
                name: response.headers.get(name)
                for name in (
                    "Content-Type",
                    "Content-Length",
                    "ETag",
                    "Last-Modified",
                )
                if response.headers.get(name) is not None
            }
            for block in iter(
                lambda: response.read(1024 * 1024),
                b"",
            ):
                handle.write(block)
                digest.update(block)
                size_bytes += len(block)
        if size_bytes == 0:
            raise RuntimeError("NESO download produced an empty file.")
        os.replace(partial, output)
        payload: dict[str, object] = {
            "schema_version": "6.0",
            "status": "DOWNLOADED_NOT_PREPARED",
            "downloaded_utc": downloaded_utc,
            "source_url": SOURCE_URL,
            "final_url": final_url,
            "resource_id": RESOURCE_ID,
            "http_headers": selected_headers,
            "size_bytes": size_bytes,
            "raw_file": output.as_posix(),
            "raw_sha256": digest.hexdigest(),
            "execution_snapshot": snapshot.as_posix(),
            "execution_snapshot_sha256": sha256_file(snapshot),
            "config_sha256": sha256_file(config),
            "protocol_sha256": sha256_file(protocol),
            "required_attribution": (
                "Supported by National Energy SO Open Data"
            ),
        }
        atomic_json(audit, payload)
    except Exception:
        partial.unlink(missing_ok=True)
        if output.exists() and not audit.exists():
            output.unlink()
        raise

    print(f"Downloaded: {output}")
    print(f"SHA-256: {digest.hexdigest()}")
    print(f"Audit: {audit}")


if __name__ == "__main__":
    main()
