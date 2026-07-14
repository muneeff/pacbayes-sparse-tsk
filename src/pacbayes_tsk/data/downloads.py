from __future__ import annotations

import hashlib
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

from tqdm import tqdm


class DownloadError(RuntimeError):
    """Raised when every configured download source fails."""


def file_hash(path: str | Path, algorithm: str = "md5") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(
    path: str | Path,
    expected: str,
    algorithm: str = "md5",
) -> bool:
    return file_hash(path, algorithm=algorithm).lower() == expected.lower()


def _download_one(
    url: str,
    destination: Path,
    *,
    timeout_seconds: int,
    user_agent: str,
) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": user_agent},
    )

    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        total = int(response.headers.get("Content-Length", "0") or 0)
        temporary = destination.with_suffix(destination.suffix + ".part")
        destination.parent.mkdir(parents=True, exist_ok=True)

        with temporary.open("wb") as output, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        ) as progress:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                progress.update(len(block))

        temporary.replace(destination)


def download_with_retries(
    urls: Iterable[str],
    destination: str | Path,
    *,
    expected_md5: str,
    retries_per_url: int = 4,
    timeout_seconds: int = 120,
    base_delay_seconds: float = 3.0,
    user_agent: str = "Sparse-TSK-PACBayes-Research/1.0",
    overwrite: bool = False,
) -> tuple[Path, str]:
    """
    Download from ordered sources, retrying rate limits and transient failures.
    """
    output = Path(destination)

    if output.exists() and not overwrite:
        if verify_hash(output, expected_md5, "md5"):
            return output, "existing_verified"
        raise DownloadError(
            f"Existing file has the wrong MD5: {output}. "
            "Delete it or run with overwrite enabled."
        )

    failures: list[str] = []
    for url in urls:
        for attempt in range(1, retries_per_url + 1):
            try:
                _download_one(
                    url,
                    output,
                    timeout_seconds=timeout_seconds,
                    user_agent=user_agent,
                )
                if not verify_hash(output, expected_md5, "md5"):
                    actual = file_hash(output, "md5")
                    output.unlink(missing_ok=True)
                    raise DownloadError(
                        f"Checksum mismatch from {url}: "
                        f"expected {expected_md5}, received {actual}."
                    )
                return output, url

            except urllib.error.HTTPError as exc:
                retry_after = exc.headers.get("Retry-After")
                wait = (
                    float(retry_after)
                    if retry_after and retry_after.isdigit()
                    else base_delay_seconds * (2 ** (attempt - 1))
                )
                failures.append(
                    f"{url} attempt {attempt}: HTTP {exc.code}"
                )
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    break
                time.sleep(wait)

            except (
                urllib.error.URLError,
                TimeoutError,
                OSError,
                DownloadError,
            ) as exc:
                failures.append(f"{url} attempt {attempt}: {exc}")
                time.sleep(base_delay_seconds * (2 ** (attempt - 1)))

    raise DownloadError(
        "All download attempts failed:\n" + "\n".join(failures)
    )


def safe_extract_zip(
    archive_path: str | Path,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Extract a ZIP while rejecting path traversal entries."""
    archive = Path(archive_path)
    output_root = Path(destination)
    output_root.mkdir(parents=True, exist_ok=True)
    root_resolved = output_root.resolve()

    extracted: list[Path] = []
    with zipfile.ZipFile(archive, "r") as zip_handle:
        for member in zip_handle.infolist():
            target = (output_root / member.filename).resolve()
            try:
                target.relative_to(root_resolved)
            except ValueError as exc:
                raise DownloadError(
                    f"Unsafe ZIP member path: {member.filename}"
                ) from exc

            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            if target.exists() and not overwrite:
                extracted.append(target)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            with zip_handle.open(member, "r") as source, target.open(
                "wb"
            ) as sink:
                shutil.copyfileobj(source, sink)
            extracted.append(target)

    return extracted
