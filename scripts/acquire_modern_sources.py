"""Acquire pinned public source files, without executing any downloaded code."""

import hashlib
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(".runtime/public/modern-trending")
REGIONS = ("BR", "US", "GB", "CA", "DE", "FR", "MX")
VERSION = 1346


def acquire(region, root=ROOT):
    name = f"{region}_youtube_trending_data.csv"
    url = (
        "https://www.kaggle.com/api/v1/datasets/download/"
        f"rsrishav/youtube-trending-video-dataset/{name}?datasetVersionNumber={VERSION}"
    )
    archive = root / f"{region}.zip"
    if not archive.exists():
        with httpx.stream("GET", url, follow_redirects=True, timeout=90) as response:
            response.raise_for_status()
            if "zip" not in response.headers.get("content-type", ""):
                raise ValueError("Expected public ZIP; refusing to save a login/error page")
            with archive.with_suffix(".partial").open("wb") as out:
                for block in response.iter_bytes(1024 * 1024):
                    out.write(block)
        archive.with_suffix(".partial").replace(archive)
    # ZipFile.read verifies CRC. Read only the exact declared member, never extract paths.
    with zipfile.ZipFile(archive) as source:
        if source.namelist() != [name]:
            raise ValueError("Unexpected archive contents")
        target = root / name
        if not target.exists():
            with source.open(name) as stream, target.with_suffix(".partial").open("wb") as out:
                while block := stream.read(1024 * 1024):
                    out.write(block)
            target.with_suffix(".partial").replace(target)
    with target.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    print(f"Verified {region}: {target.stat().st_size:,} bytes", flush=True)
    return {
        "region": region,
        "url": url,
        "file": name,
        "bytes": target.stat().st_size,
        "sha256": digest,
    }


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    # Independent public downloads; bounded concurrency and no authentication workaround.
    with ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(acquire, REGIONS))
    manifest = {
        "dataset": "rsrishav/youtube-trending-video-dataset",
        "version": VERSION,
        "files": files,
    }
    (ROOT / "acquisition.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
