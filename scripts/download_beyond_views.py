"""Download the public, pinned Dataverse archive; verify it before extracting two members."""

import hashlib
import json
import os
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

ROOT = Path(".runtime/public/beyond-views")
SIZE = 2512142102
MD5 = "1f01676fb6abd3ddc47e1289d64a2803"
URL = "https://dataverse.harvard.edu/api/access/datafile/3489832"


def main():
    root = ROOT / "source"
    root.mkdir(parents=True, exist_ok=True)
    target = root / "tweeted_videos1.tar.bz2"
    partial = root / "tweeted_videos1.tar.partial"
    # A sequential download prefix from acquisition is also resumable.
    if not target.exists():
        prefix = partial.stat().st_size if partial.exists() else 0
        parts = root / "parts"
        parts.mkdir(exist_ok=True)

        def fetch(bounds):
            start, end = bounds
            path = parts / f"{start}-{end}"
            if path.exists() and path.stat().st_size == end - start + 1:
                return path
            for attempt in range(3):
                try:
                    with httpx.stream(
                        "GET",
                        URL,
                        headers={"Range": f"bytes={start}-{end}"},
                        follow_redirects=True,
                        timeout=120,
                    ) as response:
                        response.raise_for_status()
                        expected = f"bytes {start}-{end}/{SIZE}"
                        if (
                            response.status_code != 206
                            or response.headers.get("content-range") != expected
                        ):
                            raise ValueError("Server did not honor the exact public byte range")
                        with path.with_suffix(".partial").open("wb") as out:
                            for block in response.iter_bytes(1024 * 1024):
                                out.write(block)
                    if path.with_suffix(".partial").stat().st_size != end - start + 1:
                        raise ValueError("Incomplete range")
                    path.with_suffix(".partial").replace(path)
                    return path
                except (httpx.HTTPError, ValueError):
                    if attempt == 2:
                        raise

        ranges = [(a, min(a + 64 * 1024**2, SIZE) - 1) for a in range(prefix, SIZE, 64 * 1024**2)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            for i, future in enumerate(as_completed([executor.submit(fetch, r) for r in ranges])):
                future.result()
                print(f"Downloaded range {i + 1}/{len(ranges)}", flush=True)
        with partial.open("ab") as out:
            for start, end in ranges:
                with (parts / f"{start}-{end}").open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        out.write(block)
            out.flush()
            os.fsync(out.fileno())
        partial.replace(target)
    with target.open("rb") as source:
        digest = hashlib.file_digest(source, "md5").hexdigest()
    if target.stat().st_size != SIZE or digest != MD5:
        raise ValueError("Archive differs from the version published by Dataverse")
    records = []
    # Never extract paths, links, or executable source from a remote tarball.
    with tarfile.open(target, "r|bz2") as archive:
        for member in archive:
            name = Path(member.name).name
            if not member.isfile() or name not in {"autos.json", "howto.json"}:
                continue
            destination = ROOT / name
            with archive.extractfile(member) as source, destination.open("wb") as out:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    out.write(block)
            with destination.open("rb") as source:
                checksum = hashlib.file_digest(source, "sha256").hexdigest()
            records.append({"file": name, "bytes": destination.stat().st_size, "sha256": checksum})
            print("Extracted", name, flush=True)
    (ROOT / "acquisition.json").write_text(
        json.dumps(
            {
                "doi": "10.7910/DVN/L3UWZT",
                "file_id": 3489832,
                "archive_bytes": SIZE,
                "archive_md5": digest,
                "license_record": "CC0-1.0",
                "files": records,
            },
            indent=2,
        )
        + "\n"
    )
    print("Verified acquisition complete", flush=True)


if __name__ == "__main__":
    main()
