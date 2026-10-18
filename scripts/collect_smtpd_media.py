"""Acquire declared public full-video IDs, sequentially, without bypassing access blocks.

The original streams are merged without trimming. Failures remain in the manifest;
no outcome value influences availability or download order. Secrets and signed URLs
are never persisted. Run extraction separately so the GPU is not held by downloads.
"""

import argparse
import importlib.metadata
import json
import re
import shutil
import time
from pathlib import Path

import yt_dlp

from galaxies.io import sha256_file, utcnow, write_json


class RedactedLog:
    def debug(self, message):
        pass

    def warning(self, message):
        pass

    def error(self, message):
        pass


def collect(root, node, delay_seconds=5):
    if delay_seconds < 3:
        raise ValueError("The registered plan requires at least three seconds between videos")
    plan_path = root / "short-media-plan.json"
    plan = json.loads(plan_path.read_text())
    folder = root / "videos"
    folder.mkdir(exist_ok=True)
    path = root / "media-acquisition.json"
    identity = {
        "plan_sha256": sha256_file(plan_path),
        "yt_dlp": importlib.metadata.version("yt-dlp"),
        "ejs": importlib.metadata.version("yt-dlp-ejs"),
        "node": str(node.resolve()),
        "format": "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480]",
        "trimmed": False,
    }
    manifest = (
        json.loads(path.read_text())
        if path.exists()
        else {"identity": identity, "rows": {}, "failures": {}, "created_at": utcnow()}
    )
    if manifest["identity"] != identity:
        raise ValueError("Acquisition identity changed")
    if manifest.get("blocked"):
        raise ValueError("Previous access block: explicit cooldown review required before resuming")
    manifest.setdefault("throttle_history", []).append(
        {"at": utcnow(), "delay_seconds": delay_seconds}
    )
    for number, video in enumerate(plan["ids"]):
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video):
            raise ValueError("Invalid video ID")
        if video in manifest["rows"] or video in manifest["failures"]:
            continue
        if shutil.disk_usage(root).free < 3 * 1024**3:
            manifest["stopped"] = "disk space below 3 GiB"
            write_json(path, manifest)
            raise RuntimeError(manifest["stopped"])
        target = folder / f"{video}.mp4"
        options = {
            "quiet": True,
            "logger": RedactedLog(),
            "noplaylist": True,
            "retries": 0,
            "fragment_retries": 0,
            "extractor_retries": 0,
            "socket_timeout": 25,
            "js_runtimes": {"node": {"path": str(node.resolve())}},
            "format": identity["format"],
            "merge_output_format": "mp4",
            "outtmpl": str(folder / "%(id)s.%(ext)s"),
        }
        try:
            with yt_dlp.YoutubeDL(options) as downloader:
                info = downloader.extract_info(
                    f"https://www.youtube.com/watch?v={video}", download=True
                )
            if not target.is_file():
                raise ValueError("Expected merged full video missing")
            manifest["rows"][video] = {
                "file": target.name,
                "sha256": sha256_file(target),
                "bytes": target.stat().st_size,
                "duration": info.get("duration"),
                "acquired_at": utcnow(),
            }
        except Exception as exc:
            message = re.sub(r"https?://\S+", "[redacted URL]", str(exc))[:600]
            blocked = bool(re.search(r"\b(403|429)\b|not a bot|sign in|rate.limit", message, re.I))
            manifest["failures"][video] = {
                "kind": type(exc).__name__,
                "message": message,
                "blocked": blocked,
                "at": utcnow(),
            }
            if blocked:
                manifest["blocked"] = {"id": video, "at": utcnow()}
        manifest["processed"] = number + 1
        manifest["complete"] = number + 1 == len(plan["ids"])
        write_json(path, manifest)
        print(
            json.dumps(
                {
                    "processed": number + 1,
                    "planned": len(plan["ids"]),
                    "downloaded": len(manifest["rows"]),
                    "failures": len(manifest["failures"]),
                    "blocked": bool(manifest.get("blocked")),
                }
            ),
            flush=True,
        )
        if manifest.get("blocked"):
            break
        time.sleep(delay_seconds)
    return {k: len(manifest[k]) for k in ("rows", "failures")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".runtime/public/smtpd"))
    parser.add_argument("--node", type=Path, required=True, help="Node >=22, as required by yt-dlp")
    parser.add_argument("--delay-seconds", type=float, default=5)
    args = parser.parse_args()
    print(collect(args.root, args.node, args.delay_seconds))
