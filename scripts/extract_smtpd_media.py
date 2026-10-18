"""Extract complete short videos while acquisition progresses; preserve failures."""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from galaxies.deep_media import DeepMediaAnalyzer
from galaxies.io import sha256_file, utcnow, write_json
from galaxies.retention_benchmark import validate_media_record


def extract(root, wait=False):
    plan = json.loads((root / "short-media-plan.json").read_text())
    frame = pd.read_parquet(root / "cohort.parquet").set_index("creative_id")
    features = root / "features"
    analyzer = DeepMediaAnalyzer(Path(".runtime/weights"), features, speech_backend="mlx")
    index_path = features / "index.json"
    index = (
        json.loads(index_path.read_text())
        if index_path.exists()
        else {
            "extractor_fingerprint": analyzer.fingerprint,
            "plan_sha256": sha256_file(root / "short-media-plan.json"),
            "rows": {},
            "failures": {},
        }
    )
    if index["extractor_fingerprint"] != analyzer.fingerprint:
        raise ValueError("Extractor changed; preserve previous features")
    index["extraction_complete"] = False
    write_json(index_path, index)
    while True:
        acquisition = root / "media-acquisition.json"
        acquired = json.loads(acquisition.read_text()) if acquisition.exists() else {}
        ready = [
            v
            for v in plan["ids"]
            if v in acquired.get("rows", {})
            and v not in index["rows"]
            and v not in index["failures"]
        ]
        for video in ready:
            path = root / "videos" / f"{video}.mp4"
            try:
                if sha256_file(path) != acquired["rows"][video]["sha256"]:
                    raise ValueError("Acquired media hash changed")
                result = analyzer.analyze(path, str(frame.loc[video, "title"]))
                validate_media_record(result)
                duration = float(frame.loc[video, "duration"])
                if abs(result["duration_s"] - duration) > max(1, duration * 0.02):
                    raise ValueError("Duration differs from official metadata")
                target = features / f"video-{video}.json"
                write_json(target, result)
                index["rows"][video] = {
                    "file": target.name,
                    "sha256": sha256_file(target),
                    "content_hash": result["content_hash"],
                    "at": utcnow(),
                }
            except Exception as exc:
                index["failures"][video] = f"{type(exc).__name__}: {exc}"
            write_json(index_path, index)
            print({"extracted": len(index["rows"]), "failed": len(index["failures"])}, flush=True)
        if (
            not wait
            or acquired.get("complete")
            or acquired.get("blocked")
            or acquired.get("stopped")
        ):
            break
        time.sleep(5)
    index["extraction_complete"] = True
    index["acquisition_complete"] = bool(acquired.get("complete"))
    write_json(index_path, index)
    return {k: len(index[k]) for k in ("rows", "failures")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".runtime/public/smtpd"))
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    print(extract(args.root, args.wait))
