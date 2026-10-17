"""Download full videos, extract pinned modalities and run a frozen retention study."""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from galaxies.deep_media import DeepMediaAnalyzer, provision_weights
from galaxies.io import write_json
from galaxies.shortvideo import create_acquisition_plan, download_metadata, download_videos

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "operation", choices=["download", "weights", "extract", "analyze", "develop", "final"]
)
parser.add_argument("video", type=Path, nargs="?")
parser.add_argument("--root", type=Path, default=Path(".runtime/public/shortvideo"))
parser.add_argument("--weights", type=Path, default=Path(".runtime/weights"))
parser.add_argument("--runs", type=Path, default=Path(".runtime/reports/multimodal"))
parser.add_argument("--output", type=Path)
parser.add_argument("--wait-downloads", action="store_true")
parser.add_argument("--speech-backend", choices=["cpu", "mlx"], default="cpu")
args = parser.parse_args()

if args.operation == "weights":
    provision_weights(args.weights, args.speech_backend)
elif args.operation == "download":
    download_metadata(args.root)
    plan = create_acquisition_plan(args.root)
    download_videos(args.root, plan["ids"])
elif args.operation in ("extract", "analyze"):
    analyzer = DeepMediaAnalyzer(
        args.weights, args.root / "features", speech_backend=args.speech_backend
    )
    if args.operation == "analyze":
        if args.video is None:
            parser.error("analyze requer o caminho de um vídeo local")
        result = analyzer.analyze(args.video)
        if args.output:
            write_json(args.output, result)
            print(args.output)
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        frame = pd.read_parquet(args.root / "cohort.parquet")
        pending = {row.creative_id: row for row in frame.itertuples()}
        index_path = args.root / "features/index.json"
        index = {"extractor_fingerprint": analyzer.fingerprint, "rows": {}, "failures": {}}
        if index_path.exists():
            previous = json.loads(index_path.read_text())
            if previous["extractor_fingerprint"] == analyzer.fingerprint:
                index = previous
                pending = {k: v for k, v in pending.items() if k not in index["rows"]}
        index["extraction_complete"] = False
        write_json(index_path, index)
        last_progress = time.monotonic()

        def extract_one(row):
            result = analyzer.analyze(
                args.root / "videos" / f"{row.creative_id}.mp4",
                "" if pd.isna(row.caption) else str(row.caption),
            )
            if abs(result["duration_s"] - row.duration) > max(1, row.duration * 0.05):
                raise ValueError("Duração do arquivo diverge da duração nos logs.")
            return result

        while pending:
            ready = [video for video in pending if (args.root / "videos" / f"{video}.mp4").exists()]
            if not ready:
                if not args.wait_downloads or time.monotonic() - last_progress > 300:
                    for video in pending:
                        index["failures"][video] = "Arquivo de vídeo ainda indisponível."
                    break
                time.sleep(5)
                continue
            with ThreadPoolExecutor(max_workers=3) as pool:
                tasks = {pool.submit(extract_one, pending.pop(video)): video for video in ready}
                for task in as_completed(tasks):
                    video = tasks[task]
                    try:
                        result = task.result()
                        target = args.root / "features" / f"video-{video}.json"
                        write_json(target, result)
                        index["rows"][video] = {
                            "file": target.name,
                            "content_hash": result["content_hash"],
                        }
                        index["failures"].pop(video, None)
                    except Exception as exc:
                        index["failures"][video] = f"{type(exc).__name__}: {exc}"
                        print(f"Falha {video}: {type(exc).__name__}", flush=True)
                    last_progress = time.monotonic()
                    write_json(index_path, index)
                    if (len(index["rows"]) + len(index["failures"])) % 10 == 0:
                        print(
                            f"Extraídos: {len(index['rows'])}/{len(frame)}; falhas: {len(index['failures'])}",
                            flush=True,
                        )
        index["extraction_complete"] = True
        write_json(index_path, index)
        print(
            {
                "completed": len(index["rows"]),
                "failures": len(index["failures"]),
                "extractor": analyzer.fingerprint,
            },
            flush=True,
        )
else:
    from galaxies.retention_benchmark import develop, evaluate_final

    run = {"develop": develop, "final": evaluate_final}[args.operation]
    print(json.dumps(run(args.root, args.runs), indent=2, ensure_ascii=False))
