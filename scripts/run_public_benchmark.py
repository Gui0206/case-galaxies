"""Versioned public data -> audited research -> a single frozen final evaluation."""

import argparse
import json
from pathlib import Path

from galaxies.io import sha256_file, write_json
from galaxies.public_benchmark import (
    SOURCES,
    benchmark_plan,
    develop,
    download_public,
    evaluate_final,
    load_kenjee,
    load_trending,
    prepare_public,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("operation", choices=["download", "audit", "develop", "final"])
parser.add_argument("dataset", choices=list(SOURCES))
parser.add_argument("--root", type=Path, default=Path(".runtime/public"))
parser.add_argument("--runs", type=Path, default=Path(".runtime/reports/public"))
args = parser.parse_args()
if args.operation == "download":
    print(json.dumps(download_public(args.root, args.dataset), indent=2))
else:
    source = args.root / args.dataset
    load = {"kenjee": load_kenjee, "trending": load_trending}[args.dataset]
    frame, source_audit = load(source)
    frame, audit = prepare_public(frame)
    audit["source_audit"] = source_audit
    provenance = {
        "source": f"https://www.kaggle.com/datasets/{SOURCES[args.dataset]['ref']}",
        "version": SOURCES[args.dataset]["version"],
        "files": {name: sha256_file(source / name) for name in SOURCES[args.dataset]["files"]},
    }
    folder = args.runs / args.dataset
    if args.operation == "audit":
        print(json.dumps({"provenance": provenance, "audit": audit}, indent=2))
    elif args.operation == "develop":
        plan = benchmark_plan(frame, audit, provenance)
        report = develop(frame, folder, plan)
        print(json.dumps(report["selection"], indent=2))
    else:
        report = evaluate_final(frame, folder)
        write_json(folder / "provenance.json", provenance)
        print(json.dumps(report, indent=2))
