"""Complete SMTPD publication metadata with an authorized YouTube Data API key.

Run with: uv run --env-file .env python scripts/recover_smtpd_api.py
Only 94 requests are planned for the 4,671 candidate IDs. No search requests,
engagement counts, media downloads, or authentication workarounds are used.
"""

import argparse
import os
from pathlib import Path

import pandas as pd

from galaxies.io import read_json, sha256_file, utcnow, write_json
from galaxies.smtpd import recover_cohort
from galaxies.youtube import YouTubeCollector

ROOT = Path(".runtime/public/smtpd")


def main(all_sector=False):
    key = os.environ.get("YOUTUBE_API_KEY", "")
    if not key:
        raise SystemExit("Defina YOUTUBE_API_KEY localmente. Não cole a chave no chat ou nos logs.")
    client = YouTubeCollector(key)
    plan = read_json(ROOT / "metadata-plan.json")
    source = ROOT / "basic_view_pn.csv"
    if sha256_file(source) != plan["source_sha256"]:
        raise SystemExit("Source changed after metadata plan")
    f = pd.read_csv(source, usecols=["video_id", "title", "user_url", "category", "Day 7"])
    f = f[f.category.isin(["Howto  Style", "Autos  Vehicles"])]
    f = f[~f.video_id.duplicated(keep=False)]
    if not all_sector:
        f = f[
            f.user_url.fillna("").str.startswith(
                ("http://www.youtube.com/", "https://www.youtube.com/")
            )
        ]
        f = f[f.groupby("user_url").video_id.transform("size") >= 4]
        if len(f) != plan["videos"]:
            raise SystemExit("Candidate population differs from registered metadata audit")
    else:
        write_json(
            ROOT / "metadata-plan-all-sector.json",
            {
                "created_at": utcnow(),
                "source_sha256": plan["source_sha256"],
                "videos": len(f),
                "population": "All sector videos, excluding duplicate IDs; no selection on target values",
                "reason": "Official batched API removes page-collection bottleneck; include sparse channels to improve training coverage",
            },
        )
    f = f.sort_values("video_id")
    suffix = "-all-sector" if all_sector else ""
    cache = ROOT / "api-metadata"
    cache.mkdir(exist_ok=True)
    pending = [v for v in f.video_id if not (cache / f"{v}.json").exists()]
    print(
        f"{len(pending)} metadata records pending; at most {(len(pending) + 49) // 50} API calls",
        flush=True,
    )
    for offset in range(0, len(pending), 50):
        try:
            records = client.publication_metadata(pending[offset : offset + 50])
        except ValueError as error:
            raise SystemExit(str(error)) from None
        for record in records:
            write_json(cache / f"{record['video_id']}.json", record)
        print(f"{min(offset + 50, len(pending))}/{len(pending)} records fetched", flush=True)
    meta = pd.DataFrame([read_json(cache / f"{v}.json") for v in f.video_id])
    metadata_path = ROOT / f"api-metadata{suffix}.parquet"
    cohort_path = ROOT / f"recovered-cohort{suffix}.parquet"
    meta.to_parquet(metadata_path, index=False)
    cohort, audit = recover_cohort(f, meta)
    cohort.to_parquet(cohort_path, index=False)
    audit.update(
        {
            "created_at": utcnow(),
            "metadata_sha256": sha256_file(metadata_path),
            "cohort_sha256": sha256_file(cohort_path),
            "source_sha256": plan["source_sha256"],
            "no_predictive_comparison_performed": True,
        }
    )
    write_json(ROOT / f"recovered-audit{suffix}.json", audit)
    print(audit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-sector", action="store_true")
    main(parser.parse_args().all_sector)
