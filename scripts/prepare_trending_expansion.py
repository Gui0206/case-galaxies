"""Reproduce the disjoint country cohort before any model evaluation."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from acquire_modern_sources import acquire

from galaxies.io import read_json, sha256_file, write_json
from galaxies.reach_study import epoch, history_features
from galaxies.transfer_study import build_cohort

ROOT = Path(".runtime/public/trending-expansion")
SOURCE = Path(".runtime/public/modern-trending")


def main(stage):
    ROOT.mkdir(exist_ok=True, parents=True)
    if stage == "acquire":
        with ThreadPoolExecutor(3) as executor:
            files = list(
                executor.map(lambda region: acquire(region, ROOT), ["IN", "JP", "KR", "RU"])
            )
        write_json(
            ROOT / "acquisition.json",
            {"dataset": "rsrishav/youtube-trending-video-dataset", "version": 1346, "files": files},
        )
        return
    if (ROOT / "cohort.parquet").exists():
        raise SystemExit(
            "Preserve the existing cohort and evidence; prepare only in a fresh workspace"
        )
    seen = set()
    for item in read_json(SOURCE / "acquisition.json")["files"]:
        path = SOURCE / item["file"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError("Old source checksum differs")
        seen.update(pd.read_csv(path, usecols=["video_id"]).video_id)
    pieces = []
    for item in read_json(ROOT / "acquisition.json")["files"]:
        path = ROOT / item["file"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError("New source checksum differs")
        data = pd.read_csv(
            path,
            usecols=[
                "video_id",
                "title",
                "publishedAt",
                "channelId",
                "categoryId",
                "trending_date",
                "view_count",
            ],
        )
        data["region"] = item["region"]
        pieces.append(data[~data.video_id.isin(seen)])
    cohort, audit = build_cohort(pd.concat(pieces, ignore_index=True))
    cohort.to_parquet(ROOT / "cohort.parquet", index=False)
    enriched, _ = history_features(cohort)
    warm = enriched[enriched.history_count >= 3]
    audit.update(
        {
            "prior_observed_ids_excluded": len(seen),
            "warm_videos": len(warm),
            "warm_channels": int(warm.channel_id.nunique()),
            "year_coverage": pd.to_datetime(warm.published_at, unit="s", utc=True)
            .dt.year.value_counts()
            .to_dict(),
            "cohort_sha256": sha256_file(ROOT / "cohort.parquet"),
            "performance_measured": False,
        }
    )
    for cut in ["2022-01-01", "2022-07-01", "2023-01-01"]:
        w = warm[warm.published_at >= epoch(cut)]
        audit["since_" + cut] = {"videos": len(w), "channels": int(w.channel_id.nunique())}
    write_json(ROOT / "audit.json", audit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["acquire", "prepare"])
    main(parser.parse_args().stage)
