"""Acquire repeated creators from other SMTPD categories for auxiliary training only.

Selection uses source channel URL frequency, never outcomes. Every identity and
publication date must still be verified officially. Only pre-calibration records
are exported for training; no auxiliary 2024 labels enter any modeling artifact.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from galaxies.io import read_json, sha256_file, utcnow, write_json
from galaxies.reach_study import DAY, epoch
from galaxies.smtpd import CATEGORIES, reconstruct_views
from galaxies.youtube import YouTubeCollector

ROOT = Path(".runtime/public/smtpd")
AUX = ROOT / "auxiliary"


def prepare(source, metadata):
    merged = source[["video_id", "title", "category", "Day 7"]].merge(
        metadata, on="video_id", validate="one_to_one"
    )
    date = pd.to_datetime(merged.published_at, utc=True, errors="coerce")
    duration = pd.to_numeric(merged.duration, errors="coerce")
    valid = date.notna() & (date.astype("int64") / 1e9 < epoch("2023-12-23"))
    valid &= (
        merged.channel_id.fillna("").str.fullmatch(r"UC[A-Za-z0-9_-]{22}")
        & merged.playability.eq("OK")
        & merged.is_live.eq(False)
    )
    valid &= duration.gt(0) & np.isfinite(duration) & merged.title.fillna("").str.strip().ne("")
    f = merged[valid].copy()
    published = date[valid].astype("int64").to_numpy() / 1e9
    target = reconstruct_views(f["Day 7"])
    return (
        pd.DataFrame(
            {
                "creative_id": f.video_id.to_numpy(),
                "channel_id": f.channel_id.to_numpy(),
                "published_at": published,
                "observed_at": published + 7 * DAY,
                "available_at": published + 8 * DAY,
                "title": f.title.str.strip().to_numpy(),
                "category": ("aux:" + f.category).to_numpy(),
                "duration": duration[valid].to_numpy(),
                "language": "unknown",
                "definition": "unknown",
                "target": target,
                "target_log": np.log1p(target),
                "reserved_channel": False,
            }
        )
        .sort_values(["published_at", "creative_id"])
        .reset_index(drop=True)
    )


def main():
    source_path = ROOT / "basic_view_pn.csv"
    source = pd.read_csv(
        source_path, usecols=["video_id", "title", "category", "user_url", "Day 7"]
    )
    source = source[~source.video_id.duplicated(keep=False)]
    source = source[
        source.user_url.fillna("").str.startswith(
            ("https://www.youtube.com/", "http://www.youtube.com/")
        )
    ]
    source = source[source.groupby("user_url").video_id.transform("size") >= 3]
    source = source[~source.category.isin(CATEGORIES)]
    source = source[source.video_id.str.fullmatch(r"[A-Za-z0-9_-]{11}")].sort_values("video_id")
    cache = ROOT / "api-metadata"
    pending = [v for v in source.video_id if not (cache / f"{v}.json").exists()]
    plan = {
        "source_sha256": sha256_file(source_path),
        "videos": len(source),
        "ids_sha256": __import__("hashlib").sha256("\n".join(source.video_id).encode()).hexdigest(),
        "selection": "nonsector IDs from source URLs with >=3 records, excluding all repeated IDs; metadata verification mandatory",
        "purpose": "auxiliary training only; exported publication dates strictly before Dec23,2023; no new confirmation population",
    }
    if (AUX / "plan.json").exists() and read_json(AUX / "plan.json") != plan:
        raise ValueError("Auxiliary cohort selection changed")
    write_json(AUX / "plan.json", plan)
    print({"pending": len(pending), "max_api_calls": (len(pending) + 49) // 50}, flush=True)
    client = YouTubeCollector(os.environ.get("YOUTUBE_API_KEY", ""))
    for start in range(0, len(pending), 50):
        try:
            records = client.publication_metadata(pending[start : start + 50])
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        for row in records:
            write_json(cache / f"{row['video_id']}.json", row)
        if start % 1000 == 0:
            print({"fetched": min(start + 50, len(pending)), "planned": len(pending)}, flush=True)
    metadata = pd.DataFrame([read_json(cache / f"{v}.json") for v in source.video_id])
    cohort = prepare(source, metadata)
    cohort.to_parquet(AUX / "cohort.parquet", index=False)
    write_json(
        AUX / "audit.json",
        {
            "created_at": utcnow(),
            "plan": plan,
            "eligible_videos": len(cohort),
            "channels": int(cohort.channel_id.nunique()),
            "cohort_sha256": sha256_file(AUX / "cohort.parquet"),
            "all_labels_exported_before": "2023-12-23",
            "no_performance_comparison": True,
        },
    )
    print({"prepared": len(cohort), "channels": int(cohort.channel_id.nunique())}, flush=True)


if __name__ == "__main__":
    main()
