"""Strict recovery of fixed-age outcomes from SMTPD's published popularity scores.

The source omits dates and standardizes numerical metadata across all records.
Only independently recovered publication metadata is accepted. No standardized
source numeric field, subscriber count, or early target-video outcome is a feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from galaxies.reach_study import DAY

CATEGORIES = {"Autos  Vehicles": "2", "Howto  Style": "26"}


def reconstruct_views(popularity, day=7):
    """Invert the paper's p=log2(views/day+1), requiring integer-compatible counts."""
    if day not in range(1, 31):
        raise ValueError("SMTPD day must be between 1 and 30")
    score = np.asarray(popularity, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        values = day * np.expm1(score * np.log(2))
    valid = np.isfinite(values) & (values >= 0) & (values < 2**53)
    if not valid.all():
        raise ValueError("Nonfinite, negative or non-integer-compatible source outcome")
    valid &= np.abs(values - np.round(values)) <= np.maximum(1e-4, abs(values) * 1e-12)
    if not valid.all():
        raise ValueError("Nonfinite, negative or non-integer-compatible source outcome")
    return np.round(values).astype(np.int64)


def recover_cohort(source, metadata):
    """One video per row, public exact dates, stable channel IDs, conservative maturity."""
    if metadata.video_id.duplicated().any():
        raise ValueError("Recovered metadata repeats a video ID")
    raw = source[source.category.isin(CATEGORIES)].copy()
    repeated = raw.video_id.duplicated(keep=False)
    counts = {"sector_rows": len(raw), "duplicate_rows_excluded": int(repeated.sum())}
    raw = raw[~repeated]
    # Explicit allowlist prevents accidental use of present-day engagement counters.
    allowed = [
        "video_id",
        "published_at",
        "upload_date",
        "channel_id",
        "duration",
        "is_live",
        "playability",
    ]
    joined = raw[["video_id", "title", "category", "Day 7"]].merge(
        metadata.reindex(columns=allowed), on="video_id", validate="one_to_one"
    )
    published = pd.to_datetime(joined.published_at, utc=True, errors="coerce")
    duration = pd.to_numeric(joined.duration, errors="coerce")
    valid = (
        joined.video_id.astype(str).str.fullmatch(r"[A-Za-z0-9_-]{11}")
        & joined.channel_id.fillna("").astype(str).str.fullmatch(r"UC[A-Za-z0-9_-]{22}")
        & published.notna()
        & duration.gt(0)
        & np.isfinite(duration)
        & joined.playability.eq("OK")
        & joined.is_live.eq(False)
        & joined.title.fillna("").str.strip().ne("")
    )
    counts["metadata_unusable"] = int((~valid).sum())
    f = joined[valid].copy()
    seconds = published[valid].astype("int64").to_numpy() / 1e9
    target = reconstruct_views(f["Day 7"])
    result = (
        pd.DataFrame(
            {
                "creative_id": f.video_id.to_numpy(),
                "channel_id": f.channel_id.to_numpy(),
                "published_at": seconds,
                "observed_at": seconds + 7 * DAY,
                "available_at": seconds + 8 * DAY,
                "title": f.title.str.strip().to_numpy(),
                "category": f.category.map(CATEGORIES).to_numpy(),
                "duration": duration[valid].to_numpy(dtype=float),
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
    counts["eligible_videos"] = len(result)
    counts["channels"] = int(result.channel_id.nunique())
    counts["duration_at_most_180s"] = int(result.duration.le(180).sum())
    return result, counts
