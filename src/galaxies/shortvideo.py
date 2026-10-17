"""Tsinghua's public full-video research sample, with an explicit retention target.

First observation in a volunteer panel is NOT publication. This dataset must never
be converted to a pre-publication PostRecord by inventing an upload timestamp.
"""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from galaxies.io import sha256_file, utcnow, write_json

SOURCE = "https://github.com/tsinghua-fib-lab/ShortVideo_dataset"
BASE = "https://fi.ee.tsinghua.edu.cn/datasets/short-video-dataset/"
# Public download credentials explicitly published by the dataset authors.
PUBLIC_AUTH = ("videodata", "ShortVideo@10000")
METADATA_SHA256 = "96cadb70829e853b8d7797bc566928eceeb670f19a8a0b92dbfa58b32372bf87"
EVENT_COLUMNS = ["pid", "author_id", "root_id", "exposed_time", "duration", "watch_time", "title"]


def download_metadata(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "interaction_sampled.csv"
    if not path.exists():
        _download(BASE + path.name, path)
    if sha256_file(path) != METADATA_SHA256:
        raise ValueError("A versão pública mudou; audite antes de criar outro estudo.")
    return path


def _download(url, path):
    partial = path.with_suffix(path.suffix + ".partial")
    with httpx.Client(auth=PUBLIC_AUTH, follow_redirects=True, timeout=90) as client:
        with client.stream("GET", url) as response, partial.open("wb") as handle:
            response.raise_for_status()
            for chunk in response.iter_bytes(1024 * 1024):
                handle.write(chunk)
    partial.replace(path)


def load_events(root):
    path = root / "interaction_sampled.csv"
    if sha256_file(path) != METADATA_SHA256:
        raise ValueError("Hash dos dados não corresponde à versão auditada.")
    events = pd.read_csv(path, usecols=EVENT_COLUMNS).sort_values("exposed_time")
    for col in ("duration", "watch_time", "exposed_time"):
        if not np.isfinite(events[col]).all():
            raise ValueError(f"Observação inválida: {col}")
    if (events.duration <= 0).any() or (events.watch_time < 0).any():
        raise ValueError("Duração inválida ou tempo assistido negativo.")
    if events.groupby("pid").author_id.nunique().gt(1).any():
        raise ValueError("Vídeo associado a mais de um criador.")
    # Repeat watches are capped at one full duration, not called engagement counts.
    events["retention"] = np.minimum(events.watch_time / events.duration, 1.0)
    # A watch outcome is only known after that part of the viewing has finished.
    events["outcome_available"] = events.exposed_time + np.minimum(
        events.watch_time, events.duration
    )
    return events


def make_cohort(events):
    first = events.drop_duplicates("pid").set_index("pid")
    end = int(events.exposed_time.max())
    indexed = events.join(first.exposed_time.rename("first_seen"), on="pid")
    window = indexed[indexed.exposed_time < indexed.first_seen + 86400]
    targets = window.groupby("pid").agg(
        target=("retention", "mean"),
        exposures=("retention", "size"),
        last_outcome=("outcome_available", "max"),
    )
    rows = first[["author_id", "root_id", "exposed_time", "duration", "title"]].join(targets)
    rows = rows.rename(columns={"exposed_time": "first_seen", "title": "caption"})
    rows["available_at"] = np.maximum(rows.first_seen + 86400, rows.last_outcome)
    rows = rows[
        (rows.exposures >= 10) & rows.duration.between(10, 120) & (rows.available_at <= end)
    ]
    contexts = temporal_context(events, rows)
    rows = rows.join(contexts)
    # The question concerns an existing channel history, specified before labels are inspected.
    rows = rows[rows.channel_prior_events >= 10].copy()
    rows["creative_id"] = rows.index.astype(str)
    rows["channel_id"] = rows.author_id.astype(str)
    rows["date"] = (
        pd.to_datetime(rows.first_seen, unit="s", utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.strftime("%Y-%m-%d")
    )
    rows = rows.sort_values(["first_seen", "creative_id"]).reset_index(drop=True)
    return rows, {
        "source": SOURCE,
        "metadata_sha256": METADATA_SHA256,
        "events": len(events),
        "videos_in_logs": int(events.pid.nunique()),
        "creators_in_logs": int(events.author_id.nunique()),
        "eligible_videos": len(rows),
        "eligible_creators": int(rows.channel_id.nunique()),
        "target": "mean min(watch_time / duration, 1) in [first panel exposure, +24 hours)",
        "eligibility": "10+ panel exposures in 24h, duration 10-120s, 10+ prior creator events, complete 24h within logs",
        "decision_time": "first observation in volunteer panel, NOT publication",
        "history_availability": "exposure time + min(watch time, video duration); strictly before decision",
        "population": "sampled platform recommendations; known creators only; not all platform impressions",
        "excluded_inputs": [
            "current video engagement",
            "future creator events",
            "follower snapshot",
            "viewer demographics",
        ],
        "publication_validation": False,
        "collection_end_conservative": end,
    }


def temporal_context(events, rows):
    """Histories of other videos available strictly before the first observed exposure."""

    def cumulative(group, queries):
        group = group.sort_values("outcome_available")
        times = group.outcome_available.to_numpy()
        values = group.retention.to_numpy()
        positions = np.searchsorted(times, queries, side="left")
        sums = np.r_[0.0, np.cumsum(values)]
        square = np.r_[0.0, np.cumsum(values**2)]
        return positions, sums[positions], square[positions]

    output = pd.DataFrame(index=rows.index)
    for prefix, key in (("channel", "author_id"), ("category", "root_id")):
        grouped = {k: g for k, g in events.groupby(key, sort=False)}
        for value, query in rows.groupby(key, sort=False):
            n, total, sq = cumulative(grouped[value], query.first_seen.to_numpy())
            output.loc[query.index, f"{prefix}_prior_events"] = n
            output.loc[query.index, f"{prefix}_prior_mean"] = total / np.maximum(n, 1)
            output.loc[query.index, f"{prefix}_prior_std"] = np.sqrt(
                np.maximum(0, sq / np.maximum(n, 1) - (total / np.maximum(n, 1)) ** 2)
            )
    n, total, _ = cumulative(events, rows.first_seen.to_numpy())
    global_mean = np.where(n > 0, total / np.maximum(n, 1), 0.5)
    category = np.where(output.category_prior_events > 0, output.category_prior_mean, global_mean)
    # Fixed 20-event shrinkage, declared before any model comparison.
    weight = output.channel_prior_events / (output.channel_prior_events + 20)
    channel = np.where(output.channel_prior_events > 0, output.channel_prior_mean, category)
    output["history_raw"] = channel
    output["history_shrunk"] = weight * channel + (1 - weight) * category
    output["global_prior_mean"] = global_mean
    return output


def create_acquisition_plan(root):
    path = root / "acquisition.json"
    if path.exists():
        return json.loads(path.read_text())
    events = load_events(root)
    frame, audit = make_cohort(events)
    plan = {
        "created_at": utcnow().isoformat(),
        "audit": audit,
        "ids": frame.creative_id.tolist(),
        "cohort_sha256": hashlib.sha256(frame.to_json(orient="records").encode()).hexdigest(),
        "download_policy": "All eligible videos, maximum three simultaneous downloads as requested by the source. No outcome-based selection.",
        "purpose": "Full-content extraction and temporal retention comparison; no publication forecast claim.",
    }
    write_json(path, plan)
    # Targets remain local; no test metrics are inspected during preparation.
    frame.to_parquet(root / "cohort.parquet", index=False)
    return plan


def download_videos(root, ids):
    folder = root / "videos"
    folder.mkdir(exist_ok=True)
    failures = []

    def fetch(video):
        if not str(video).isdigit():
            raise ValueError("ID de arquivo inválido.")
        path = folder / f"{video}.mp4"
        if not path.exists():
            for attempt in range(3):
                try:
                    _download(BASE + f"raw_file/{video}.mp4", path)
                    break
                except (httpx.HTTPError, OSError):
                    if attempt == 2:
                        raise
                    time.sleep(2**attempt)
        return {
            "creative_id": str(video),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    manifest = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks = {pool.submit(fetch, video): video for video in ids}
        for task in as_completed(tasks):
            try:
                manifest.append(task.result())
            except (httpx.HTTPError, OSError) as exc:
                failures.append({"creative_id": tasks[task], "error": type(exc).__name__})
            if (len(manifest) + len(failures)) % 10 == 0:
                print(
                    f"Vídeos completos: {len(manifest)}/{len(ids)}; falhas: {len(failures)}",
                    flush=True,
                )
                write_json(root / "download.json", {"files": manifest, "failures": failures})
    result = {"files": manifest, "failures": failures}
    write_json(root / "download.json", result)
    return result
