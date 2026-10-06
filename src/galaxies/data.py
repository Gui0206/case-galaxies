"""Dataset contracts, temporal partitions and point-in-time channel history."""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from galaxies.io import write_json
from galaxies.schemas import PostRecord


def load_records(path: Path) -> list[PostRecord]:
    if path.suffix == ".parquet":
        rows = pd.read_parquet(path).to_dict("records")
        for row in rows:
            for field in ("features", "visual_embedding", "evidence"):
                if isinstance(row.get(field), str):
                    row[field] = json.loads(row[field])
    elif path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    elif path.suffix == ".json":
        rows = json.loads(path.read_text())
    else:
        raise ValueError("Use JSONL, JSON ou Parquet no contrato PostRecord.")
    if not rows:
        raise ValueError("Base vazia.")
    return [PostRecord.model_validate(row) for row in rows]


def save_records(path: Path, records: list[PostRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [record.model_dump(mode="json") for record in records]
    if path.suffix == ".parquet":
        for row in rows:
            for name in ("features", "visual_embedding", "evidence"):
                row[name] = json.dumps(row[name], ensure_ascii=False)
        pd.DataFrame(rows).to_parquet(path, index=False)
    elif path.suffix == ".jsonl":
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    else:
        write_json(path, rows)


def audit(records: list[PostRecord]) -> dict:
    ids = [r.creative_id for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("creative_id duplicado na base.")
    profiles = {(r.metric, r.horizon_days) for r in records}
    if len(profiles) != 1:
        raise ValueError("Treine um modelo por métrica e horizonte.")
    if len({r.synthetic for r in records}) != 1:
        raise ValueError("Dados sintéticos e reais não podem ser misturados.")
    hashes = Counter(r.content_hash for r in records)
    channels = Counter(r.channel_id for r in records)
    return {
        "rows": len(records), "channels": len(channels),
        "sectors": dict(Counter(r.sector for r in records)),
        "metric": records[0].metric, "horizon_days": records[0].horizon_days,
        "synthetic": records[0].synthetic,
        "period_start": min(r.published_at for r in records).isoformat(),
        "period_end": max(r.published_at for r in records).isoformat(),
        "duplicate_content_rows": sum(count - 1 for count in hashes.values()),
        "unknown_distribution_fraction": sum(r.distribution == "unknown" for r in records) / len(records),
        "largest_channel_fraction": max(channels.values()) / len(records),
        "missing_subscribers": sum(r.subscribers is None for r in records),
        "sources": sorted({r.source for r in records}),
        "usage_bases": sorted({r.usage_basis for r in records}),
        "warnings": ["Dados sintéticos: resultados não medem desempenho no mundo real."]
        if records[0].synthetic else [],
    }


def temporal_split(records: list[PostRecord], held_out: list[str] | None = None) -> dict:
    """55/15/15/15 by publication time; purge labels maturing across each boundary.

    Exact duplicates are kept only once. Campaign groups appearing in earlier partitions
    are removed from later ones. Held-out channels appear only in the final time window.
    """
    audit(records)
    ordered = sorted(records, key=lambda r: (r.published_at, r.creative_id))
    seen_hashes = set()
    unique = []
    for record in ordered:
        if record.content_hash not in seen_hashes:
            unique.append(record)
            seen_hashes.add(record.content_hash)
    dates = sorted({r.published_at for r in unique})
    if len(dates) < 20:
        raise ValueError("São necessárias pelo menos 20 datas distintas de publicação.")
    cuts = [dates[int(len(dates) * fraction)] for fraction in (0.55, 0.70, 0.85)]
    parts = {name: [] for name in ("train", "validation", "calibration", "test")}
    excluded = set(held_out or [])
    discarded = 0
    groups: dict[str, str] = {}
    for record in unique:
        index = sum(record.published_at >= cut for cut in cuts)
        name = list(parts)[index]
        if (index < 3 and record.observed_at >= cuts[index]) or (
            record.channel_id in excluded and name != "test"
        ):
            discarded += 1
            continue
        group = record.content_group or record.content_hash
        if group in groups and groups[group] != name:
            discarded += 1
            continue
        groups[group] = name
        parts[name].append(record)
    return {
        "parts": parts, "cutoffs": [cut.isoformat() for cut in cuts],
        "discarded": discarded, "exact_duplicates_removed": len(ordered) - len(unique),
    }


class ChannelHistory:
    """Query only outcomes observed before the decision; no final-count shortcuts."""

    def __init__(self, records: list[PostRecord]):
        self.records = records
        self.by_channel: dict[str, list[PostRecord]] = {}
        for record in records:
            self.by_channel.setdefault(record.channel_id, []).append(record)

    def get(self, channel_id: str, sector: str, as_of: datetime) -> dict:
        eligible = [
            r for r in self.by_channel.get(channel_id, [])
            if r.observed_at < as_of and r.published_at < as_of
        ]
        eligible.sort(key=lambda r: r.published_at)
        recent = eligible[-20:]
        peer = [r.target for r in self.records if r.sector == sector and r.observed_at < as_of]
        peer_median = float(np.median(peer)) if peer else 1.0
        targets = [r.target for r in recent]
        median = float(np.median(targets)) if targets else peer_median
        # Partial pooling, explicit and reproducible, regularizes small channels.
        weight = len(recent) / (len(recent) + 5)
        pooled = weight * median + (1 - weight) * peer_median
        return {
            "history_count": len(eligible), "channel_median": median,
            "channel_mean": float(np.mean(targets)) if targets else peer_median,
            "pooled_median": pooled, "sector_median": peer_median,
            "history_log_std": float(np.std(np.log1p(targets))) if targets else 0.0,
        }


def assemble_observations(creative_rows: list[dict], observations: list[dict],
                          metric: str, horizon_days: int) -> tuple[list[PostRecord], list[str]]:
    """Join a pre-publication manifest to mature snapshots, never fabricate a past label."""
    from datetime import timedelta

    result, pending = [], []
    for creative in creative_rows:
        published = datetime.fromisoformat(creative["published_at"].replace("Z", "+00:00"))
        maturity = published + timedelta(days=horizon_days)
        candidates = []
        for observation in observations:
            when = datetime.fromisoformat(observation["observed_at"].replace("Z", "+00:00"))
            if observation["creative_id"] == creative["creative_id"] and (
                maturity <= when <= maturity + timedelta(hours=24)
            ) and observation.get(metric) is not None:
                candidates.append((when, observation))
        if not candidates:
            pending.append(creative["creative_id"])
            continue
        when, observation = min(candidates, key=lambda pair: pair[0])
        result.append(PostRecord.model_validate({
            **creative, "metric": metric, "horizon_days": horizon_days,
            "observed_at": when, "target": observation[metric],
        }))
    return result, pending
