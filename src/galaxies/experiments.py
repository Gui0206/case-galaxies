"""Ablations and ranking challengers on development data; final holdout stays sealed."""

import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
from catboost import CatBoostRanker, Pool
from sklearn.linear_model import Ridge

from galaxies.data import ChannelHistory, temporal_split
from galaxies.features import FeatureEncoder
from galaxies.io import utcnow, write_json
from galaxies.metrics import pair_accuracy, regression_metrics
from galaxies.model import train
from galaxies.schemas import PostRecord, TrainConfig


def challengers(records: list[PostRecord], iterations: int = 120) -> dict:
    split = temporal_split(records)["parts"]
    history = ChannelHistory(records)
    encoder = FeatureEncoder(["context", "text", "visual", "rubric"])
    x_train = encoder.fit(split["train"], history)
    x_test = encoder.transform_records(split["test"], history)
    y_train = np.log1p([r.target for r in split["train"]])
    linear = Ridge(alpha=10).fit(encoder.standardized(x_train), y_train)
    linear_metrics = regression_metrics(split["test"], linear.predict(encoder.standardized(x_test)))
    groups = defaultdict(list)
    for i, record in enumerate(split["train"]):
        groups[(record.channel_id, int(record.published_at.timestamp() // (14 * 86400)))].append(i)
    ids, group_ids = [], []
    for group, indices in enumerate(groups.values()):
        if len(indices) >= 2:
            ids.extend(indices)
            group_ids.extend([group] * len(indices))
    if len(ids) < 30:
        ranking = {"status": "insufficient_pairs"}
    else:
        ranker = CatBoostRanker(
            iterations=iterations,
            depth=4,
            loss_function="YetiRank",
            verbose=False,
            allow_writing_files=False,
            random_seed=42,
            thread_count=2,
        )
        ranker.fit(Pool(x_train[ids], label=y_train[ids], group_id=group_ids))
        scores = ranker.predict(x_test)
        ranking = {
            "status": "evaluated",
            "pair_accuracy": pair_accuracy(split["test"], scores),
            "note": "Ranking scores have no count or probability interpretation.",
        }
    return {"ridge": linear_metrics, "catboost_ranker": ranking}


def run_suite(
    records: list[PostRecord], model_root: Path, report_root: Path, iterations: int = 120
) -> dict:
    initial = temporal_split(records)
    final_start = initial["cutoffs"][-1]
    from datetime import datetime

    cutoff = datetime.fromisoformat(final_start)
    development = [r for r in records if r.observed_at < cutoff]
    protected = [r.creative_id for r in records if r.published_at >= cutoff]
    rows = []
    for name, blocks in [
        ("context", ["context"]),
        ("context_text", ["context", "text"]),
        ("context_text_visual", ["context", "text", "visual"]),
        ("full_multimodal", ["context", "text", "visual", "rubric"]),
    ]:
        result = train(
            development, model_root, TrainConfig(iterations=iterations, feature_blocks=blocks)
        )
        rows.append(
            {
                "experiment": name,
                "model_id": result["model_id"],
                "development_evaluation": result["metrics"],
                "config": result["config"],
                "by_sector": result["by_sector"],
            }
        )
    for sector in sorted({r.sector for r in development}):
        subset = [r for r in development if r.sector == sector]
        try:
            result = train(subset, model_root, TrainConfig(iterations=iterations))
            rows.append(
                {
                    "experiment": f"sector_{sector}",
                    "model_id": result["model_id"],
                    "development_evaluation": result["metrics"],
                    "config": result["config"],
                }
            )
        except ValueError as exc:
            rows.append({"experiment": f"sector_{sector}", "skipped": str(exc)})
    channels = sorted({r.channel_id for r in development})
    held_out = channels[::5]
    cold = train(
        development, model_root, TrainConfig(iterations=iterations, held_out_channels=held_out)
    )
    rows.append(
        {
            "experiment": "cold_start",
            "model_id": cold["model_id"],
            "development_evaluation": cold["metrics"],
            "by_generalization": cold["by_generalization"],
            "config": cold["config"],
            "held_out_channels": held_out,
        }
    )
    output = {
        "suite_id": uuid.uuid4().hex,
        "created_at": utcnow().isoformat(),
        "evidence_level": "synthetic_demo" if records[0].synthetic else "development_only",
        "protocol": "Final 15% time window excluded from all suite experiments; no automatic promotion.",
        "protected_final_ids": protected,
        "development_rows": len(development),
        "environment": cold["environment"],
        "iterations": iterations,
        "experiments": rows,
        "challengers": challengers(development, iterations),
    }
    write_json(report_root / f"suite-{output['suite_id']}.json", output)
    return output
