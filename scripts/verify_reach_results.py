"""Recompute the released reach metrics independently from individual forecasts.

This does not train models, select candidates, or open a new holdout. The vectorized
history count uses a separate implementation from the experiment's event queue.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from galaxies.io import read_json, sha256_file, utcnow, write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".runtime/reports/reach"
DATA = ROOT / ".runtime/public/beyond-views"


def main():
    report = read_json(RUN / "final.json")
    plan = read_json(RUN / "plan.json")
    forecasts = pd.read_csv(RUN / "test_predictions.csv", dtype={"category": str})
    source = (
        pd.read_parquet(DATA / "cohort.parquet")
        .sort_values(["published_at", "creative_id"])
        .reset_index(drop=True)
    )
    development = pd.read_csv(RUN / "development_predictions.csv")
    assert sha256_file(RUN / "test_predictions.csv") == report["test_predictions_sha256"]
    assert sha256_file(DATA / "cohort.parquet") == plan["audit"]["cohort_sha256"]
    assert forecasts.creative_id.is_unique
    assert not set(forecasts.creative_id) & set(development.creative_id)
    assert (forecasts.published_at >= pd.Timestamp("2016-08-20", tz="UTC").timestamp()).all()
    aligned = source.set_index("creative_id").loc[forecasts.creative_id]
    np.testing.assert_array_equal(forecasts.target, aligned.target)
    np.testing.assert_allclose(forecasts.target_log, np.log1p(forecasts.target), atol=1e-12)
    counts = np.zeros(len(source), dtype=int)
    for _, group in source[~source.reserved_channel].groupby("channel_id", sort=False):
        counts[group.index] = np.searchsorted(
            group.available_at.to_numpy(), group.published_at.to_numpy(), side="left"
        )
    lookup = pd.Series(counts, index=source.creative_id)
    np.testing.assert_array_equal(forecasts.history_count, lookup.loc[forecasts.creative_id])
    part = forecasts[(~forecasts.reserved_channel) & (forecasts.history_count >= 3)]
    assert len(part) == report["sections"]["primary"]["metrics"]["content"]["videos"]
    verified = {}
    for label, expected in report["sections"]["primary"]["metrics"].items():
        predicted = part[label].to_numpy()
        assert np.isfinite(predicted).all() and (predicted >= 0).all()
        errors = np.abs(np.log1p(part.target.to_numpy()) - predicted)
        by_channel = pd.Series(errors).groupby(part.channel_id.to_numpy()).mean()
        radius = part[f"{label}_radius"].to_numpy()
        values = {
            "macro_mae_log": float(by_channel.mean()),
            "mae_log": float(errors.mean()),
            "mae_views": float(np.abs(part.target.to_numpy() - np.expm1(predicted)).mean()),
            "within_factor_two": float((errors <= np.log(2)).mean()),
            "coverage90": float((errors <= radius).mean()),
        }
        for key, value in values.items():
            np.testing.assert_allclose(value, expected[key], rtol=1e-10, atol=1e-10)
        verified[label] = values
    # Enumerate actual pairs independently of the experiment's integer-slot decoder.
    ranking_rows = part.reset_index(drop=True).copy()
    ranking_rows["week"] = (
        (ranking_rows.published_at - pd.Timestamp("2016-07-01", tz="UTC").timestamp()) // 604800
    ).astype(int)
    channel_scores, pair_count = [], 0
    for channel, group in ranking_rows.groupby("channel_id"):
        all_pairs = []
        for _, week in group.groupby("week"):
            ids = week.index.to_numpy()
            left, right = np.triu_indices(len(ids), k=1)
            all_pairs.extend(zip(ids[left], ids[right], strict=True))
        if not all_pairs:
            continue
        seed = int(hashlib.sha256(channel.encode()).hexdigest()[:16], 16)
        chosen = np.random.default_rng(seed).choice(
            len(all_pairs), min(500, len(all_pairs)), replace=False
        )
        pairs = np.asarray(all_pairs)[chosen]
        left, right = pairs[:, 0], pairs[:, 1]
        target = ranking_rows.target.to_numpy()
        actual = np.sign(target[left] - target[right])
        valid = actual != 0
        if not valid.any():
            continue
        pair_count += int(valid.sum())
        values = {}
        for label in verified:
            prediction = ranking_rows[label].to_numpy()
            direction = np.sign(prediction[left] - prediction[right])
            values[label] = float(np.where(direction == 0, 0.5, direction == actual)[valid].mean())
        channel_scores.append(values)
    scores = pd.DataFrame(channel_scores)
    expected_ranking = report["sections"]["primary"]["ranking"]
    assert pair_count == expected_ranking["pairs"]
    assert len(scores) == expected_ranking["channels"]
    for label in verified:
        np.testing.assert_allclose(
            scores[label].mean(), expected_ranking["macro_accuracy"][label], atol=1e-12
        )
    generator = np.random.default_rng(20261018)
    draws = {name: [] for name in ("history", "context")}
    for _ in range(50):
        sample = generator.integers(0, len(scores), (100, len(scores)))
        for name in draws:
            difference = (scores.content - scores[name]).to_numpy()
            draws[name].extend(difference[sample].mean(axis=1))
    for name, samples in draws.items():
        np.testing.assert_allclose(
            np.quantile(samples, [0.025, 0.975]),
            expected_ranking[f"content_vs_{name}_ci95"],
            atol=1e-12,
        )
    result = {
        "verified_at": utcnow(),
        "prediction_sha256": sha256_file(RUN / "test_predictions.csv"),
        "checks": [
            "Unique IDs; no overlap with development; final publication dates only",
            "Targets and source cohort hashes match",
            "History counts match independent per-channel binary searches",
            "Primary macro/row errors, raw view errors, factor-two rates and coverage match",
            "Independent pair enumeration reproduces ranking scores and both bootstrap intervals",
        ],
        "primary_videos": len(part),
        "primary_channels": int(part.channel_id.nunique()),
        "recomputed": verified,
        "ranking_pairs": pair_count,
        "ranking_channels": len(scores),
        "ranking_accuracy": scores.mean().to_dict(),
    }
    write_json(RUN / "verification.json", result)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
