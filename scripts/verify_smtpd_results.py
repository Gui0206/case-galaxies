"""Recompute released SMTPD results independently, without fitting or selecting models."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from galaxies.io import read_json, sha256_file, utcnow, write_json

RUN = Path(".runtime/reports/smtpd-final")
SOURCE = Path(".runtime/public/smtpd")


def main():
    report = read_json(RUN / "final.json")
    predictions = pd.read_csv(RUN / "test_predictions.csv", dtype={"category": str})
    source = (
        pd.read_parquet(SOURCE / "cohort.parquet")
        .sort_values(["published_at", "creative_id"])
        .reset_index(drop=True)
    )
    development = pd.read_csv(RUN / "development.csv")
    assert predictions.creative_id.is_unique
    assert len(predictions) == 515 and predictions.channel_id.nunique() == 165
    assert sha256_file(RUN / "test_predictions.csv") == report["predictions_sha256"]
    assert not set(predictions.creative_id) & set(development.creative_id)
    assert (predictions.published_at >= pd.Timestamp("2024-01-05", tz="UTC").timestamp()).all()
    aligned = source.set_index("creative_id").loc[predictions.creative_id]
    np.testing.assert_array_equal(predictions.target, aligned.target)
    np.testing.assert_allclose(predictions.target_log, np.log1p(predictions.target), atol=1e-12)
    counts = np.zeros(len(source), dtype=int)
    for _, group in source.groupby("channel_id", sort=False):
        counts[group.index] = np.searchsorted(
            group.available_at.to_numpy(), group.published_at.to_numpy(), side="left"
        )
    lookup = pd.Series(counts, index=source.creative_id)
    np.testing.assert_array_equal(predictions.history_count, lookup.loc[predictions.creative_id])
    assert (predictions.history_count >= 3).all()
    metrics = {}
    for role, expected in report["primary_metrics"].items():
        estimate = predictions[role].to_numpy()
        assert np.isfinite(estimate).all() and (estimate >= 0).all()
        error = np.abs(np.log1p(predictions.target.to_numpy()) - estimate)
        metrics[role] = {
            "macro_mae_log": float(
                pd.Series(error).groupby(predictions.channel_id.to_numpy()).mean().mean()
            ),
            "mae_log": float(error.mean()),
            "mae_views": float(abs(predictions.target.to_numpy() - np.expm1(estimate)).mean()),
            "within_factor_two": float((error <= np.log(2)).mean()),
            "coverage90": float((error <= predictions[f"{role}_radius"]).mean()),
        }
        for name, value in metrics[role].items():
            np.testing.assert_allclose(value, expected[name], rtol=1e-10, atol=1e-10)
    for role in ["history", "context"]:
        differences = (
            (
                abs(predictions.target_log - predictions[role])
                - abs(predictions.target_log - predictions.content)
            )
            .groupby(predictions.channel_id)
            .mean()
            .to_numpy()
        )
        rng = np.random.default_rng(20261017)
        boot = np.concatenate(
            [rng.choice(differences, (100, len(differences))).mean(axis=1) for _ in range(50)]
        )
        np.testing.assert_allclose(
            np.quantile(boot, [0.025, 0.975]), report["content_gains"][role]["ci95"], atol=1e-12
        )
    pairs = predictions.copy()
    pairs["week"] = (
        (pairs.published_at - pd.Timestamp("2016-07-01", tz="UTC").timestamp()) // 604800
    ).astype(int)
    group_scores = []
    pair_count = 0
    for channel, group in pairs.groupby("channel_id"):
        all_pairs = []
        for _, week in group.groupby("week"):
            ids = week.index.to_numpy()
            a, b = np.triu_indices(len(ids), k=1)
            all_pairs.extend(zip(ids[a], ids[b], strict=True))
        if not all_pairs:
            continue
        seed = int(hashlib.sha256(channel.encode()).hexdigest()[:16], 16)
        chosen = np.random.default_rng(seed).choice(
            len(all_pairs), min(500, len(all_pairs)), replace=False
        )
        selected = np.asarray(all_pairs)[chosen]
        left, right = selected[:, 0], selected[:, 1]
        actual = np.sign(pairs.target.to_numpy()[left] - pairs.target.to_numpy()[right])
        valid = actual != 0
        if not valid.any():
            continue
        pair_count += int(valid.sum())
        scores = {}
        for role in ["history", "context", "content"]:
            prediction = pairs[f"ranking_{role}"].to_numpy()
            direction = np.sign(prediction[left] - prediction[right])
            scores[role] = float(np.where(direction == 0, 0.5, direction == actual)[valid].mean())
        group_scores.append(scores)
    scores = pd.DataFrame(group_scores)
    assert pair_count == report["ranking"]["pairs"] and len(scores) == report["ranking"]["channels"]
    for role in ["history", "context", "content"]:
        np.testing.assert_allclose(
            scores[role].mean(), report["ranking"]["macro_accuracy"][role], atol=1e-12
        )
    rng = np.random.default_rng(20261018)
    boot = {r: [] for r in ["history", "context"]}
    for _ in range(50):
        chosen = rng.integers(0, len(scores), (100, len(scores)))
        for role in boot:
            diff = (scores.content - scores[role]).to_numpy()
            boot[role].extend(diff[chosen].mean(axis=1))
    for role in boot:
        np.testing.assert_allclose(
            np.quantile(boot[role], [0.025, 0.975]),
            report["ranking"][f"content_vs_{role}_ci95"],
            atol=1e-12,
        )
    result = {
        "verified_at": utcnow(),
        "prediction_sha256": sha256_file(RUN / "test_predictions.csv"),
        "videos": len(predictions),
        "channels": int(predictions.channel_id.nunique()),
        "metrics_independently_recomputed": metrics,
        "history_maturity_verified": True,
        "development_test_disjoint": True,
        "ranking_pairs_recomputed": pair_count,
        "bootstrap_intervals_recomputed": True,
    }
    write_json(RUN / "verification.json", result)
    print(
        {
            "verified": True,
            "videos": len(predictions),
            "channels": int(predictions.channel_id.nunique()),
            "ranking_pairs": pair_count,
        }
    )


if __name__ == "__main__":
    main()
