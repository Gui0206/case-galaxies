"""Regularized within-channel transfer from other SMTPD categories.

The auxiliary source is training-only and predates calibration. Domain weights
control its influence; a zero-transfer control distinguishes transfer from a new
regression parameterization. No target-sector test labels select those weights.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from galaxies.io import read_json, sha256_file, utcnow, write_json
from galaxies.reach_study import DAY, load_semantic, macro_error
from galaxies.smtpd_study import FOLDS, ROOT, data, matrices, partitions

AUX = ROOT / "auxiliary"
PENALTIES = [100, 1000, 10000]
FRACTIONS = [0.0, 0.1, 0.3, 0.5]


def pure_features(frame, semantic):
    date = pd.to_datetime(frame.published_at, unit="s", utc=True)
    duration = np.log1p(frame.duration.to_numpy())
    title = frame.title.astype(str)
    return np.column_stack(
        [
            duration,
            frame.duration.le(180),
            *[np.maximum(duration - np.log1p(knot), 0) for knot in [30, 60, 180, 600]],
            np.sin(date.dt.hour * np.pi / 12),
            np.cos(date.dt.hour * np.pi / 12),
            np.sin(date.dt.dayofweek * 2 * np.pi / 7),
            np.cos(date.dt.dayofweek * 2 * np.pi / 7),
            np.log1p(title.str.len()),
            np.log1p(title.str.split().str.len()),
            title.map(lambda s: sum(c.isdigit() for c in s) / len(s)),
            title.str.contains("?", regex=False).astype(float),
            title.str.contains("!", regex=False).astype(float),
            semantic,
        ]
    )


@lru_cache(maxsize=1)
def auxiliary_data():
    frame = pd.read_parquet(AUX / "cohort.parquet")
    semantic = load_semantic(AUX, frame)
    return frame, pure_features(frame, semantic)


def sufficient_statistics(x, frame, rows, monthly=True):
    chosen = frame.iloc[rows].copy()
    group = chosen.channel_id.astype(str)
    if monthly:
        group = group + ":" + (chosen.published_at // (28 * DAY)).astype(int).astype(str)
    groups, inverse, counts = np.unique(group.to_numpy(), return_inverse=True, return_counts=True)
    keep = counts[inverse] >= 2
    selected = rows[keep]
    group = group.to_numpy()[keep]
    if len(selected) < 10:
        raise ValueError("Too few within-channel training comparisons")
    values = pd.DataFrame(x[selected])
    centered = values.to_numpy() - values.groupby(group).transform("mean").to_numpy()
    target = pd.Series(frame.target_log.to_numpy()[selected])
    centered_y = target.to_numpy() - target.groupby(group).transform("mean").to_numpy()
    channels = frame.iloc[selected].channel_id
    weight = channels.map(1 / channels.value_counts()).to_numpy()
    weight /= weight.sum()
    return (
        centered.T @ (weight[:, None] * centered),
        centered.T @ (weight * centered_y),
        len(selected),
    )


def fit_grid(frame, train, neighbors, predict, primary_x, folder=None):
    auxiliary, aux_x = auxiliary_data()
    # Conservative maturity cutoff: never later than the latest accepted primary training
    # outcome. Also auxiliary exports exclude all calibration and 2024 publications.
    cutoff = frame.available_at.to_numpy()[train].max()
    aux_train = np.flatnonzero(auxiliary.available_at.to_numpy() <= cutoff)
    scale = StandardScaler().fit(primary_x[train])
    px, ax = scale.transform(primary_x), scale.transform(aux_x)
    pg, ph, pn = sufficient_statistics(px, frame, train)
    ag, ah, an = sufficient_statistics(ax, auxiliary, aux_train)
    predictions = {}
    coefficients = {}
    for fraction in FRACTIONS:
        gram = ((1 - fraction) * pg + fraction * ag) * pn
        rhs = ((1 - fraction) * ph + fraction * ah) * pn
        for bundle, dimensions in [("context", 10), ("semantic", gram.shape[0])]:
            for penalty in PENALTIES:
                beta = np.zeros(gram.shape[0])
                beta[:dimensions] = np.linalg.solve(
                    gram[:dimensions, :dimensions] + penalty * np.eye(dimensions), rhs[:dimensions]
                )
                scores = px @ beta
                values = []
                for i in predict:
                    prior = neighbors[i][-20:]
                    values.append(
                        float(frame.target_log.iloc[prior].mean())
                        + scores[i]
                        - scores[prior].mean()
                        if prior
                        else frame.category_prior.iloc[i]
                    )
                name = f"aux_{bundle}{fraction:g}_ridge{penalty}"
                predictions[name] = np.maximum(0, values)
                coefficients[name] = beta
    if folder:
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            folder / "auxiliary_models.npz", mean=scale.mean_, scale=scale.scale_, **coefficients
        )
        write_json(
            folder / "auxiliary_models.json",
            {
                "auxiliary_cohort_sha256": sha256_file(AUX / "cohort.parquet"),
                "auxiliary_semantic_sha256": sha256_file(AUX / "semantic.npy"),
                "primary_comparison_rows": pn,
                "auxiliary_comparison_rows": an,
                "maturity_cutoff": float(cutoff),
                "centering": "channel and fixed 28-day bin; domain weights normalized separately",
            },
        )
    return predictions, {
        "primary_comparisons": pn,
        "auxiliary_comparisons": an,
        "auxiliary_mature_videos": len(aux_train),
    }


def develop(
    run=Path(".runtime/reports/smtpd-auxiliary"), base_run=Path(".runtime/reports/smtpd-v2")
):
    if (run / "selection.json").exists():
        raise ValueError("Keep previous auxiliary development")
    frame, neighbors, semantic = data(development=True)
    base = pd.read_csv(base_run / "ensemble-development.csv").set_index("creative_id")
    base_selection = read_json(base_run / "ensemble-selection.json")
    write_json(
        run / "plan.json",
        {
            "created_at": utcnow(),
            "auxiliary_cohort_sha256": sha256_file(AUX / "cohort.parquet"),
            "auxiliary_semantic_manifest_sha256": sha256_file(AUX / "semantic.json"),
            "fractions": FRACTIONS,
            "penalties": PENALTIES,
            "context_blends": [0.5, 1.0],
            "bundles": [
                "context: duration and schedule only",
                "semantic: context, title statistics and multilingual representation",
            ],
            "base_selection_sha256": sha256_file(base_run / "ensemble-selection.json"),
            "folds": FOLDS,
            "future_test_opened": False,
        },
    )
    parts = []
    for start, stop in FOLDS:
        train, _, valid = partitions(frame, start, stop)
        _, x = matrices(frame, neighbors, semantic, train)
        output, counts = fit_grid(frame, train, neighbors, valid, x["within_semantic_full"])
        part = frame.iloc[valid][
            ["creative_id", "channel_id", "category", "published_at", "target_log"]
        ].copy()
        for role in ["history", "context", "content"]:
            part[role] = base.loc[part.creative_id, base_selection[role]].to_numpy()
        for name, values in output.items():
            for weight in [0.5, 1.0]:
                part[f"{name}@{weight:g}"] = (1 - weight) * part.context + weight * values
        parts.append(part)
        print({"fold": start, **counts}, flush=True)
    predictions = pd.concat(parts, ignore_index=True)
    names = ["content", "context"] + [c for c in predictions if c.startswith("aux")]
    errors = {n: macro_error(predictions, predictions[n].to_numpy()) for n in names}
    choice = min(
        ["content"] + [n for n in names if n.startswith("aux_semantic")],
        key=lambda n: (errors[n], n),
    )
    context = min(
        ["context"] + [n for n in names if n.startswith("aux_context")],
        key=lambda n: (errors[n], n),
    )
    predictions.to_csv(run / "development.csv", index=False)
    result = {
        "created_at": utcnow(),
        "content": choice,
        "context": context,
        "errors": errors,
        "plan_sha256": sha256_file(run / "plan.json"),
        "predictions_sha256": sha256_file(run / "development.csv"),
    }
    write_json(run / "selection.json", result)
    return {
        "selected": choice,
        "error": errors[choice],
        "base_error": errors["content"],
        "context": context,
        "context_error": errors[context],
    }
