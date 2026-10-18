"""Prepublication SMTPD study with strict event-time histories and a reserved 2024 test.

All transforms are fitted within each training window. Within-channel regression
removes channel means using training labels only; at prediction time it restores
only already-mature historical outcomes. Final evaluation is deliberately separate.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from galaxies.io import sha256_file, utcnow, write_json
from galaxies.reach_study import (
    BASELINES,
    DAY,
    TEXT_STATS,
    ReachEncoder,
    epoch,
    history_features,
    load_semantic,
    macro_error,
    weights,
)

ROOT = Path(".runtime/public/smtpd")
RUN = Path(".runtime/reports/smtpd")
FOLDS = [("2023-11-17", "2023-11-28"), ("2023-12-18", "2023-12-23")]
TRAIN_END = "2023-12-23"
CALIBRATION = ("2023-12-23", "2023-12-28")
TEST_START = "2024-01-05"
ALPHAS = [0.25, 0.5, 1.0]
HISTORY = BASELINES + ["recent_median_30", "recent_median_90"]
CONTEXT_RULES = ["duration_05", "duration_10", "format_median"]


def fingerprint():
    return hashlib.sha256(
        b"".join(
            Path(__file__).with_name(n).read_bytes()
            for n in ["smtpd_study.py", "reach_study.py", "smtpd.py"]
        )
    ).hexdigest()


def weighted_median(values, weight):
    order = np.argsort(values, kind="stable")
    v, w = np.asarray(values)[order], np.asarray(weight)[order]
    if not len(v) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("Weighted median needs finite positive total weight")
    return float(v[np.searchsorted(np.cumsum(w), w.sum() / 2, side="left")])


def enrich(frame, neighbors):
    """Duration/recency matching uses the same mature source outcomes as other references."""
    frame = frame.copy()
    values = {name: [] for name in HISTORY[-2:] + CONTEXT_RULES}
    y, duration, date = (
        frame.target_log.to_numpy(),
        frame.duration.to_numpy(),
        frame.published_at.to_numpy(),
    )
    for i, prior in enumerate(neighbors):
        fallback = frame.median_20.iloc[i]
        if not prior:
            for name in values:
                values[name].append(fallback)
            continue
        ids = np.array(prior[-20:])
        for days in [30, 90]:
            weight = np.exp(-np.log(2) * (date[i] - date[ids]) / DAY / days)
            values[f"recent_median_{days}"].append(weighted_median(y[ids], weight))
        distance = np.abs(np.log1p(duration[ids]) - np.log1p(duration[i]))
        for tag, bandwidth in [("05", 0.5), ("10", 1.0)]:
            weight = np.exp(-distance / bandwidth)
            values[f"duration_{tag}"].append(weighted_median(y[ids], weight))
        same = (duration[ids] <= 180) == (duration[i] <= 180)
        # Two examples minimum prevents calling one unusually successful upload "usual".
        values["format_median"].append(
            float(np.median(y[ids][same])) if same.sum() >= 2 else fallback
        )
    for name, value in values.items():
        frame[name] = value
    return frame


def data(development=True):
    full = pd.read_parquet(ROOT / "cohort.parquet")
    semantic = load_semantic(ROOT, full)
    keep = (
        full.published_at < epoch(CALIBRATION[1]) if development else np.ones(len(full), dtype=bool)
    )
    frame, neighbors = history_features(full[keep].reset_index(drop=True))
    return enrich(frame, neighbors), neighbors, semantic[keep]


def partitions(frame, start, stop=None, final_train=False):
    if final_train:
        mature = (frame.available_at < epoch(TEST_START)) & (frame.published_at < epoch(TRAIN_END))
    else:
        mature = frame.available_at < epoch(start)
    all_train = np.flatnonzero(mature)
    warm_train = all_train[frame.history_count.to_numpy()[all_train] >= 1]
    valid = (frame.published_at >= epoch(start)) & (frame.history_count >= 3)
    if stop:
        valid &= frame.published_at < epoch(stop)
    return all_train, warm_train, np.flatnonzero(valid)


def matrices(frame, neighbors, semantic, train):
    encoder = ReachEncoder()
    values = encoder.fit_transform(frame, train, neighbors, semantic)
    extra = frame[CONTEXT_RULES + HISTORY[-2:]].to_numpy()
    values = {
        name: np.column_stack([values[name], extra])
        for name in ["context", "retrieval", "semantic_retrieval"]
    }
    # Pure prepublication attributes for centered within-channel regression. No outcome-derived
    # context enters these differences; historical outcomes enter only the prediction anchor.
    duration = np.log1p(frame.duration.to_numpy())
    pure = np.column_stack(
        [
            duration,
            frame.duration.le(180),
            *[np.maximum(duration - np.log1p(knot), 0) for knot in [30, 60, 180, 600]],
            frame[["hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]].to_numpy(),
        ]
    )
    title = encoder.vectorizer.transform(frame.title)
    values["within_context"] = pure
    values["within_text"] = np.column_stack([pure, frame[TEXT_STATS], encoder.svd.transform(title)])
    values["within_semantic"] = np.column_stack(
        [pure, frame[TEXT_STATS], encoder.semantic_pca.transform(semantic)]
    )
    values["within_semantic_full"] = np.column_stack([pure, frame[TEXT_STATS], semantic])
    prefix = np.column_stack([pure, frame[TEXT_STATS]])
    prefix = StandardScaler().fit(prefix[train]).transform(prefix)
    values["within_lexical"] = sparse.hstack([sparse.csr_matrix(prefix), title], format="csr")
    return encoder, values


def within_prediction(x, frame, train, neighbors, predict, alpha, folder=None, name="within"):
    """Fixed-effects ridge; no target channel ID or future outcome is a deployment feature."""
    scaler = None if sparse.issparse(x) else StandardScaler().fit(x[train])
    normalized = x if scaler is None else scaler.transform(x)
    frequency = frame.iloc[train].channel_id.value_counts()
    train = train[frame.iloc[train].channel_id.map(frequency).to_numpy() >= 2]
    if len(train) < 10:
        raise ValueError("Too few repeated training channels")
    groups = frame.iloc[train].channel_id.to_numpy()
    if sparse.issparse(normalized):
        _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
        assignment = sparse.csr_matrix(
            (np.ones(len(train)), (np.arange(len(train)), inverse)), shape=(len(train), len(counts))
        )
        means = sparse.diags(1 / counts) @ assignment.T @ normalized[train]
        centered_x = normalized[train] - assignment @ means
    else:
        train_x = pd.DataFrame(normalized[train])
        centered_x = train_x.to_numpy() - train_x.groupby(groups).transform("mean").to_numpy()
    target = frame.target_log.to_numpy()
    train_y = pd.Series(target[train])
    centered_y = train_y.to_numpy() - train_y.groupby(groups).transform("mean").to_numpy()
    model = Ridge(alpha=alpha, fit_intercept=False, solver="lsqr").fit(
        centered_x, centered_y, sample_weight=weights(frame.iloc[train])
    )
    scores = model.predict(normalized)
    output = []
    for i in predict:
        prior = neighbors[i][-20:]
        if prior:
            output.append(float(np.mean(target[prior])) + scores[i] - scores[prior].mean())
        else:
            output.append(frame.category_prior.iloc[i])
    if folder:
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            folder / f"{name}.npz",
            coef=model.coef_,
            mean=scaler.mean_ if scaler is not None else np.zeros(x.shape[1]),
            scale=scaler.scale_ if scaler is not None else np.ones(x.shape[1]),
        )
    return np.maximum(0, output)


def candidates():
    result = {}
    for bundle in ["context", "retrieval", "semantic_retrieval"]:
        for depth in [3, 5]:
            for anchor in ["median_20", "logmean_20"]:
                name = f"{bundle}_d{depth}_{anchor}"
                result[name] = {"kind": "tree", "bundle": bundle, "depth": depth, "anchor": anchor}
    for bundle in ["within_context", "within_text", "within_semantic"]:
        for penalty in [10, 100, 1000]:
            result[f"{bundle}_{penalty}"] = {
                "kind": "within",
                "bundle": bundle,
                "alpha": penalty,
                "anchor": "logmean_20",
            }
    for bundle, penalties in [
        ("within_semantic_full", [100, 1000, 10000]),
        ("within_lexical", [1, 10, 100]),
    ]:
        for penalty in penalties:
            result[f"{bundle}_{penalty}"] = {
                "kind": "within",
                "bundle": bundle,
                "alpha": penalty,
                "anchor": "logmean_20",
            }
    return result


def predict_candidate(name, x, frame, all_train, warm_train, neighbors, predict, folder=None):
    spec = candidates()[name]
    if spec["kind"] == "within":
        return within_prediction(
            x[spec["bundle"]], frame, all_train, neighbors, predict, spec["alpha"], folder, name
        )
    anchor = frame[spec["anchor"]].to_numpy()
    model = CatBoostRegressor(
        iterations=500,
        depth=spec["depth"],
        learning_rate=0.035,
        loss_function="MAE",
        l2_leaf_reg=30,
        random_seed=42,
        thread_count=4,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        x[spec["bundle"]][warm_train],
        frame.target_log.to_numpy()[warm_train] - anchor[warm_train],
        sample_weight=weights(frame.iloc[warm_train]),
    )
    if folder:
        folder.mkdir(parents=True, exist_ok=True)
        model.save_model(str(folder / f"{name}.cbm"))
    return np.maximum(0, anchor[predict] + model.predict(x[spec["bundle"]][predict]))


def inputs():
    return {str(p): sha256_file(p) for p in [ROOT / "cohort.parquet", ROOT / "semantic.json"]}


def develop(run=RUN):
    if (run / "base-selection.json").exists():
        raise ValueError("Preserve previous development; use another run directory")
    frame, neighbors, semantic = data(development=True)
    specs = candidates()
    write_json(
        run / "base-plan.json",
        {
            "created_at": utcnow(),
            "code_sha256": fingerprint(),
            "inputs": inputs(),
            "folds": FOLDS,
            "train_end": TRAIN_END,
            "calibration": CALIBRATION,
            "test_start": TEST_START,
            "candidates": specs,
            "shrinkage": ALPHAS,
            "history": HISTORY,
            "context_rules": CONTEXT_RULES,
            "primary": ">=3 mature same-channel results; equal weight per channel",
            "success": ">=500 videos/50 channels; >=5% lower log MAE and positive paired channel-bootstrap CI against BOTH selected history and context",
            "test_opened": False,
        },
    )
    results = []
    for start, stop in FOLDS:
        all_train, warm_train, valid = partitions(frame, start, stop)
        _, x = matrices(frame, neighbors, semantic, all_train)
        part = frame.iloc[valid][
            ["creative_id", "channel_id", "category", "published_at", "target_log", "duration"]
        ].copy()
        part["fold"] = start
        for name in HISTORY + CONTEXT_RULES:
            part[name] = frame[name].to_numpy()[valid]
        print(
            {
                "fold": start,
                "all_training": len(all_train),
                "warm_training": len(warm_train),
                "validation": len(valid),
            },
            flush=True,
        )
        for name, spec in specs.items():
            prediction = predict_candidate(name, x, frame, all_train, warm_train, neighbors, valid)
            anchor = frame[spec["anchor"]].to_numpy()[valid]
            for shrink in ALPHAS:
                part[f"{name}@{shrink:g}"] = anchor + shrink * (prediction - anchor)
        results.append(part)
    predictions = pd.concat(results, ignore_index=True)
    predictions.to_csv(run / "base-development.csv", index=False)
    names = HISTORY + CONTEXT_RULES + [f"{n}@{a:g}" for n in specs for a in ALPHAS]
    errors = {n: macro_error(predictions, predictions[n].to_numpy()) for n in names}
    context = (
        HISTORY
        + CONTEXT_RULES
        + [n for n in names if n.startswith(("context_", "within_context_"))]
    )
    content = [n for n in names if "@" in n and n not in context]

    def choose(group):
        return min(group, key=lambda n: (errors[n], n))

    selected = {
        "history": choose(HISTORY),
        "context": choose(context),
        "content": choose(content),
        "errors": errors,
        "created_at": utcnow(),
        "plan_sha256": sha256_file(run / "base-plan.json"),
        "predictions_sha256": sha256_file(run / "base-development.csv"),
    }
    write_json(run / "base-selection.json", selected)
    return {
        role: {"model": selected[role], "error": errors[selected[role]]}
        for role in ["history", "context", "content"]
    }


def selected_prediction(name, x, frame, all_train, warm_train, neighbors, predict, folder=None):
    if name in HISTORY + CONTEXT_RULES:
        return frame[name].to_numpy()[predict]
    model, alpha = name.split("@")
    raw = predict_candidate(model, x, frame, all_train, warm_train, neighbors, predict, folder)
    anchor = frame[candidates()[model]["anchor"]].to_numpy()[predict]
    return anchor + float(alpha) * (raw - anchor)
