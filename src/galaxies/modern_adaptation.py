"""Learn on recent development data; evaluate on disjoint, later country cohorts.

The seven previously evaluated countries are explicitly development data now.
Confirmation uses never-evaluated IDs from IN/JP/KR/RU after July 2022. Selection
and calibration end before that date, including outcome maturity.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from galaxies.io import read_json, sha256_file, utcnow, write_json
from galaxies.reach_study import (
    BASELINES,
    ReachEncoder,
    conformal_radius,
    epoch,
    history_features,
    indices,
    load_semantic,
    macro_error,
    paired_gain,
    prediction_metrics,
    ranking,
)
from galaxies.transfer_study import fingerprint as dependency_fingerprint
from galaxies.transfer_study import fit_regressor, model_matrices

SOURCE = Path(".runtime/public/modern-trending")
EXTERNAL = Path(".runtime/public/trending-expansion")
RUN = Path(".runtime/reports/modern-adaptation")
FOLDS = [("2021-07-01", "2022-01-01"), ("2022-01-01", "2022-05-01")]
FIT_BEFORE = "2022-05-01"
CALIBRATION = ("2022-05-01", "2022-06-21")
TEST_START = "2022-07-01"
MODELS = [f"{b}_mae{d}" for b in ("context", "lexical", "semantic") for d in (3, 5)] + [
    "ridge_100",
    "ridge_1000",
]
SHRINKAGE = [0.0, 0.1, 0.25, 0.5, 1.0]


def fingerprint():
    return hashlib.sha256(
        Path(__file__).read_bytes() + dependency_fingerprint().encode()
    ).hexdigest()


def inputs():
    return {
        str(p): sha256_file(p)
        for root in [SOURCE, EXTERNAL]
        for p in [root / "cohort.parquet", root / "semantic.json", root / "audit.json"]
    }


def source_data():
    full = pd.read_parquet(SOURCE / "cohort.parquet")
    keep = full.published_at < epoch(CALIBRATION[1])
    embedding = load_semantic(SOURCE, full)[keep]
    frame, neighbors = history_features(full[keep].reset_index(drop=True))
    return frame, neighbors, embedding


def training_rows(frame, before):
    rows = indices(frame, before, train=True, warm=False)
    return rows[frame.iloc[rows].history_count.to_numpy() >= 1]


def blended_name(model, alpha):
    return f"{model}@{alpha:g}"


def select(predictions):
    names = list(BASELINES) + [blended_name(m, a) for m in MODELS for a in SHRINKAGE]
    errors = {n: macro_error(predictions, predictions[n].to_numpy()) for n in names}

    def choose(candidates):
        return min(candidates, key=lambda n: (errors[n], n))

    return {
        "history": choose(BASELINES),
        "context": choose([n for n in names if n.startswith("context_")]),
        "content": choose([n for n in names if "@" in n and not n.startswith("context_")]),
        "errors": errors,
    }


def develop(run=RUN):
    if run.exists():
        raise ValueError("Preserve previous development; choose a new run directory")
    frame, neighbors, semantic = source_data()
    plan = {
        "created_at": utcnow(),
        "code_sha256": fingerprint(),
        "inputs": inputs(),
        "folds": FOLDS,
        "fit_before": FIT_BEFORE,
        "calibration": CALIBRATION,
        "test_start": TEST_START,
        "models": MODELS,
        "shrinkage": SHRINKAGE,
        "training": "at least one mature historical result; evaluation >=3",
        "selection": "minimum macro log MAE; shrinkage multiplies correction to logmean20; independent context selection",
        "success": ">=500 videos/50 channels; >=5% error reduction and positive paired channel bootstrap CI against history AND context",
        "test_ids_exclude_every_id_previously_observed": True,
    }
    write_json(run / "plan.json", plan)
    saved = []
    for start, stop in FOLDS:
        train, valid = training_rows(frame, start), indices(frame, start, stop)
        encoder = ReachEncoder()
        mat = model_matrices(encoder, encoder.fit_transform(frame, train, neighbors, semantic))
        part = frame.iloc[valid][
            ["creative_id", "channel_id", "category", "published_at", "target_log"]
        ].copy()
        print("Fold", start, "train", len(train), "validation", len(valid), flush=True)
        for name in BASELINES:
            part[name] = frame.iloc[valid][name].to_numpy()
        baseline = frame.logmean_20.to_numpy()[valid]
        for name in MODELS:
            raw = fit_regressor(name, mat, frame, train, valid)
            for alpha in SHRINKAGE:
                part[blended_name(name, alpha)] = baseline + alpha * (raw - baseline)
        saved.append(part)
    data = pd.concat(saved, ignore_index=True)
    data.to_csv(run / "development_predictions.csv", index=False)
    selection = select(data)
    selection.update(
        {
            "created_at": utcnow(),
            "plan_sha256": sha256_file(run / "plan.json"),
            "predictions_sha256": sha256_file(run / "development_predictions.csv"),
        }
    )
    write_json(run / "selection.json", selection)
    return selection


def verify(run):
    plan = read_json(run / "plan.json")
    selection = read_json(run / "selection.json")
    if plan["code_sha256"] != fingerprint() or plan["inputs"] != inputs():
        raise ValueError("Frozen inputs/code changed")
    if selection["plan_sha256"] != sha256_file(run / "plan.json") or selection[
        "predictions_sha256"
    ] != sha256_file(run / "development_predictions.csv"):
        raise ValueError("Frozen selection evidence changed")
    return selection


def final(run=RUN):
    selection = verify(run)
    import json

    with (run / "evaluation-opened.json").open("x") as stream:
        json.dump(
            {
                "opened_at": str(utcnow()),
                "selection_sha256": sha256_file(run / "selection.json"),
                "code_sha256": fingerprint(),
            },
            stream,
            indent=2,
        )
    frame, neighbors, semantic = source_data()
    train = training_rows(frame, FIT_BEFORE)
    cal = indices(frame, *CALIBRATION)
    external, en = history_features(pd.read_parquet(EXTERNAL / "cohort.parquet"))
    es = load_semantic(EXTERNAL, external)
    test = np.flatnonzero(
        (external.published_at >= epoch(TEST_START)) & (external.history_count >= 3)
    )
    encoder = ReachEncoder()
    source_m = model_matrices(encoder, encoder.fit_transform(frame, train, neighbors, semantic))
    external_m = model_matrices(encoder, encoder.transform(external, en, es))
    matrices = {
        n: sparse.vstack([source_m[n], external_m[n]], format="csr")
        if sparse.issparse(source_m[n])
        else np.vstack([source_m[n], external_m[n]])
        for n in source_m
    }
    combined = pd.concat([frame, external], ignore_index=True)
    positions = np.r_[cal, len(frame) + test]
    folder = run / "models"
    encoder.save(folder)
    output = {}
    roles = {}
    for role in ["context", "content"]:
        model, alpha = selection[role].split("@")
        alpha = float(alpha)
        if model not in output:
            output[model] = fit_regressor(model, matrices, combined, train, positions, folder)
        baseline = combined.logmean_20.to_numpy()[positions]
        roles[role] = baseline + alpha * (output[model] - baseline)
    roles["history"] = combined[selection["history"]].to_numpy()[positions]
    primary = external.iloc[test].copy()
    predictions = {}
    radii = {}
    export = primary[
        [
            "creative_id",
            "channel_id",
            "category",
            "published_at",
            "target",
            "target_log",
            "history_count",
        ]
    ].copy()
    for role, values in roles.items():
        radii[role] = conformal_radius(
            np.abs(frame.target_log.to_numpy()[cal] - values[: len(cal)])
        )
        predictions[role] = values[len(cal) :]
        export[role] = predictions[role]
    gains = {
        r: paired_gain(primary, predictions["content"], predictions[r])
        for r in ["history", "context"]
    }
    enough = len(primary) >= 500 and primary.channel_id.nunique() >= 50
    result = {
        "created_at": utcnow(),
        "selection": selection,
        "train_videos": len(train),
        "calibration_videos": len(cal),
        "primary_videos": len(primary),
        "primary_channels": int(primary.channel_id.nunique()),
        "primary_metrics": {
            r: prediction_metrics(primary, v, radii[r]) for r, v in predictions.items()
        },
        "content_gains": gains,
        "ranking_diagnostic": ranking(primary, predictions),
        "confirmed_error_gain": bool(
            enough
            and all(g["relative_reduction_pct"] >= 5 and g["ci95"][0] > 0 for g in gains.values())
        ),
        "case_solved": False,
        "production_eligible": False,
        "limitations": [
            "Selected Trending population and incomplete channel histories",
            "Small recent training population; countries differ between development and confirmation",
            "First observed title may have changed after publication",
            "No full-video modalities; no claim about their effectiveness",
            "Ranking is diagnostic, not a separate optimized confirmation",
        ],
    }
    export.to_csv(run / "test_predictions.csv", index=False)
    result["predictions_sha256"] = sha256_file(run / "test_predictions.csv")
    write_json(run / "final.json", result)
    return result
