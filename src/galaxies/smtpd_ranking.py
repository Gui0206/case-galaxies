"""A bounded, development-only check of direct same-channel pairwise learning.

Regression optimizes numerical error, whereas the case also asks which creative
will perform better. PairLogit learns that ordering directly, using mature videos
from the same channel and week, with equal total pair weight per channel.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

from galaxies.io import read_json, sha256_file, utcnow, write_json
from galaxies.reach_study import DAY, epoch
from galaxies.smtpd_study import FOLDS, data, matrices, partitions
from galaxies.transfer_study import ranking_scores

RUN = Path(".runtime/reports/smtpd-ranking")
BASE = Path(".runtime/reports/smtpd-v2")
MINIMUM_ADOPTION_GAIN = 0.03


def candidates():
    return {
        f"pair_{bundle}_d{depth}_l2{penalty}": {
            "bundle": bundle,
            "depth": depth,
            "penalty": penalty,
        }
        for bundle in ["context", "semantic_retrieval"]
        for depth in [3, 5]
        for penalty in [30, 100]
    }


def training_pairs(frame, train):
    chosen = frame.iloc[train].copy()
    chosen["position"] = train
    chosen["week"] = ((chosen.published_at - epoch("2016-07-01")) // (7 * DAY)).astype(int)
    chosen["group"] = chosen.channel_id + ":" + chosen.week.astype(str)
    chosen = chosen.sort_values(["group", "published_at", "creative_id"]).reset_index(drop=True)
    pairs, channels = [], []
    for _, group in chosen.groupby("group", sort=False):
        positions = group.index.to_numpy()
        left, right = np.triu_indices(len(positions), k=1)
        y = group.target_log.to_numpy()
        for a, b in zip(left, right, strict=True):
            if y[a] != y[b]:
                pairs.append(
                    (positions[a], positions[b]) if y[a] > y[b] else (positions[b], positions[a])
                )
                channels.append(group.channel_id.iloc[0])
    if not pairs:
        raise ValueError("No mature same-channel training pairs")
    frequencies = pd.Series(channels).value_counts()
    weight = np.array([1 / frequencies[c] for c in channels])
    # Unit average pair weight keeps regularization comparable across training folds.
    weight *= len(weight) / weight.sum()
    return chosen, np.asarray(pairs, dtype=int), weight


def predict_candidate(name, x, frame, train, predict, folder=None):
    spec = candidates()[name]
    chosen, pairs, weights = training_pairs(frame, train)
    positions = chosen.position.to_numpy()
    anchor = frame.logmean_20.to_numpy()
    pool = Pool(
        x[spec["bundle"]][positions],
        label=chosen.target_log.to_numpy(),
        group_id=chosen.group.to_numpy(),
        pairs=pairs,
        pairs_weight=weights,
        baseline=anchor[positions],
    )
    model = CatBoostRanker(
        iterations=400,
        depth=spec["depth"],
        learning_rate=0.035,
        loss_function="PairLogit",
        l2_leaf_reg=spec["penalty"],
        random_seed=42,
        thread_count=4,
        verbose=False,
        allow_writing_files=False,
    ).fit(pool)
    if folder:
        folder.mkdir(parents=True, exist_ok=True)
        model.save_model(str(folder / f"{name}.cbm"))
    return anchor[predict] + model.predict(x[spec["bundle"]][predict])


def develop(run=RUN):
    if (run / "plan.json").exists():
        raise ValueError("Preserve the previous ranking experiment")
    frame, neighbors, semantic = data(development=True)
    base = pd.read_csv(BASE / "ensemble-development.csv").set_index("creative_id")
    selection = read_json(BASE / "ensemble-selection.json")
    write_json(
        run / "plan.json",
        {
            "created_at": utcnow(),
            "code_sha256": sha256_file(Path(__file__)),
            "base_selection_sha256": sha256_file(BASE / "ensemble-selection.json"),
            "folds": FOLDS,
            "candidates": candidates(),
            "iterations": 400,
            "shrinkage": [0.5, 1.0],
            "training_pairs": "same channel and week, mature before each fold",
            "weights": "equal total pair weight per channel",
            "adoption_rule": "at least 3 percentage points above existing content ranking, and no worse in either validation fold; independently maintain the strongest context control",
            "test_opened": False,
        },
    )
    parts = []
    counts = []
    for start, stop in FOLDS:
        train, _, valid = partitions(frame, start, stop)
        _, x = matrices(frame, neighbors, semantic, train)
        chosen, pairs, _ = training_pairs(frame, train)
        counts.append({"fold": start, "pairs": len(pairs), "channels": chosen.channel_id.nunique()})
        part = frame.iloc[valid][
            ["creative_id", "channel_id", "category", "published_at", "target_log"]
        ].copy()
        part["fold"] = start
        for role in ["history", "context", "content"]:
            part[role] = base.loc[part.creative_id, selection[f"ranking_{role}"]].to_numpy()
        for name in candidates():
            output = predict_candidate(name, x, frame, train, valid)
            for weight in [0.5, 1.0]:
                part[f"{name}@{weight:g}"] = np.maximum(
                    0, frame.logmean_20.to_numpy()[valid] * (1 - weight) + weight * output
                )
        parts.append(part)
        print(counts[-1], flush=True)
    predictions = pd.concat(parts, ignore_index=True)
    names = ["history", "context", "content"] + [n for n in predictions if n.startswith("pair_")]
    scores = ranking_scores(predictions, {n: predictions[n].to_numpy() for n in names})
    folds = {
        start: ranking_scores(part, {n: part[n].to_numpy() for n in names})
        for start, part in predictions.groupby("fold")
    }
    best = max([n for n in names if n.startswith("pair_semantic_")], key=lambda n: (scores[n], n))
    context = max(
        ["context"] + [n for n in names if n.startswith("pair_context_")],
        key=lambda n: (scores[n], n),
    )
    eligible = scores[best] - scores["content"] >= MINIMUM_ADOPTION_GAIN and all(
        f[best] >= f["content"] for f in folds.values()
    )
    predictions.to_csv(run / "development.csv", index=False)
    result = {
        "created_at": utcnow(),
        "candidate": best,
        "context": context,
        "adoption_eligible": bool(eligible),
        "scores": scores,
        "fold_scores": folds,
        "training_counts": counts,
        "plan_sha256": sha256_file(run / "plan.json"),
        "predictions_sha256": sha256_file(run / "development.csv"),
    }
    write_json(run / "selection.json", result)
    return {
        "candidate": best,
        "score": scores[best],
        "previous": scores["content"],
        "eligible": bool(eligible),
    }
