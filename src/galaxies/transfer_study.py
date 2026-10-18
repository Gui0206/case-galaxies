"""Independent temporal transfer test; never reopens the closed 2016 evaluation.

New models are developed on the old development folds only. Modern popularity
labels are used exclusively after a signed local freeze, for external evaluation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, CatBoostRegressor, Pool
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge

from galaxies.io import read_json, sha256_file, utcnow, write_json
from galaxies.reach_study import (
    BASELINES,
    CONTEXT,
    DAY,
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
    weights,
)
from galaxies.reach_study import (
    ROOT as SOURCE,
)

ROOT = Path(".runtime/public/modern-trending")
RUN = Path(".runtime/reports/transfer")
STOP = "2024-04-01"
REGRESSORS = {
    f"{bundle}_mae{depth}": {"bundle": bundle, "depth": depth}
    for bundle in ("context", "lexical", "semantic", "analogy")
    for depth in (3, 5, 7)
}
REGRESSORS.update({f"ridge_{a}": {"bundle": "linear", "alpha": a} for a in (100, 1000)})
RANKERS = {
    f"rank_{bundle}_{depth}": {"bundle": bundle, "depth": depth}
    for bundle in ("context", "lexical", "semantic", "analogy")
    for depth in (4, 6)
}
RANKERS.update({f"pair_linear_{c}": {"bundle": "linear", "c": c} for c in (0.1, 1.0)})


def build_cohort(snapshots):
    """One video, one first-week observation; source timestamps are only nominal.

    Metadata from the earliest recorded snapshot is retained. It is retrospective,
    not an assertion of prepublication title availability. Country copies never
    multiply examples, and inconsistent identity/publication metadata is rejected.
    """
    f = snapshots.copy()
    f["published"] = pd.to_datetime(f.publishedAt, utc=True, errors="coerce")
    f["observed"] = pd.to_datetime(f.trending_date, utc=True, errors="coerce")
    counts = {"snapshots": len(f), "unique_ids": int(f.video_id.nunique())}
    valid = (
        f.video_id.astype(str).str.fullmatch(r"[A-Za-z0-9_-]{11}")
        & f.channelId.notna()
        & f.published.notna()
        & f.observed.notna()
        & (f.observed >= f.published)
    )
    counts["invalid_snapshot_metadata"] = int((~valid).sum())
    f = f[valid].sort_values(["observed", "region", "video_id"], kind="stable")
    conflicts = f.groupby("video_id").agg(
        channels=("channelId", "nunique"), dates=("published", "nunique")
    )
    bad = conflicts.index[(conflicts.channels > 1) | (conflicts.dates > 1)]
    counts["inconsistent_identity_or_publication"] = len(bad)
    f = f[~f.video_id.isin(bad)]
    first = f.drop_duplicates("video_id").set_index("video_id")
    first = first[first.categoryId.isin([2, 26]) & first.title.fillna("").str.strip().ne("")]
    f = f[f.video_id.isin(first.index)]
    counts["sector_videos"] = len(first)
    counts["videos_with_title_changes"] = int(f.groupby("video_id").title.nunique().gt(1).sum())
    age = (f.observed - f.published).dt.total_seconds()
    candidates = f[(age >= 7 * DAY) & (age < 8 * DAY)]
    # At the same nominal observation time countries must agree on the global count.
    # Reject conflicting earliest snapshots rather than selecting favorable counts.
    earliest = candidates.groupby("video_id")["observed"].transform("min")
    candidates = candidates[candidates.observed == earliest]
    same_time = candidates.groupby("video_id").view_count.nunique()
    bad_counts = same_time.index[same_time > 1]
    counts["conflicting_target_counts"] = len(bad_counts)
    target = candidates[~candidates.video_id.isin(bad_counts)].drop_duplicates("video_id")
    numeric = pd.to_numeric(target.view_count, errors="coerce")
    target = target[np.isfinite(numeric) & (numeric >= 0) & (numeric % 1 == 0)].copy()
    rows = []
    for item in target.itertuples():
        meta = first.loc[item.video_id]
        if item.published >= pd.Timestamp(STOP, tz="UTC"):
            continue
        rows.append(
            {
                "creative_id": item.video_id,
                "channel_id": str(meta.channelId),
                "published_at": item.published.timestamp(),
                "observed_at": item.observed.timestamp(),
                "available_at": item.observed.timestamp() + DAY,
                "metadata_observed_at": meta.observed.timestamp(),
                "title": str(meta.title).strip(),
                "category": str(int(meta.categoryId)),
                "duration": 0.0,  # Missing: excluded from EVERY model, never treated as a measurement.
                "language": "unknown",
                "definition": "unknown",
                "target": int(item.view_count),
                "target_log": float(np.log1p(item.view_count)),
                "reserved_channel": False,  # All channels are external; no target-domain fitting.
                "region": item.region,
            }
        )
    cohort = pd.DataFrame(rows).sort_values(["published_at", "creative_id"]).reset_index(drop=True)
    counts["eligible_videos"] = len(cohort)
    return cohort, counts


def prepare(root=ROOT):
    if (root / "cohort.parquet").exists():
        raise ValueError("External cohort is immutable; preserve the existing audit")
    manifest = read_json(root / "acquisition.json")
    columns = [
        "video_id",
        "title",
        "publishedAt",
        "channelId",
        "categoryId",
        "trending_date",
        "view_count",
    ]
    pieces = []
    for item in manifest["files"]:
        path = root / item["file"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError("Source changed since acquisition")
        part = pd.read_csv(path, usecols=columns)
        part["region"] = item["region"]
        pieces.append(part)
    cohort, audit = build_cohort(pd.concat(pieces, ignore_index=True))
    del pieces
    cohort.to_parquet(root / "cohort.parquet", index=False)
    frame, _ = history_features(cohort)
    warm = frame[frame.history_count >= 3]
    audit.update(
        {
            "created_at": utcnow(),
            "cohort_sha256": sha256_file(root / "cohort.parquet"),
            "acquisition_sha256": sha256_file(root / "acquisition.json"),
            "warm_videos": len(warm),
            "warm_channels": int(warm.channel_id.nunique()),
            "by_year": pd.to_datetime(cohort.published_at, unit="s", utc=True)
            .dt.year.value_counts()
            .to_dict(),
            "target": "First reported snapshot at nominal age [168,192) hours; +24h availability margin",
            "duration_available": False,
            "test_outcomes_summarized": False,
        }
    )
    write_json(root / "audit.json", audit)
    return audit


def paired_indices(frame, cap=500):
    """Label-independent same-channel/week pair sampling, with stable channel seeds."""
    weeks = ((frame.published_at.to_numpy() - epoch("2016-07-01")) // (7 * DAY)).astype(int)
    grouped = defaultdict(list)
    for (channel, _), positions in frame.groupby(
        [frame.channel_id.to_numpy(), weeks]
    ).indices.items():
        if len(positions) >= 2:
            grouped[channel].append(np.asarray(positions))
    left, right, channels = [], [], []
    for channel, groups in sorted(grouped.items()):
        sizes = np.array([len(g) * (len(g) - 1) // 2 for g in groups])
        ends = np.cumsum(sizes)
        seed = int(hashlib.sha256(channel.encode()).hexdigest()[:16], 16)
        slots = np.random.default_rng(seed).choice(
            int(ends[-1]), min(cap, int(ends[-1])), replace=False
        )
        for slot in sorted(slots):
            group_id = int(np.searchsorted(ends, slot, side="right"))
            local = slot - (ends[group_id - 1] if group_id else 0)
            group = groups[group_id]
            offsets = np.arange(len(group))
            starts = offsets * (2 * len(group) - offsets - 1) // 2
            i = int(np.searchsorted(starts, local, side="right") - 1)
            j = int(i + 1 + local - starts[i])
            left.append(group[i])
            right.append(group[j])
            channels.append(channel)
    return np.asarray(left, dtype=int), np.asarray(right, dtype=int), np.asarray(channels)


def ranking_scores(frame, predictions):
    left, right, channels = paired_indices(frame)
    actual = np.sign(frame.target_log.to_numpy()[left] - frame.target_log.to_numpy()[right])
    keep = actual != 0
    scores = {}
    for name, values in predictions.items():
        predicted = np.sign(np.asarray(values)[left] - np.asarray(values)[right])
        accuracy = (predicted == actual).astype(float)
        accuracy[predicted == 0] = 0.5
        scores[name] = float(pd.Series(accuracy[keep]).groupby(channels[keep]).mean().mean())
    return scores


def model_matrices(encoder, matrices):
    """Drop unavailable duration/language/definition in source and target alike."""
    excluded = {"log_duration", "duration_relative", "recent_duration"}
    keep = [
        i
        for i, name in enumerate(encoder.context_names)
        if name in CONTEXT and name not in excluded or name.startswith("category:")
    ]
    count = len(encoder.context_names)
    context = matrices["context"][:, keep]
    return {
        "context": context,
        "lexical": np.column_stack(
            [context, matrices["text"][:, count:], matrices["retrieval"][:, -4:-1]]
        ),
        "semantic": np.column_stack(
            [context, matrices["semantic"][:, count:], matrices["semantic_retrieval"][:, -3:]]
        ),
        "linear": matrices["linear"][:, keep + list(range(count, matrices["linear"].shape[1]))],
    }


def add_analogies(matrices, query, embeddings, neighbors, pool, pool_embeddings, rows):
    """Retrieve relative performance, rather than inheriting another channel's size.

    Cross-channel neighbors come ONLY from the training pool and must have matured
    before each query publication, including queries used for fitting. Local
    neighbors are provided by the strict availability join. Query outcomes never
    enter their own features. The frozen encoder supplies semantic similarity.
    """
    features = np.zeros((len(query), 8), dtype=np.float32)
    pool_residual = np.clip((pool.target_log - pool.logmean_20).to_numpy(), -4, 4)
    local_residual = np.clip((query.target_log - query.logmean_20).to_numpy(), -4, 4)
    pool_available = pool.available_at.to_numpy()
    pool_channels = pool.channel_id.to_numpy()
    for start in range(0, len(rows), 256):
        ids = rows[start : start + 256]
        similarity = embeddings[ids] @ pool_embeddings.T
        admissible = (pool_available[None, :] < query.published_at.to_numpy()[ids, None]) & (
            pool_channels[None, :] != query.channel_id.to_numpy()[ids, None]
        )
        similarity[~admissible] = -1
        count = min(5, len(pool))
        top = np.argpartition(-similarity, count - 1, axis=1)[:, :count]
        sim = np.take_along_axis(similarity, top, axis=1)
        strength = np.maximum(sim - 0.3, 0) ** 4
        total = strength.sum(axis=1)
        residuals = pool_residual[top]
        features[ids, 4] = (strength * residuals).sum(axis=1) / (0.01 + total)
        features[ids, 5] = np.maximum(sim.max(axis=1), 0)
        features[ids, 6] = total
        features[ids, 7] = (strength * residuals**2).sum(axis=1) / (0.01 + total)
        for i in ids:
            previous = np.array(
                [j for j in neighbors[i] if query.history_count.iloc[j] >= 3], dtype=int
            )
            if not len(previous):
                continue
            sim_local = np.maximum(embeddings[previous] @ embeddings[i] - 0.3, 0)
            order = np.argsort(-sim_local, kind="stable")[:5]
            strength_local = sim_local[order] ** 4
            total_local = strength_local.sum()
            residual = local_residual[previous[order]]
            features[i, :4] = [
                np.dot(strength_local, residual) / (0.01 + total_local),
                sim_local.max(),
                total_local,
                np.dot(strength_local, residual**2) / (0.01 + total_local),
            ]
    matrices["analogy"] = np.column_stack([matrices["semantic"], features])
    return matrices


def fit_regressor(name, matrices, frame, train, predict, folder=None):
    spec = REGRESSORS[name]
    x = matrices[spec["bundle"]]
    anchor = frame.logmean_20.to_numpy()
    residual = frame.target_log.to_numpy()[train] - anchor[train]
    if spec["bundle"] == "linear":
        model = Ridge(alpha=spec["alpha"], solver="lsqr").fit(
            x[train], residual, sample_weight=weights(frame.iloc[train])
        )
    else:
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
        ).fit(x[train], residual, sample_weight=weights(frame.iloc[train]))
    if folder:
        folder.mkdir(parents=True, exist_ok=True)
        if spec["bundle"] == "linear":
            np.savez_compressed(
                folder / f"{name}.npz", coef=model.coef_, intercept=model.intercept_
            )
        else:
            model.save_model(str(folder / f"{name}.cbm"))
    return np.maximum(0, anchor[predict] + model.predict(x[predict]))


def fit_ranker(name, matrices, frame, train, predict, folder=None):
    spec = RANKERS[name]
    x = matrices[spec["bundle"]]
    left, right, channel = paired_indices(frame.iloc[train].reset_index(drop=True))
    y = frame.target_log.to_numpy()[train]
    keep = y[left] != y[right]
    left, right, channel = left[keep], right[keep], channel[keep]
    if len(left) < 20:
        raise ValueError("Too few training pairs")
    counts = Counter(channel)
    pair_weight = np.array([1 / counts[c] for c in channel])
    pair_weight /= pair_weight.mean()
    positive = y[left] > y[right]
    if "c" in spec:
        delta = x[train[left]] - x[train[right]]
        features = sparse.vstack([delta, -delta], format="csr")
        labels = np.r_[positive, ~positive].astype(int)
        model = LogisticRegression(
            C=spec["c"], fit_intercept=False, solver="liblinear", max_iter=1000, random_state=42
        ).fit(features, labels, sample_weight=np.tile(pair_weight, 2))
        result = np.asarray(x[predict] @ model.coef_.ravel()).ravel()
        if folder:
            np.savez_compressed(folder / f"{name}.npz", coef=model.coef_.ravel())
    else:
        pairs = np.column_stack([np.where(positive, left, right), np.where(positive, right, left)])
        model = CatBoostRanker(
            iterations=400,
            depth=spec["depth"],
            learning_rate=0.035,
            loss_function="PairLogit",
            l2_leaf_reg=30,
            random_seed=42,
            thread_count=4,
            verbose=False,
            allow_writing_files=False,
        ).fit(Pool(x[train], label=y, pairs=pairs, pairs_weight=pair_weight))
        result = model.predict(x[predict])
        if folder:
            model.save_model(str(folder / f"{name}.cbm"))
    return result


def fingerprint():
    h = hashlib.sha256(Path(__file__).read_bytes())
    from galaxies import reach_study

    h.update(Path(reach_study.__file__).read_bytes())
    for name in ("numpy", "pandas", "scikit-learn", "catboost", "scipy"):
        h.update(f"{name}:{importlib.metadata.version(name)}".encode())
    return h.hexdigest()


def source_data():
    # Closed final publications cannot enter training, feature vocabularies, or neighbors.
    full = pd.read_parquet(SOURCE / "cohort.parquet")
    selected = full.published_at < epoch("2016-08-11")
    semantic = load_semantic(SOURCE, full)[selected]
    frame, neighbors = history_features(full[selected].reset_index(drop=True))
    return frame, neighbors, semantic


def develop(run=RUN):
    if run.exists():
        raise ValueError("Use a fresh development directory and preserve all previous attempts")
    frame, neighbors, semantic = source_data()
    write_json(
        run / "plan.json",
        {
            "created_at": utcnow(),
            "code_sha256": fingerprint(),
            "source_audit_sha256": sha256_file(SOURCE / "audit.json"),
            "external_audit_sha256": sha256_file(ROOT / "audit.json"),
            "source_semantic_sha256": sha256_file(SOURCE / "semantic.json"),
            "external_semantic_sha256": sha256_file(ROOT / "semantic.json"),
            "regressors": REGRESSORS,
            "rankers": RANKERS,
            "baselines": BASELINES,
            "selection": "Minimum macro log MAE for counts; maximum macro channel pair accuracy for ordering. Context selected separately for each task. Exact ties use name.",
            "external_target_labels_used_for_selection": False,
            "primary": "Known external channels with >=3 mature in-sample outcomes; 2020 to March 2024",
            "success": ">=5% error reduction and positive paired 95% CI versus both references; >=500 videos/50 channels. Ranking separately positive 95% lower versus both; >=200 pairs/30 channels.",
        },
    )
    saved = []
    for start, stop in [("2016-07-25", "2016-08-01"), ("2016-08-01", "2016-08-08")]:
        train, valid = indices(frame, start, train=True), indices(frame, start, stop)
        encoder = ReachEncoder()
        matrices = model_matrices(encoder, encoder.fit_transform(frame, train, neighbors, semantic))
        add_analogies(
            matrices,
            frame,
            semantic,
            neighbors,
            frame.iloc[train],
            semantic[train],
            np.unique(np.r_[train, valid]),
        )
        part = frame.iloc[valid][
            ["creative_id", "channel_id", "category", "published_at", "target_log"]
        ].copy()
        print(f"Fold {start}: train={len(train)}, valid={len(valid)}", flush=True)
        for name in BASELINES:
            part[name] = frame.iloc[valid][name].to_numpy()
        for name in REGRESSORS:
            part[name] = fit_regressor(name, matrices, frame, train, valid)
            print(name, macro_error(part, part[name].to_numpy()), flush=True)
        for name in RANKERS:
            part[name] = fit_ranker(name, matrices, frame, train, valid)
            print(name, ranking_scores(part, {name: part[name].to_numpy()})[name], flush=True)
        saved.append(part)
    data = pd.concat(saved, ignore_index=True)
    errors = {n: macro_error(data, data[n].to_numpy()) for n in BASELINES + list(REGRESSORS)}
    ranks = ranking_scores(
        data, {n: data[n].to_numpy() for n in BASELINES + list(REGRESSORS) + list(RANKERS)}
    )

    def choose_error(names):
        return min(names, key=lambda n: (errors[n], n))

    def choose_rank(names):
        return min(names, key=lambda n: (-ranks[n], n))

    selection = {
        "history": choose_error(BASELINES),
        "context": choose_error([n for n, s in REGRESSORS.items() if s["bundle"] == "context"]),
        "content": choose_error([n for n, s in REGRESSORS.items() if s["bundle"] != "context"]),
        "ranking_history": choose_rank(BASELINES),
        "ranking_context": choose_rank(
            [n for n, s in (REGRESSORS | RANKERS).items() if s["bundle"] == "context"]
        ),
        "ranking_content": choose_rank(
            [n for n, s in (REGRESSORS | RANKERS).items() if s["bundle"] != "context"]
        ),
        "best_direct_ranker": choose_rank(list(RANKERS)),
        "errors": errors,
        "ranking_scores": ranks,
    }
    data.to_csv(run / "development_predictions.csv", index=False)
    selection.update(
        {
            "created_at": utcnow(),
            "plan_sha256": sha256_file(run / "plan.json"),
            "predictions_sha256": sha256_file(run / "development_predictions.csv"),
        }
    )
    write_json(run / "selection.json", selection)
    return selection


def verify_frozen(root=ROOT, run=RUN):
    plan, selection = read_json(run / "plan.json"), read_json(run / "selection.json")
    audit = read_json(root / "audit.json")
    if plan["code_sha256"] != fingerprint():
        raise ValueError("Code/dependencies changed after model selection")
    checks = [
        (run / "plan.json", selection["plan_sha256"]),
        (run / "development_predictions.csv", selection["predictions_sha256"]),
        (root / "audit.json", plan["external_audit_sha256"]),
        (SOURCE / "audit.json", plan["source_audit_sha256"]),
        (SOURCE / "semantic.json", plan["source_semantic_sha256"]),
        (root / "semantic.json", plan["external_semantic_sha256"]),
        (root / "cohort.parquet", audit["cohort_sha256"]),
        (SOURCE / "cohort.parquet", read_json(SOURCE / "audit.json")["cohort_sha256"]),
    ]
    if any(sha256_file(path) != expected for path, expected in checks):
        raise ValueError("Frozen artifacts changed")
    return plan, selection


def final(root=ROOT, run=RUN):
    """Open external evaluation once; no model selection or tuning on its outcomes."""
    _, selection = verify_frozen(root, run)
    ledger = run / "evaluation-opened.json"
    claim = {
        "opened_at": str(utcnow()),
        "selection_sha256": sha256_file(run / "selection.json"),
        "code_sha256": fingerprint(),
    }
    import json

    with ledger.open("x") as stream:
        json.dump(claim, stream, indent=2)
    frame, neighbors, semantic = source_data()
    external, external_neighbors = history_features(pd.read_parquet(root / "cohort.parquet"))
    external_semantic = load_semantic(root, external)
    train = indices(frame, "2016-08-08", train=True)
    calibration = indices(frame, "2016-08-08", "2016-08-11")
    encoder = ReachEncoder()
    source_matrices = model_matrices(
        encoder, encoder.fit_transform(frame, train, neighbors, semantic)
    )
    external_matrices = model_matrices(
        encoder, encoder.transform(external, external_neighbors, external_semantic)
    )
    add_analogies(
        source_matrices,
        frame,
        semantic,
        neighbors,
        frame.iloc[train],
        semantic[train],
        np.unique(np.r_[train, calibration]),
    )
    add_analogies(
        external_matrices,
        external,
        external_semantic,
        external_neighbors,
        frame.iloc[train],
        semantic[train],
        np.arange(len(external)),
    )
    matrices = {
        name: sparse.vstack([source_matrices[name], external_matrices[name]], format="csr")
        if sparse.issparse(source_matrices[name])
        else np.vstack([source_matrices[name], external_matrices[name]])
        for name in source_matrices
    }
    combined = pd.concat([frame, external], ignore_index=True)
    positions = np.r_[calibration, np.arange(len(frame), len(combined))]
    model_folder = run / "models"
    encoder.save(model_folder)
    names = {
        selection[role]
        for role in (
            "context",
            "content",
            "ranking_context",
            "ranking_content",
            "best_direct_ranker",
        )
    }
    output = {}
    for name in sorted(names):
        print(f"Frozen fit: {name}", flush=True)
        fit = fit_regressor if name in REGRESSORS else fit_ranker
        output[name] = fit(name, matrices, combined, train, positions, model_folder)
    history_name = selection["history"]
    output[history_name] = combined[history_name].to_numpy()[positions]
    rank_history = selection["ranking_history"]
    output[rank_history] = combined[rank_history].to_numpy()[positions]
    prediction_frame = external[
        [
            "creative_id",
            "channel_id",
            "category",
            "published_at",
            "observed_at",
            "available_at",
            "target",
            "target_log",
            "history_count",
        ]
    ].copy()
    point, rank, radii = {}, {}, {}
    for role in ("history", "context", "content"):
        values = output[selection[role]]
        errors = np.abs(frame.target_log.to_numpy()[calibration] - values[: len(calibration)])
        radii[role] = conformal_radius(errors)
        point[role] = values[len(calibration) :]
        rank[role] = output[selection[f"ranking_{role}"]][len(calibration) :]
        prediction_frame[role] = point[role]
        prediction_frame[f"ranking_{role}"] = rank[role]
    rank["direct_ranker"] = output[selection["best_direct_ranker"]][len(calibration) :]
    prediction_frame["direct_ranker"] = rank["direct_ranker"]
    warm = external.history_count.to_numpy() >= 3
    primary = external[warm]
    gains = {
        name: paired_gain(primary, point["content"][warm], point[name][warm])
        for name in ("history", "context")
    }
    rank_result = ranking(primary, {name: values[warm] for name, values in rank.items()})
    enough = len(primary) >= 500 and primary.channel_id.nunique() >= 50
    error_confirmed = enough and all(
        g["relative_reduction_pct"] >= 5 and g["ci95"][0] > 0 for g in gains.values()
    )
    rank_confirmed = (
        rank_result["pairs"] >= 200
        and rank_result["channels"] >= 30
        and all(rank_result[f"content_vs_{name}_ci95"][0] > 0 for name in ("history", "context"))
    )
    subgroups = {}
    for name, mask in {
        "autos": warm & external.category.eq("2").to_numpy(),
        "howto_style": warm & external.category.eq("26").to_numpy(),
        "little_history": ~warm,
    }.items():
        if mask.any():
            subgroups[name] = {
                role: prediction_metrics(external[mask], values[mask], radii[role])
                for role, values in point.items()
            }
    result = {
        "created_at": utcnow(),
        "selection": selection,
        "train_videos": len(train),
        "calibration_videos": len(calibration),
        "external_videos": len(external),
        "primary_videos": len(primary),
        "primary_channels": int(primary.channel_id.nunique()),
        "primary_metrics": {
            role: prediction_metrics(primary, values[warm], radii[role])
            for role, values in point.items()
        },
        "content_gains": gains,
        "ranking": rank_result,
        "subgroups": subgroups,
        "calibration_radii": radii,
        "confirmed_error_gain": bool(error_confirmed),
        "confirmed_ranking_gain": bool(rank_confirmed),
        "case_solved": False,
        "production_eligible": False,
        "limitations": [
            "Selected trending videos, incomplete channel history",
            "2016 training vs 2020-2024 evaluation; 7 calendar dates vs nominal 168-192h target",
            "Retrospective first-observed titles; duration unavailable and removed in both domains",
            "No image, speech or editing gain established by this test",
            "Source-calibrated intervals may fail under temporal shift; coverage is measured",
            "Sparse within-week pairs cannot establish reliable ordering on the external population",
        ],
    }
    prediction_frame.to_csv(run / "test_predictions.csv", index=False)
    result["predictions_sha256"] = sha256_file(run / "test_predictions.csv")
    write_json(run / "final.json", result)
    return result
