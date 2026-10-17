"""Time-separated full-video retention benchmark, never a popularity certification."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from galaxies.io import sha256_file, utcnow, write_json

BUNDLES = {
    "context": [],
    "caption": ["caption"],
    "image": ["image"],
    "speech": ["speech"],
    "editing": ["editing"],
    "full": ["caption", "image", "speech", "editing"],
}
CANDIDATES = {
    "ridge_10": {"family": "ridge", "alpha": 10},
    "ridge_100": {"family": "ridge", "alpha": 100},
    "ridge_1000": {"family": "ridge", "alpha": 1000},
    "catboost_3": {"family": "catboost", "depth": 3, "iterations": 300},
    "catboost_5": {"family": "catboost", "depth": 5, "iterations": 300},
}
CONTEXT_COLUMNS = [
    "channel_prior_mean",
    "channel_prior_std",
    "category_prior_mean",
    "category_prior_std",
    "history_raw",
    "history_shrunk",
    "global_prior_mean",
]


def epoch(local_time):
    return pd.Timestamp(local_time, tz="Asia/Shanghai").timestamp()


def code_hash():
    digest = hashlib.sha256()
    for filename in (
        "retention_benchmark.py",
        "shortvideo.py",
        "deep_media.py",
        "video_structure.py",
        "media.py",
    ):
        digest.update(Path(__file__).with_name(filename).read_bytes())
    for name in ("numpy", "pandas", "scikit-learn", "catboost"):
        digest.update(f"{name}:{importlib.metadata.version(name)}".encode())
    return digest.hexdigest()


def load_study(root):
    frame = pd.read_parquet(root / "cohort.parquet")
    index = json.loads((root / "features/index.json").read_text())
    if not index.get("extraction_complete"):
        raise ValueError("A extração de todos os candidatos ainda não terminou.")
    extractor = json.loads((root / "features/extractor.json").read_text())
    if index["extractor_fingerprint"] != extractor["fingerprint"]:
        raise ValueError("Extratores diferentes na base.")
    features, file_hashes = {}, {}
    for video, info in index["rows"].items():
        path = root / "features" / info["file"]
        if path.parent.resolve() != (root / "features").resolve():
            raise ValueError("Arquivo de atributos fora da pasta do estudo.")
        record = json.loads(path.read_text())
        if record["extractor_fingerprint"] != index["extractor_fingerprint"]:
            raise ValueError("Versão inconsistente de atributos.")
        if record["content_hash"] != info["content_hash"]:
            raise ValueError("Hash de mídia inconsistente.")
        features[video] = record
        file_hashes[video] = sha256_file(path)
    before = len(frame)
    frame = frame[frame.creative_id.isin(features)].copy()
    frame["content_hash"] = frame.creative_id.map(lambda v: features[v]["content_hash"])
    frame = frame.sort_values(["first_seen", "creative_id"])
    duplicate_count = int(frame.content_hash.duplicated().sum())
    frame = frame.drop_duplicates("content_hash").reset_index(drop=True)
    return (
        frame,
        features,
        {
            "acquisition_sha256": sha256_file(root / "acquisition.json"),
            "cohort_sha256": sha256_file(root / "cohort.parquet"),
            "index_sha256": sha256_file(root / "features/index.json"),
            "feature_sha256": file_hashes,
            "extractor": extractor,
            "eligible": before,
            "missing_or_failed_extractions": before - len(frame) - duplicate_count,
            "exact_media_duplicates_removed": duplicate_count,
            "used": len(frame),
            "deduplication_limit": "Exact original-file hashes among extracted videos; not a guarantee against modified reposts.",
        },
    )


def make_plan(frame, provenance):
    final_start = epoch("2022-09-20 12:00")
    folds = []
    for start, stop in (
        ("2022-09-18", "2022-09-18 12:00"),
        ("2022-09-18 12:00", "2022-09-19 12:00"),
    ):
        a, b = epoch(start), epoch(stop)
        training = frame.index[frame.available_at < a].tolist()
        validation = frame.index[
            (frame.first_seen >= a) & (frame.first_seen < b) & (frame.available_at < final_start)
        ].tolist()
        if len(training) < 60 or len(validation) < 20:
            raise ValueError("Amostra insuficiente para o desenvolvimento temporal.")
        folds.append(
            {"decision_start": a, "decision_end": b, "train": training, "validation": validation}
        )
    final = {
        "train": frame.index[frame.available_at < final_start].tolist(),
        "test": frame.index[frame.first_seen >= final_start].tolist(),
    }
    if len(final["test"]) < 50:
        raise ValueError("Menos de 50 vídeos no teste reservado; não concluir superioridade.")
    return {
        "created_at": utcnow().isoformat(),
        "protocol": "full-video-retention-v1",
        "code_sha256": code_hash(),
        "frame_sha256": hashlib.sha256(frame.to_json(orient="records").encode()).hexdigest(),
        "provenance": provenance,
        "target": "Mean capped watch fraction in first 24h after first panel observation; NOT publication popularity.",
        "primary_metric": "MAE in retention percentage points, equally weighted by video",
        "secondary_metrics": [
            "macro creator MAE",
            "RMSE",
            "within 10 percentage points",
            "paired creator bootstrap",
            "same-creator pair ranking",
        ],
        "folds": folds,
        "final_start": final_start,
        "final": final,
        "creative_ids": frame.creative_id.tolist(),
        "candidates": CANDIDATES,
        "bundles": BUNDLES,
        "selection": "lowest video-weighted MAE across disjoint development windows; candidates within 1% prefer Ridge then stronger regularization/shallow trees",
        "baselines": [
            "raw creator history",
            "creator history shrunk by 20 events to category",
            "independently selected context+duration regressor",
            "caption+context model",
        ],
        "success_rule": "Full model reduces test MAE by at least 5% versus BOTH adjusted history and selected context, and lower 95% creator-bootstrap gain is positive versus both. At least 100 test videos and 50 creators. Otherwise no confirmed advantage.",
        "predictive_intervals": "Not calibrated in this short observational window; no coverage guarantee is claimed.",
        "production_eligible": False,
    }


class FusionEncoder:
    """Each modality's PCA and numerical normalization are fit ONLY on training videos."""

    def __init__(self, bundle):
        self.bundle = bundle
        self.blocks = BUNDLES[bundle]
        self.projections = {}

    def _numeric(self, frame, records):
        matrix = frame[CONTEXT_COLUMNS].to_numpy(float)
        counts = np.log1p(
            frame[["channel_prior_events", "category_prior_events", "duration"]].to_numpy(float)
        )
        categories = np.column_stack(
            [(frame.root_id == c).to_numpy(float) for c in self.categories]
        )
        pieces = [matrix, counts, categories]
        for block, field in (("speech", "speech_features"), ("editing", "editing")):
            if block not in self.blocks:
                continue
            values = [r[field] if field != "editing" else r[field]["features"] for r in records]
            keys = self.speech_keys if block == "speech" else self.editing_keys
            pieces.append(np.array([[r.get(k, np.nan) for k in keys] for r in values]))
        return np.column_stack(pieces)

    def _vectors(self, records):
        fields = []
        if "caption" in self.blocks:
            fields.append("caption_embedding")
        if "speech" in self.blocks:
            fields.append("speech_embedding")
        if "image" in self.blocks:
            fields.extend(["visual_mean", "visual_opening", "visual_variation"])
        return {field: np.array([r[field] for r in records], dtype=float) for field in fields}

    def fit_transform(self, frame, features):
        records = [features[v] for v in frame.creative_id]
        self.categories = sorted(frame.root_id.unique().tolist())
        self.speech_keys = sorted({k for r in records for k in r["speech_features"]})
        self.editing_keys = sorted({k for r in records for k in r["editing"]["features"]})
        numeric = self._numeric(frame, records)
        self.medians = np.array([np.nanmedian(c) if np.isfinite(c).any() else 0 for c in numeric.T])
        filled = np.where(np.isfinite(numeric), numeric, self.medians)
        self.scales = np.maximum(filled.std(axis=0), 0.01)
        for name, values in self._vectors(records).items():
            projection = PCA(
                n_components=min(16, len(frame) - 1, values.shape[1]), svd_solver="full"
            )
            projection.fit(values)
            self.projections[name] = projection
        return self.transform(frame, features)

    def transform(self, frame, features):
        records = [features[v] for v in frame.creative_id]
        numeric = self._numeric(frame, records)
        numeric = (
            np.where(np.isfinite(numeric), numeric, self.medians) - self.medians
        ) / self.scales
        pieces = [np.clip(numeric, -10, 10)]
        for name, values in self._vectors(records).items():
            projection = self.projections[name]
            # Variance scale learned only on training data; bounded near-zero components.
            scale = np.maximum(np.sqrt(projection.explained_variance_), 0.01)
            pieces.append(np.clip(projection.transform(values) / scale, -10, 10))
        return np.column_stack(pieces)

    def save(self, folder):
        arrays = {"medians": self.medians, "scales": self.scales}
        for name, pca in self.projections.items():
            arrays.update(
                {
                    name + "_mean": pca.mean_,
                    name + "_components": pca.components_,
                    name + "_variance": pca.explained_variance_,
                }
            )
        np.savez_compressed(folder / "encoder.npz", **arrays)
        write_json(
            folder / "encoder.json",
            {
                "bundle": self.bundle,
                "categories": self.categories,
                "speech_keys": self.speech_keys,
                "editing_keys": self.editing_keys,
                "projections": list(self.projections),
            },
        )


def fit_predict(training, evaluation, features, bundle, candidate, folder=None):
    encoder = FusionEncoder(bundle)
    train_x = encoder.fit_transform(training, features)
    eval_x = encoder.transform(evaluation, features)
    config = CANDIDATES[candidate]
    if config["family"] == "ridge":
        model = Ridge(alpha=config["alpha"]).fit(train_x, training.target)
    else:
        model = CatBoostRegressor(
            iterations=config["iterations"],
            depth=config["depth"],
            learning_rate=0.03,
            l2_leaf_reg=10,
            loss_function="RMSE",
            random_seed=42,
            thread_count=4,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(train_x, training.target)
    prediction = np.clip(model.predict(eval_x), 0, 1)
    if folder:
        folder.mkdir(parents=True, exist_ok=False)
        encoder.save(folder)
        write_json(folder / "config.json", {"candidate": candidate, "config": config})
        if config["family"] == "ridge":
            np.savez_compressed(folder / "ridge.npz", coef=model.coef_, intercept=model.intercept_)
        else:
            model.save_model(str(folder / "catboost.cbm"))
    return prediction


def metrics(frame, prediction):
    errors = np.abs(frame.target.to_numpy() - prediction)
    macro = pd.Series(errors).groupby(frame.channel_id.to_numpy()).mean()
    pairs = []
    # Report small-sample pairs rather than present a percentage without denominator.
    for _, group in frame.assign(prediction=prediction).groupby("channel_id"):
        rows = list(group.itertuples())
        for i, first in enumerate(rows):
            for second in rows[i + 1 :]:
                if abs(first.target - second.target) >= 0.05:
                    direction = (first.target - second.target) * (
                        first.prediction - second.prediction
                    )
                    pairs.append(1.0 if direction > 0 else 0.5 if direction == 0 else 0.0)
    return {
        "videos": len(frame),
        "creators": int(frame.channel_id.nunique()),
        "mae_pp": float(errors.mean() * 100),
        "macro_creator_mae_pp": float(macro.mean() * 100),
        "rmse_pp": float(np.sqrt(np.mean(errors**2)) * 100),
        "within_10pp": float(np.mean(errors <= 0.1)),
        "pair_count": len(pairs),
        "pair_accuracy": float(np.mean(pairs)) if pairs else None,
    }


def paired_gain(frame, prediction, reference):
    gain = (
        np.abs(frame.target.to_numpy() - reference) - np.abs(frame.target.to_numpy() - prediction)
    ) * 100
    groups = (
        pd.DataFrame({"gain": gain, "channel": frame.channel_id.to_numpy()})
        .groupby("channel")
        .gain.agg(["sum", "count"])
    )
    rng = np.random.default_rng(20261017)
    ids = rng.integers(0, len(groups), size=(5000, len(groups)))
    boot = groups["sum"].to_numpy()[ids].sum(axis=1) / groups["count"].to_numpy()[ids].sum(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "mean_error_reduction_pp": float(gain.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "creators": len(groups),
        "resamples": 5000,
        "method": "paired creator-cluster bootstrap, video-weighted MAE; observational panel, shared-viewer dependence not resolved",
    }


def choose(results, bundle):
    candidates = []
    for name, config in CANDIDATES.items():
        rows = [r for r in results if r["bundle"] == bundle and r["candidate"] == name]
        error = sum(r["metrics"]["mae_pp"] * r["metrics"]["videos"] for r in rows) / sum(
            r["metrics"]["videos"] for r in rows
        )
        complexity = (0, -config["alpha"]) if config["family"] == "ridge" else (1, config["depth"])
        candidates.append({"candidate": name, "mae_pp": error, "complexity": complexity})
    best = min(c["mae_pp"] for c in candidates)
    competitive = [c for c in candidates if c["mae_pp"] <= 1.01 * best]
    return min(competitive, key=lambda c: (c["complexity"], c["mae_pp"]))["candidate"]


def develop(root: Path, folder: Path):
    frame, features, provenance = load_study(root)
    plan = make_plan(frame, provenance)
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "plan.json", plan)
    results = []
    for i, fold in enumerate(plan["folds"]):
        train, validation = frame.loc[fold["train"]], frame.loc[fold["validation"]]
        for bundle in BUNDLES:
            for candidate in CANDIDATES:
                print(f"Desenvolvimento {i + 1}/2 · {bundle} · {candidate}", flush=True)
                prediction = fit_predict(train, validation, features, bundle, candidate)
                results.append(
                    {
                        "fold": i,
                        "bundle": bundle,
                        "candidate": candidate,
                        "metrics": metrics(validation, prediction),
                    }
                )
        for baseline in ("history_raw", "history_shrunk"):
            results.append(
                {
                    "fold": i,
                    "bundle": "baseline",
                    "candidate": baseline,
                    "metrics": metrics(validation, validation[baseline].to_numpy()),
                }
            )
    selection = {bundle: choose(results, bundle) for bundle in BUNDLES}
    write_json(
        folder / "development.json",
        {"results": results, "selection": selection, "final_test_used": False},
    )
    frozen = {
        "selection": selection,
        "plan_sha256": sha256_file(folder / "plan.json"),
        "development_sha256": sha256_file(folder / "development.json"),
    }
    write_json(folder / "selection.json", frozen)
    return frozen


def evaluate_final(root: Path, folder: Path):
    plan = json.loads((folder / "plan.json").read_text())
    frozen = json.loads((folder / "selection.json").read_text())
    for name in ("plan", "development"):
        if sha256_file(folder / f"{name}.json") != frozen[name + "_sha256"]:
            raise ValueError("O desenvolvimento congelado foi alterado.")
    original = json.loads((folder / "development.json").read_text())["selection"]
    if original != frozen["selection"] or code_hash() != plan["code_sha256"]:
        raise ValueError("Código ou seleção mudou após o desenvolvimento.")
    frame, features, provenance = load_study(root)
    if (
        provenance != plan["provenance"]
        or hashlib.sha256(frame.to_json(orient="records").encode()).hexdigest()
        != plan["frame_sha256"]
    ):
        raise ValueError("Dados ou extratores mudaram após o plano.")
    ledger = folder.parent / f"retention-final-opened-{provenance['cohort_sha256']}.json"
    with ledger.open("x") as handle:
        json.dump({"opened_at": utcnow().isoformat(), "study": str(folder)}, handle)
    train, test = frame.loc[plan["final"]["train"]], frame.loc[plan["final"]["test"]]
    predictions = {name: test[name].to_numpy() for name in ("history_raw", "history_shrunk")}
    for bundle, candidate in original.items():
        print(f"Teste final reservado · {bundle} · {candidate}", flush=True)
        predictions[bundle] = fit_predict(
            train, test, features, bundle, candidate, folder / "models" / bundle
        )
    all_metrics = {name: metrics(test, values) for name, values in predictions.items()}
    comparisons = {
        name: paired_gain(test, predictions["full"], predictions[name])
        for name in ("history_raw", "history_shrunk", "context", "caption")
    }
    improvements = {
        name: 1 - all_metrics["full"]["mae_pp"] / all_metrics[name]["mae_pp"]
        for name in comparisons
    }
    enough = len(test) >= 100 and test.channel_id.nunique() >= 50
    passed = bool(
        enough
        and all(
            improvements[name] >= 0.05 and comparisons[name]["ci95_low"] > 0
            for name in ("history_shrunk", "context")
        )
    )
    exported = test[
        ["creative_id", "channel_id", "first_seen", "target", "exposures", "channel_prior_events"]
    ].copy()
    for name, values in predictions.items():
        exported[name] = values
    exported.to_csv(folder / "test_predictions.csv", index=False)
    report = {
        "completed_at": utcnow().isoformat(),
        "evidence": "real full videos, retrospective first-panel-observation retention; NOT upload-time popularity",
        "train_videos": len(train),
        "test_videos": len(test),
        "test_creators": int(test.channel_id.nunique()),
        "test_first_seen_period_utc": [
            pd.Timestamp(test.first_seen.min(), unit="s", tz="UTC").isoformat(),
            pd.Timestamp(test.first_seen.max(), unit="s", tz="UTC").isoformat(),
        ],
        "selection": original,
        "metrics": all_metrics,
        "full_model_comparisons": comparisons,
        "relative_mae_reduction": improvements,
        "success_rule_met": passed,
        "eligible_for_production": False,
        "limitations": [
            "First panel exposure is not publication; no fixed-horizon view/like prediction validated.",
            "Observed recommendations to a volunteer panel; exposure and audience selection remain confounded.",
            "Requires at least 10 prior creator events and 10 outcome exposures; no cold-start claim.",
            "One short collection window in September 2022; not Brazilian advertising or prospective validation.",
            "Original upload-version integrity and modified reposts cannot be fully verified.",
            "Whisper transcription and scene detection are automatic; human-label extraction accuracy was not established.",
            "Pretrained encoder overlap with source videos is unknown; no popularity labels were supplied to the encoders.",
        ],
    }
    write_json(folder / "final.json", report)
    return report
