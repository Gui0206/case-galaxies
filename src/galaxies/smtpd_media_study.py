"""Ablate complete short-video modalities against the same textual/contextual models.

All eligible test videos remain in the primary population. Missing/long media use
the frozen text prediction; no fabricated zero-valued media enter model fitting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from galaxies.io import read_json, sha256_file, utcnow, write_json
from galaxies.reach_study import macro_error, weights
from galaxies.retention_benchmark import validate_media_record
from galaxies.smtpd_study import FOLDS, ROOT, data, matrices, partitions, selected_prediction
from galaxies.transfer_study import ranking_scores

BUNDLES = {
    "text_control": [],
    "image": ["image"],
    "speech": ["speech"],
    "editing": ["editing"],
    "full": ["image", "speech", "editing"],
}
MODELS = ["tree3", "ridge100", "ridge1000"]
BLENDS = [0.25, 0.5, 1.0]


def load_media(frame, require_complete=True):
    index = read_json(ROOT / "features/index.json")
    acquisition = read_json(ROOT / "media-acquisition.json")
    if require_complete and (
        not index.get("extraction_complete") or not acquisition.get("complete")
    ):
        raise ValueError("Finish the declared acquisition/extraction before comparing media models")
    records, seen, hashes, duplicates = {}, set(), {}, []
    for video in frame.creative_id:
        if video not in index["rows"]:
            continue
        entry = index["rows"][video]
        path = ROOT / "features" / entry["file"]
        if path.resolve().parent != (ROOT / "features").resolve():
            raise ValueError("Media feature path outside declared folder")
        if sha256_file(path) != entry["sha256"]:
            raise ValueError("Media feature checksum changed")
        record = read_json(path)
        validate_media_record(record)
        if record["extractor_fingerprint"] != index["extractor_fingerprint"]:
            raise ValueError("Mixed media extractors")
        digest = record["content_hash"]
        if digest in seen:
            duplicates.append(video)
            continue
        seen.add(digest)
        records[video] = record
        hashes[video] = entry["sha256"]
    return records, {
        "index_sha256": sha256_file(ROOT / "features/index.json"),
        "acquisition_sha256": sha256_file(ROOT / "media-acquisition.json"),
        "extractor_fingerprint": index["extractor_fingerprint"],
        "feature_hashes": hashes,
        "exact_duplicates_fallback_to_text": duplicates,
        "available_videos": len(records),
    }


class MediaEncoder:
    def __init__(self, bundle):
        self.blocks = BUNDLES[bundle]
        self.projections = {}

    def numeric(self, records):
        columns = []
        for block, keys in self.keys.items():
            for key in keys:
                columns.append(
                    [
                        r["speech_features" if block == "speech" else "editing"].get(key, np.nan)
                        if block == "speech"
                        else r["editing"]["features"].get(key, np.nan)
                        for r in records
                    ]
                )
        return np.column_stack(columns) if columns else np.empty((len(records), 0))

    def fit(self, records):
        self.keys = {}
        for block in ["speech", "editing"]:
            if block in self.blocks:
                self.keys[block] = sorted(
                    {
                        k
                        for r in records
                        for k in (
                            r["speech_features"] if block == "speech" else r["editing"]["features"]
                        )
                    }
                )
        fields = []
        if "image" in self.blocks:
            fields += ["visual_mean", "visual_opening", "visual_variation"]
        if "speech" in self.blocks:
            fields += ["speech_embedding"]
        for field in fields:
            vectors = np.array([r[field] for r in records])
            self.projections[field] = PCA(
                n_components=min(16, len(records) - 1, vectors.shape[1]),
                svd_solver="randomized",
                random_state=42,
            ).fit(vectors)
        numeric = self.numeric(records)
        self.imputer = (
            SimpleImputer(strategy="median", keep_empty_features=True).fit(numeric)
            if numeric.shape[1]
            else None
        )
        return self

    def transform(self, records):
        values = []
        if self.imputer is not None:
            values.append(self.imputer.transform(self.numeric(records)))
        for field, projection in self.projections.items():
            values.append(projection.transform(np.array([r[field] for r in records])))
        return np.column_stack(values) if values else np.empty((len(records), 0))

    def save(self, folder):
        folder.mkdir(exist_ok=True, parents=True)
        arrays = {"numeric_medians": self.imputer.statistics_ if self.imputer is not None else []}
        for field, projection in self.projections.items():
            arrays[f"{field}_mean"] = projection.mean_
            arrays[f"{field}_components"] = projection.components_
        np.savez_compressed(folder / "media_encoder.npz", **arrays)
        write_json(
            folder / "media_encoder.json",
            {"blocks": self.blocks, "keys": self.keys, "projections": list(self.projections)},
        )


def fit_media(bundle, model_name, x, frame, train, predict, records, folder=None):
    encoder = MediaEncoder(bundle).fit([records[v] for v in frame.iloc[train].creative_id])
    tx = np.column_stack(
        [x[train], encoder.transform([records[v] for v in frame.iloc[train].creative_id])]
    )
    px = np.column_stack(
        [x[predict], encoder.transform([records[v] for v in frame.iloc[predict].creative_id])]
    )
    anchor = frame.logmean_20.to_numpy()
    residual = frame.target_log.to_numpy()[train] - anchor[train]
    scale = StandardScaler().fit(tx)
    if model_name.startswith("ridge"):
        model = Ridge(alpha=float(model_name.removeprefix("ridge"))).fit(
            scale.transform(tx), residual, sample_weight=weights(frame.iloc[train])
        )
        output = anchor[predict] + model.predict(scale.transform(px))
    else:
        model = CatBoostRegressor(
            iterations=400,
            depth=3,
            learning_rate=0.035,
            loss_function="MAE",
            l2_leaf_reg=30,
            random_seed=42,
            thread_count=4,
            verbose=False,
            allow_writing_files=False,
        ).fit(tx, residual, sample_weight=weights(frame.iloc[train]))
        output = anchor[predict] + model.predict(px)
    if folder:
        encoder.save(folder)
        np.savez_compressed(folder / "scale.npz", mean=scale.mean_, scale=scale.scale_)
        if model_name.startswith("ridge"):
            np.savez_compressed(folder / "model.npz", coef=model.coef_, intercept=model.intercept_)
        else:
            model.save_model(str(folder / "model.cbm"))
    return np.maximum(0, output)


def predict_base(name, selection, x, frame, all_train, warm_train, neighbors, predict, folder=None):
    if name in selection["combinations"]:
        spec = selection["combinations"][name]
        parts = {
            k: selected_prediction(
                spec[k], x, frame, all_train, warm_train, neighbors, predict, folder
            )
            for k in ["context", "content"]
        }
        return (1 - spec["content_weight"]) * parts["context"] + spec["content_weight"] * parts[
            "content"
        ]
    return selected_prediction(name, x, frame, all_train, warm_train, neighbors, predict, folder)


def develop(base_run=Path(".runtime/reports/smtpd-v2"), run=Path(".runtime/reports/smtpd-media")):
    if (run / "selection.json").exists():
        raise ValueError("Preserve previous media selection")
    frame, neighbors, semantic = data(development=True)
    records, provenance = load_media(frame)
    available = frame.creative_id.isin(records).to_numpy()
    selection = read_json(base_run / "ensemble-selection.json")
    base = pd.read_csv(base_run / "ensemble-development.csv").set_index("creative_id")
    write_json(
        run / "plan.json",
        {
            "created_at": utcnow(),
            "provenance": provenance,
            "base_selection_sha256": sha256_file(base_run / "ensemble-selection.json"),
            "bundles": BUNDLES,
            "models": MODELS,
            "blends": BLENDS,
            "fallback": "frozen text ensemble",
            "primary_includes_missing_media": True,
        },
    )
    parts = []
    for start, stop in FOLDS:
        all_train, warm, valid = partitions(frame, start, stop)
        _, x = matrices(frame, neighbors, semantic, all_train)
        train = warm[available[warm]]
        ready = valid[available[valid]]
        if len(train) < 20 or len(ready) < 10:
            raise ValueError("Too few complete-media examples in a declared validation fold")
        part = frame.iloc[valid][
            ["creative_id", "channel_id", "category", "published_at", "target_log", "duration"]
        ].copy()
        part["media_available"] = available[valid]
        for role in ["history", "context", "content"]:
            part[role] = base.loc[part.creative_id, selection[role]].to_numpy()
        for bundle in BUNDLES:
            for model in MODELS:
                prediction = fit_media(
                    bundle, model, x["semantic_retrieval"], frame, train, ready, records
                )
                for blend in BLENDS:
                    current = part.content.to_numpy().copy()
                    current[available[valid]] = (1 - blend) * current[
                        available[valid]
                    ] + blend * prediction
                    part[f"{bundle}:{model}@{blend:g}"] = current
        parts.append(part)
        print(
            {"fold": start, "media_train": len(train), "media_validation": len(ready)}, flush=True
        )
    predictions = pd.concat(parts, ignore_index=True)
    names = ["content"] + [f"{b}:{m}@{a:g}" for b in BUNDLES for m in MODELS for a in BLENDS]
    errors = {n: macro_error(predictions, predictions[n].to_numpy()) for n in names}
    selected = {
        b: min(
            ["content"] + [n for n in names if n.startswith(b + ":")], key=lambda n: (errors[n], n)
        )
        for b in BUNDLES
    }
    scores = ranking_scores(predictions, {n: predictions[n].to_numpy() for n in names})
    predictions.to_csv(run / "development.csv", index=False)
    result = {
        "created_at": utcnow(),
        "selected": selected,
        "errors": errors,
        "ranking_diagnostic": scores,
        "plan_sha256": sha256_file(run / "plan.json"),
        "predictions_sha256": sha256_file(run / "development.csv"),
    }
    write_json(run / "selection.json", result)
    return {b: {"model": n, "error": errors[n]} for b, n in selected.items()}
