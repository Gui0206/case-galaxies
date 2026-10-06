"""CatBoost, temporal evaluation, split conformal intervals and auditable artifacts."""

import importlib.metadata
import math
import os
import platform
import uuid
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor, Pool

from galaxies.data import ChannelHistory, audit, load_records, save_records, temporal_split
from galaxies.features import FeatureEncoder, label
from galaxies.io import read_json, sha256_file, utcnow, write_json
from galaxies.metrics import (
    calibrate_pairs,
    paired_channel_bootstrap,
    probability_metrics,
    regression_metrics,
)
from galaxies.schemas import FEATURES, Driver, PostRecord, TrainConfig


def conformal_adjustment(scores: np.ndarray, coverage: float) -> float:
    """Finite-sample split conformal quantile. Reject unsupported coverage levels."""
    rank = math.ceil((len(scores) + 1) * coverage)
    if not len(scores) or rank > len(scores):
        raise ValueError("Amostra de calibração insuficiente para a cobertura solicitada.")
    return float(max(0, np.sort(scores)[rank - 1]))


def predict_intervals(point, quantiles, matrix, correction: float):
    center = np.asarray(point.predict(matrix)).ravel()
    quantile = np.asarray(quantiles.predict(matrix))
    low = np.minimum(np.min(quantile, axis=1), center) - correction
    high = np.maximum(np.max(quantile, axis=1), center) + correction
    return center, np.expm1(np.clip(low, 0, 50)), np.expm1(np.clip(high, 0, 50))


def train(records: list[PostRecord], model_root: Path, config: TrainConfig | None = None,
          dataset_hash: str | None = None) -> dict:
    config = config or TrainConfig()
    dataset_audit = audit(records)
    split = temporal_split(records, config.held_out_channels)
    parts = split["parts"]
    if len(parts["train"]) < config.min_train_rows:
        raise ValueError(f"Treino requer {config.min_train_rows} observações após os expurgos.")
    if any(len(parts[name]) < 20 for name in ("validation", "calibration", "test")):
        raise ValueError("Validação, calibração e teste precisam de pelo menos 20 registros cada.")
    history_records = [r for r in records if r.channel_id not in config.held_out_channels]
    history = ChannelHistory(history_records)
    encoder = FeatureEncoder(config.feature_blocks, config.seed)
    matrices = {"train": encoder.fit(parts["train"], history)}
    for name in ("validation", "calibration", "test"):
        matrices[name] = encoder.transform_records(parts[name], history)
    targets = {name: np.log1p([r.target for r in rows]) for name, rows in parts.items()}
    kwargs = {"iterations": config.iterations, "depth": config.depth,
              "learning_rate": config.learning_rate, "random_seed": config.seed,
              "verbose": False, "allow_writing_files": False, "thread_count": 2}
    point = CatBoostRegressor(loss_function="RMSE", **kwargs)
    point.fit(matrices["train"], targets["train"],
              eval_set=(matrices["validation"], targets["validation"]),
              early_stopping_rounds=35, use_best_model=True)
    alpha = (1 - config.coverage) / 2
    quantiles = CatBoostRegressor(loss_function=f"MultiQuantile:alpha={alpha},{1 - alpha}", **kwargs)
    quantiles.fit(matrices["train"], targets["train"],
                  eval_set=(matrices["validation"], targets["validation"]),
                  early_stopping_rounds=35, use_best_model=True)
    cal_center = point.predict(matrices["calibration"])
    cal_quantiles = quantiles.predict(matrices["calibration"])
    low_log = np.minimum(np.min(cal_quantiles, axis=1), cal_center)
    high_log = np.maximum(np.max(cal_quantiles, axis=1), cal_center)
    scores = np.maximum(low_log - targets["calibration"], targets["calibration"] - high_log)
    correction = conformal_adjustment(scores, config.coverage)
    calibration = calibrate_pairs(parts["validation"], point.predict(matrices["validation"]))
    predicted_log, low, high = predict_intervals(point, quantiles, matrices["test"], correction)
    baseline = np.log1p([history.get(r.channel_id, r.sector, r.published_at)["channel_median"] for r in parts["test"]])
    baseline_mean = np.log1p([history.get(r.channel_id, r.sector, r.published_at)["channel_mean"] for r in parts["test"]])
    metrics = regression_metrics(parts["test"], predicted_log, low, high)
    baseline_metrics = regression_metrics(parts["test"], baseline)
    uncertainty = paired_channel_bootstrap(parts["test"], predicted_log, baseline, config.seed)
    by_sector = {}
    for sector in sorted({r.sector for r in parts["test"]}):
        ids = [i for i, r in enumerate(parts["test"]) if r.sector == sector]
        by_sector[sector] = regression_metrics([parts["test"][i] for i in ids], predicted_log[ids], low[ids], high[ids])
    # A research eligibility gate; it is not proof of business/causal impact.
    supported = bool(
        not dataset_audit["synthetic"] and uncertainty["ci95_low"] is not None
        and uncertainty["ci95_low"] > 0 and metrics["mae_log"] < baseline_metrics["mae_log"]
        and metrics["interval_coverage"] >= config.coverage - 0.05
        and len({r.channel_id for r in parts["test"]}) >= 5
    )
    model_id = f"model-{utcnow():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:8]}"
    folder = model_root / model_id
    folder.mkdir(parents=True, exist_ok=False)
    report = {
        "model_id": model_id, "created_at": utcnow().isoformat(),
        "evidence_level": "synthetic_demo" if dataset_audit["synthetic"] else "retrospective",
        "eligible_for_decision": supported, "metric": records[0].metric,
        "horizon_days": records[0].horizon_days, "config": config.model_dump(),
        "audit": dataset_audit, "dataset_sha256": dataset_hash,
        "splits": {name: len(rows) for name, rows in parts.items()},
        "cutoffs": split["cutoffs"], "purged_rows": split["discarded"],
        "duplicates_removed": split["exact_duplicates_removed"],
        "trained_until": max(r.observed_at for r in parts["train"]).isoformat(),
        "inference_not_before": max(r.observed_at for r in parts["test"]).isoformat(),
        "metrics": metrics, "baseline": baseline_metrics,
        "baseline_mean": regression_metrics(parts["test"], baseline_mean),
        "by_sector": by_sector, "paired_channel_bootstrap": uncertainty,
        "pair_calibration": calibration,
        "pair_probability_test": probability_metrics(parts["test"], predicted_log, calibration),
        "conformal_adjustment_log": correction,
        "feature_names": encoder.names, "sectors": sorted({r.sector for r in parts["train"]}),
        "languages": sorted({r.language for r in parts["train"]}),
        "distribution": sorted({r.distribution for r in parts["train"]}),
        "model_family": "CatBoost + MultiQuantile + split conformal",
        "environment": {"python": platform.python_version(), **{
            package: importlib.metadata.version(package)
            for package in ("catboost", "scikit-learn", "numpy", "pydantic")}},
        "limitations": [
            "Dados observacionais; explicações não representam efeitos causais.",
            "Cobertura conformal depende da estabilidade da distribuição e é medida empiricamente.",
            "Pares históricos do mesmo canal/janela não são um experimento aleatório de campanha.",
            "Impulsionamento desconhecido não é inferido a partir das métricas públicas.",
            *dataset_audit["warnings"],
        ],
    }
    point.save_model(str(folder / "point.cbm"))
    quantiles.save_model(str(folder / "quantiles.cbm"))
    encoder.save(folder)
    save_records(folder / "history.jsonl", history_records)
    save_records(folder / "reference.jsonl", parts["train"])
    write_json(folder / "partitions.json", {name: [r.creative_id for r in rows] for name, rows in parts.items()})
    write_json(folder / "test_predictions.json", [
        {"creative_id": r.creative_id, "channel_id": r.channel_id, "sector": r.sector,
         "published_at": r.published_at.isoformat(), "actual": r.target,
         "prediction": float(np.expm1(max(0, predicted_log[i]))),
         "baseline": float(np.expm1(baseline[i])), "low": float(low[i]), "high": float(high[i])}
        for i, r in enumerate(parts["test"])
    ])
    write_json(folder / "report.json", report)
    write_json(folder / "manifest.json", {p.name: sha256_file(p) for p in folder.iterdir() if p.is_file()})
    _track(report)
    return report


def _track(report: dict) -> None:
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return
    try:
        import mlflow
    except ImportError as exc:
        raise ValueError("MLFLOW_TRACKING_URI exige o extra tracking.") from exc
    mlflow.set_experiment("galaxies-creative-intelligence")
    with mlflow.start_run(run_name=report["model_id"]):
        mlflow.log_params({"evidence_level": report["evidence_level"], "metric": report["metric"],
                           "horizon_days": report["horizon_days"], **{
                               key: report["config"][key] for key in ("seed", "depth", "iterations")}})
        mlflow.log_metrics({key: value for key, value in report["metrics"].items() if isinstance(value, (float, int))})
        mlflow.log_dict(report, "report.json")


class ModelBundle:
    def __init__(self, folder: Path):
        manifest = read_json(folder / "manifest.json")
        for filename, digest in manifest.items():
            path = (folder / filename).resolve()
            if not path.is_relative_to(folder.resolve()) or sha256_file(path) != digest:
                raise ValueError("Artefato de modelo inválido ou alterado.")
        self.report = read_json(folder / "report.json")
        self.encoder = FeatureEncoder.load(folder)
        self.point = CatBoostRegressor().load_model(str(folder / "point.cbm"))
        self.quantiles = CatBoostRegressor().load_model(str(folder / "quantiles.cbm"))
        self.history = ChannelHistory(load_records(folder / "history.jsonl"))
        self.references = load_records(folder / "reference.jsonl")
        self.reference_matrix = self.encoder.standardized(
            self.encoder.transform_records(self.references, self.history))

    def predict(self, matrix: np.ndarray):
        return predict_intervals(self.point, self.quantiles, matrix,
                                 self.report["conformal_adjustment_log"])

    def explain(self, vector: np.ndarray) -> list[Driver]:
        shap = self.point.get_feature_importance(Pool(vector.reshape(1, -1)), type="ShapValues")[0][:-1]
        indices = np.argsort(np.abs(shap))[-5:][::-1]
        return [Driver(feature=self.encoder.names[i], label=label(self.encoder.names[i]),
                       contribution_log=float(shap[i]), direction="positive" if shap[i] >= 0 else "negative")
                for i in indices if abs(shap[i]) > 1e-8]

    def comparables(self, vector, sector, as_of, exclude_id: str) -> list[dict]:
        candidates = [i for i, r in enumerate(self.references)
                      if r.sector == sector and r.observed_at < as_of and r.creative_id != exclude_id]
        if not candidates:
            return []
        # Content-only distance avoids presenting "same channel size" as creative similarity.
        indices = [i for i, name in enumerate(self.encoder.names)
                   if name in FEATURES or name.startswith(("text_component", "visual_component"))]
        if not indices:
            return []
        query = self.encoder.standardized(vector)[indices]
        distances = np.mean((self.reference_matrix[candidates][:, indices] - query) ** 2, axis=1)
        nearest = np.argsort(distances)[:3]
        return [{"creative_id": self.references[candidates[j]].creative_id,
                 "caption": self.references[candidates[j]].caption,
                 "observed_metric": self.references[candidates[j]].target,
                 "observed_at": self.references[candidates[j]].observed_at.isoformat(),
                 "distance": float(distances[j])} for j in nearest]
