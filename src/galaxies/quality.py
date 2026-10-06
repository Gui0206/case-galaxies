"""Operational summaries and human-versus-extractor agreement, without synthetic claims."""

from collections import Counter

import numpy as np
from sklearn.metrics import cohen_kappa_score

from galaxies.config import Settings
from galaxies.io import read_json
from galaxies.jobs import Store
from galaxies.prospective import report as prospective_report
from galaxies.schemas import FEATURES


def extraction_agreement(reference: list[dict], extracted: list[dict]) -> dict:
    by_id = {r["creative_id"]: r for r in extracted}
    metrics = {}
    for name in sorted(FEATURES):
        pairs = [
            (row["features"][name], by_id[row["creative_id"]]["features"][name])
            for row in reference
            if row["creative_id"] in by_id
            and row.get("features", {}).get(name) is not None
            and by_id[row["creative_id"]].get("features", {}).get(name) is not None
        ]
        if not pairs:
            continue
        truth, prediction = np.array(pairs, dtype=float).T
        if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
            raise ValueError("Anotações precisam conter valores finitos.")
        value = {"n": len(pairs), "mae": float(np.mean(np.abs(truth - prediction)))}
        if name in ("demonstration", "cta"):
            if not np.isin(np.r_[truth, prediction], [0, 1]).all():
                raise ValueError(f"Atributo binário precisa ser 0 ou 1: {name}")
            value["agreement"] = float(np.mean(truth == prediction))
            kappa = (
                cohen_kappa_score(truth, prediction)
                if len(np.unique(np.r_[truth, prediction])) > 1
                else None
            )
            value["kappa"] = float(kappa) if kappa is not None and np.isfinite(kappa) else None
        metrics[name] = value
    if not metrics:
        raise ValueError("Sem atributos anotados comparáveis.")
    return {
        "reference_rows": len(reference),
        "extracted_rows": len(extracted),
        "by_feature": metrics,
        "note": "Avalie uma amostra cega e representativa; concordância não prova utilidade preditiva.",
    }


def operational_report(settings: Settings) -> dict:
    paths = sorted((settings.workspace / "runs").glob("*.json"), key=lambda p: p.stat().st_mtime)[
        -1000:
    ]
    runs = [read_json(p) for p in paths]
    duration = [r["duration_ms"] for r in runs if "duration_ms" in r]
    flags = Counter(r["prediction"]["confidence_flag"] for r in runs)
    levels = Counter(r["prediction"]["evidence_level"] for r in runs)
    missing = Counter(name for r in runs for name, value in r["features"].items() if value is None)
    return {
        "window": "last 1000 persisted predictions",
        "predictions": len(runs),
        "evidence_levels": dict(levels),
        "confidence": dict(flags),
        "latency_ms_p50": float(np.median(duration)) if duration else None,
        "latency_ms_p95": float(np.quantile(duration, 0.95)) if duration else None,
        "missing_feature_counts": dict(missing),
        "prospective": prospective_report(Store(settings)),
        "note": "Ausências e sinais de domínio são diagnósticos; não substituem teste formal de drift.",
    }
