"""Predeclared rolling-origin model selection, with a separate final time window."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from galaxies.data import split_at, temporal_split, validate_split
from galaxies.schemas import PostRecord, TrainConfig


def implementation_fingerprint() -> str:
    """Freeze training, validation and selection code as well as the input dataset."""
    import importlib.metadata

    digest = hashlib.sha256()
    for name in (
        "data",
        "estimators",
        "features",
        "metrics",
        "model",
        "protocol",
        "research",
        "schemas",
    ):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(f"{name}.py").read_bytes())
    for name in ("numpy", "catboost", "lightgbm", "scikit-learn"):
        digest.update(f"{name}:{importlib.metadata.version(name)}".encode())
    return digest.hexdigest()


def dataset_fingerprint(records: list[PostRecord]) -> str:
    digest = hashlib.sha256()
    for row in sorted(records, key=lambda r: r.creative_id):
        digest.update(
            json.dumps(row.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def make_plan(
    records: list[PostRecord], folds: int = 3
) -> tuple[dict, list[PostRecord], list[dict]]:
    if not 2 <= folds <= 5:
        raise ValueError("Use de 2 a 5 janelas de desenvolvimento.")
    initial = temporal_split(records)
    final_start = datetime.fromisoformat(initial["cutoffs"][-1])
    development = [r for r in records if r.observed_at < final_start]
    dates = sorted({r.published_at for r in development})
    if len(dates) < 40:
        raise ValueError("Histórico curto para seleção temporal: amplie o período observado.")
    boundaries = [
        dates[min(len(dates) - 1, int(len(dates) * f))]
        for f in np.linspace(0.64, 1, folds + 1)[:-1]
    ]
    boundaries.append(final_start)
    splits, specifications = [], []
    for i in range(folds):
        start, end = boundaries[i : i + 2]
        position = dates.index(start)
        width = max(1, int(len(dates) * 0.11))
        cuts = [dates[position - 2 * width], dates[position - width], start]
        split = split_at(development, cuts, test_end=end)
        validate_split(split, development)
        if any(
            len(split["parts"][name]) < minimum
            for name, minimum in (
                ("train", 60),
                ("validation", 20),
                ("calibration", 20),
                ("test", 20),
            )
        ):
            raise ValueError(
                "Janela insuficiente após expurgo: são necessários mais canais/tempo, sem reduzir o isolamento temporal."
            )
        splits.append(split)
        specifications.append(
            {
                "fold": i,
                "cutoffs": split["cutoffs"],
                "evaluation_end": end.isoformat(),
                "ids": {
                    name: [r.creative_id for r in rows] for name, rows in split["parts"].items()
                },
            }
        )
    final_ids = [r.creative_id for r in records if r.published_at >= final_start]
    development_ids = {r.creative_id for r in development}
    assert not set(final_ids) & development_ids
    evaluation_ids = [creative_id for f in specifications for creative_id in f["ids"]["test"]]
    assert len(evaluation_ids) == len(set(evaluation_ids))
    return (
        {
            "protocol_version": 3,
            "implementation_sha256": implementation_fingerprint(),
            "dataset_sha256": dataset_fingerprint(records),
            "final_start": final_start.isoformat(),
            "protected_final_ids": final_ids,
            "development_rows": len(development),
            "folds": specifications,
            "selection_rule": "One-standard-error heuristic on macro channel pair accuracy; retain candidates within 10% of the best macro log-MAE among ranking-competitive models, then prefer simplicity. Apply the same rule to the context-only reference. Folds are temporally dependent: this is not a confidence guarantee.",
            "final_test_policy": "One selected candidate and preselected strongest context-only reference; no final-test-driven tuning or promotion.",
        },
        development,
        splits,
    )


def candidate_catalog(iterations: int = 180, seed: int = 42) -> dict[str, TrainConfig]:
    common = {"iterations": iterations, "seed": seed, "channel_weighting": "balanced"}
    definitions = {
        "context_ridge": {"family": "ridge", "feature_blocks": ["context"]},
        "context_catboost": {"family": "catboost", "feature_blocks": ["context"], "depth": 4},
        "context_lightgbm": {"family": "lightgbm", "feature_blocks": ["context"], "depth": 4},
        "ridge": {"family": "ridge"},
        "ridge_regularized": {"family": "ridge", "ridge_alpha": 100},
        "ridge_residual": {"family": "ridge", "target_mode": "channel_residual"},
        "catboost": {"family": "catboost", "depth": 4},
        "catboost_residual": {"family": "catboost", "target_mode": "channel_residual", "depth": 4},
        "catboost_deeper": {"family": "catboost", "depth": 6, "l2_regularization": 10},
        "lightgbm": {"family": "lightgbm", "depth": 4},
        "lightgbm_residual": {"family": "lightgbm", "target_mode": "channel_residual", "depth": 4},
        "blend_residual": {"family": "blend", "target_mode": "channel_residual", "depth": 4},
    }
    return {name: TrainConfig(**common, **values) for name, values in definitions.items()}


def select_candidate(rows: list[dict], configurations: dict[str, TrainConfig]) -> dict:
    summaries = []
    expected_folds = {r["fold"] for r in rows}
    complexity = {"ridge": 1, "catboost": 2, "lightgbm": 2, "blend": 3}
    for name, config in configurations.items():
        runs = [r for r in rows if r["candidate"] == name]
        if {r["fold"] for r in runs} != expected_folds or len(runs) != len(expected_folds):
            raise ValueError("Candidatos precisam dos mesmos folds, sem resultados faltantes.")
        values = [r["metrics"]["macro_channel_pair_accuracy"] for r in runs]
        if any(v is None for v in values):
            raise ValueError("Não há pares suficientes para escolher um modelo de comparação.")
        summaries.append(
            {
                "candidate": name,
                "pair_accuracy_mean": float(np.mean(values)),
                "pair_accuracy_sd": float(np.std(values, ddof=1)),
                "pair_accuracy_se": float(np.std(values, ddof=1) / np.sqrt(len(values))),
                "worst_fold_pair_accuracy": float(min(values)),
                "macro_mae_log": float(
                    np.mean([r["metrics"]["macro_channel_mae_log"] for r in runs])
                ),
                "interval_score_log": float(
                    np.mean([r["metrics"]["interval_score_log"] for r in runs])
                ),
                "complexity": complexity[config.family],
            }
        )

    def balanced_choice(candidates):
        best = max(candidates, key=lambda r: r["pair_accuracy_mean"])
        threshold = best["pair_accuracy_mean"] - best["pair_accuracy_se"]
        competitive = [r for r in candidates if r["pair_accuracy_mean"] >= threshold]
        maximum_error = 1.10 * min(r["macro_mae_log"] for r in competitive)
        competitive = [r for r in competitive if r["macro_mae_log"] <= maximum_error + 1e-12]
        winner = min(
            competitive,
            key=lambda r: (
                r["complexity"],
                r["macro_mae_log"],
                r["interval_score_log"],
                r["candidate"],
            ),
        )
        return winner, threshold, maximum_error

    selected, threshold, maximum_error = balanced_choice(summaries)
    context, _, _ = balanced_choice(
        [r for r in summaries if configurations[r["candidate"]].feature_blocks == ["context"]]
    )
    return {
        "selected": selected["candidate"],
        "strongest_context": context["candidate"],
        "competitive_threshold": threshold,
        "macro_mae_log_ceiling": maximum_error,
        "maximum_relative_error_degradation": 0.10,
        "leaderboard": sorted(summaries, key=lambda r: -r["pair_accuracy_mean"]),
    }
