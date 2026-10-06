"""Decision metrics on comparable pairs, with channel-level uncertainty."""

from collections import defaultdict

import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr

from galaxies.schemas import PostRecord


def comparable_pairs(records: list[PostRecord]) -> list[tuple[int, int]]:
    groups = defaultdict(list)
    for index, record in enumerate(records):
        window = int(record.published_at.timestamp() // (14 * 86400))
        groups[(record.channel_id, window)].append(index)
    return [(a, b) for indices in groups.values() for pos, a in enumerate(indices)
            for b in indices[pos + 1:] if records[a].target != records[b].target]


def pair_accuracy(records: list[PostRecord], predicted: np.ndarray) -> float | None:
    pairs = comparable_pairs(records)
    if not pairs:
        return None
    return float(np.mean([
        0.5 if predicted[a] == predicted[b]
        else (predicted[a] > predicted[b]) == (records[a].target > records[b].target)
        for a, b in pairs
    ]))


def regression_metrics(records: list[PostRecord], prediction_log: np.ndarray,
                       low: np.ndarray | None = None, high: np.ndarray | None = None) -> dict:
    y = np.array([r.target for r in records], dtype=float)
    actual_log = np.log1p(y)
    predicted = np.expm1(np.clip(prediction_log, 0, 50))
    spearman = float(spearmanr(actual_log, prediction_log).statistic) if (
        len(np.unique(actual_log)) > 1 and len(np.unique(prediction_log)) > 1
    ) else None
    by_channel = defaultdict(list)
    for i, record in enumerate(records):
        by_channel[record.channel_id].append(i)
    intra = []
    for ids in by_channel.values():
        if len(ids) >= 3 and len(np.unique(prediction_log[ids])) > 1 and len(np.unique(y[ids])) > 1:
            intra.append(float(spearmanr(y[ids], prediction_log[ids]).statistic))
    grouped = defaultdict(list)
    for i, record in enumerate(records):
        grouped[(record.channel_id, int(record.published_at.timestamp() // (14 * 86400)))].append(i)
    regrets = []
    for ids in grouped.values():
        if len(ids) >= 2:
            chosen = ids[int(np.argmax(prediction_log[ids]))]
            best = float(np.max(y[ids]))
            regrets.append((best - y[chosen]) / max(best, 1))
    result = {
        "rows": len(records), "mae_log": float(np.mean(np.abs(actual_log - prediction_log))),
        "mae": float(np.mean(np.abs(y - predicted))),
        "mape_nonzero": float(np.mean(np.abs((y[y > 0] - predicted[y > 0]) / y[y > 0]))) if np.any(y > 0) else None,
        "mape_excluded_zero_rows": int(sum(y == 0)), "spearman_global": spearman,
        "spearman_within_channel": float(np.mean(intra)) if intra else None,
        "pair_accuracy": pair_accuracy(records, prediction_log),
        "pair_count": len(comparable_pairs(records)),
        "mean_selection_regret": float(np.mean(regrets)) if regrets else None,
    }
    if low is not None and high is not None:
        result.update({"interval_coverage": float(np.mean((y >= low) & (y <= high))),
                       "mean_interval_width": float(np.mean(high - low)),
                       "median_interval_width": float(np.median(high - low))})
    return result


def paired_channel_bootstrap(records: list[PostRecord], prediction: np.ndarray,
                              baseline: np.ndarray, seed: int = 42) -> dict:
    groups = defaultdict(list)
    for i, record in enumerate(records):
        groups[record.channel_id].append(i)
    deltas = []
    for indices in groups.values():
        subset = [records[i] for i in indices]
        accuracy, reference = pair_accuracy(subset, prediction[indices]), pair_accuracy(subset, baseline[indices])
        if accuracy is not None and reference is not None:
            deltas.append(accuracy - reference)
    if len(deltas) < 5:
        return {"channels": len(deltas), "mean_delta": None, "ci95_low": None, "ci95_high": None}
    rng = np.random.default_rng(seed)
    means = rng.choice(deltas, size=(1000, len(deltas)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"channels": len(deltas), "mean_delta": float(np.mean(deltas)),
            "ci95_low": float(low), "ci95_high": float(high),
            "method": "paired bootstrap of macro channel pair-accuracy differences; 1000 resamples"}


def calibrate_pairs(records: list[PostRecord], predictions: np.ndarray) -> dict:
    """Symmetric monotone Platt calibration on a separate chronological validation set."""
    from scipy.optimize import minimize_scalar

    pairs = comparable_pairs(records)
    if len(pairs) < 30 or len({r.channel_id for r in records}) < 5:
        return {"available": False, "pairs": len(pairs), "slope": None}
    delta = np.array([predictions[a] - predictions[b] for a, b in pairs])
    truth = np.array([float(records[a].target > records[b].target) for a, b in pairs])
    def loss(slope):
        probabilities = np.clip(expit(slope * delta), 1e-6, 1 - 1e-6)
        return -np.mean(truth * np.log(probabilities) + (1 - truth) * np.log1p(-probabilities))
    fit = minimize_scalar(loss, bounds=(0, 20), method="bounded")
    return {"available": True, "pairs": len(pairs), "slope": float(fit.x),
            "method": "symmetric Platt calibration; validation pairs within channel/14-day window"}


def probability_metrics(records: list[PostRecord], predictions: np.ndarray, calibration: dict):
    pairs = comparable_pairs(records)
    if not calibration.get("available") or not pairs:
        return {"brier_score": None, "pairs": len(pairs)}
    probabilities = np.array([expit(calibration["slope"] * (predictions[a] - predictions[b])) for a, b in pairs])
    truth = np.array([float(records[a].target > records[b].target) for a, b in pairs])
    return {"brier_score": float(np.mean((probabilities - truth) ** 2)), "pairs": len(pairs)}
