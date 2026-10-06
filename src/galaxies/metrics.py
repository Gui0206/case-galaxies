"""Decision metrics on comparable pairs, with channel-level uncertainty."""

from collections import defaultdict

import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

from galaxies.schemas import PostRecord


def comparable_pairs(records: list[PostRecord]) -> list[tuple[int, int]]:
    groups = defaultdict(list)
    for index, record in enumerate(records):
        window = int(record.published_at.timestamp() // (14 * 86400))
        groups[(record.channel_id, window)].append(index)
    return [
        (a, b)
        for indices in groups.values()
        for pos, a in enumerate(indices)
        for b in indices[pos + 1 :]
        if records[a].target != records[b].target
    ]


def pair_accuracy(records: list[PostRecord], predicted: np.ndarray) -> float | None:
    pairs = comparable_pairs(records)
    if not pairs:
        return None
    return float(
        np.mean(
            [
                0.5
                if predicted[a] == predicted[b]
                else (predicted[a] > predicted[b]) == (records[a].target > records[b].target)
                for a, b in pairs
            ]
        )
    )


def regression_metrics(
    records: list[PostRecord],
    prediction_log: np.ndarray,
    low: np.ndarray | None = None,
    high: np.ndarray | None = None,
    coverage: float = 0.9,
) -> dict:
    prediction_log = np.clip(np.asarray(prediction_log, dtype=float), 0, 50)
    y = np.array([r.target for r in records], dtype=float)
    actual_log = np.log1p(y)
    predicted = np.expm1(np.clip(prediction_log, 0, 50))
    spearman = (
        float(spearmanr(actual_log, prediction_log).statistic)
        if (len(np.unique(actual_log)) > 1 and len(np.unique(prediction_log)) > 1)
        else None
    )
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
        "rows": len(records),
        "mae_log": float(np.mean(np.abs(actual_log - prediction_log))),
        "mae": float(np.mean(np.abs(y - predicted))),
        "mape_nonzero": float(np.mean(np.abs((y[y > 0] - predicted[y > 0]) / y[y > 0])))
        if np.any(y > 0)
        else None,
        "mape_excluded_zero_rows": int(sum(y == 0)),
        "spearman_global": spearman,
        "spearman_within_channel": float(np.mean(intra)) if intra else None,
        "pair_accuracy": pair_accuracy(records, prediction_log),
        "pair_count": len(comparable_pairs(records)),
        "mean_selection_regret": float(np.mean(regrets)) if regrets else None,
        "macro_channel_mae_log": float(
            np.mean(
                [
                    np.mean(np.abs(actual_log[ids] - prediction_log[ids]))
                    for ids in by_channel.values()
                ]
            )
        ),
        "macro_channel_pair_accuracy": macro_pair_accuracy(records, prediction_log),
    }
    if low is not None and high is not None:
        result.update(
            {
                "interval_coverage": float(np.mean((y >= low) & (y <= high))),
                "mean_interval_width": float(np.mean(high - low)),
                "median_interval_width": float(np.median(high - low)),
                "interval_score_log": interval_score(
                    actual_log, np.log1p(low), np.log1p(high), coverage
                ),
            }
        )
    return result


def macro_pair_accuracy(records: list[PostRecord], prediction: np.ndarray) -> float | None:
    groups = defaultdict(list)
    for i, row in enumerate(records):
        groups[row.channel_id].append(i)
    values = [pair_accuracy([records[i] for i in ids], prediction[ids]) for ids in groups.values()]
    available = [v for v in values if v is not None]
    return float(np.mean(available)) if available else None


def interval_score(y, low, high, coverage=0.9) -> float:
    """Proper interval score: sharpness plus penalties for misses, in the supplied scale."""
    if not 0 < coverage < 1:
        raise ValueError("Cobertura precisa estar entre 0 e 1.")
    alpha = 1 - coverage
    return float(
        np.mean(high - low + 2 / alpha * (np.maximum(low - y, 0) + np.maximum(y - high, 0)))
    )


def paired_channel_bootstrap(
    records: list[PostRecord], prediction: np.ndarray, baseline: np.ndarray, seed: int = 42
) -> dict:
    groups = defaultdict(list)
    for i, record in enumerate(records):
        groups[record.channel_id].append(i)
    deltas = []
    for indices in groups.values():
        subset = [records[i] for i in indices]
        accuracy, reference = (
            pair_accuracy(subset, prediction[indices]),
            pair_accuracy(subset, baseline[indices]),
        )
        if accuracy is not None and reference is not None:
            deltas.append(accuracy - reference)
    if len(deltas) < 5:
        return {"channels": len(deltas), "mean_delta": None, "ci95_low": None, "ci95_high": None}
    rng = np.random.default_rng(seed)
    means = rng.choice(deltas, size=(1000, len(deltas)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "channels": len(deltas),
        "mean_delta": float(np.mean(deltas)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "method": "paired bootstrap of macro channel pair-accuracy differences; 1000 resamples",
    }


def performance_bands(
    records: list[PostRecord], predictions: np.ndarray, reference_log: np.ndarray
) -> dict:
    """Descriptive relative bands with thresholds fixed before seeing the test labels."""
    base = np.maximum(np.expm1(reference_log), 1)
    truth = np.digitize(np.array([r.target for r in records]) / base, [0.75, 1.25, 3])
    predicted = np.digitize(np.expm1(np.clip(predictions, 0, 50)) / base, [0.75, 1.25, 3])
    return {
        "thresholds_relative_to_channel": [0.75, 1.25, 3],
        "labels": ["below", "typical", "above", "exceptional"],
        "macro_f1": float(
            f1_score(truth, predicted, labels=[0, 1, 2, 3], average="macro", zero_division=0)
        ),
        "note": "Exceptional is relative performance, not a universal definition of virality.",
    }


def calibrate_pairs(records: list[PostRecord], predictions: np.ndarray) -> dict:
    """Symmetric monotone Platt calibration on held-out chronological calibration data."""
    from scipy.optimize import minimize_scalar

    predictions = np.clip(predictions, 0, 50)
    pairs = comparable_pairs(records)
    channels = [records[a].channel_id for a, _ in pairs]
    if len(pairs) < 30 or len(set(channels)) < 5:
        return {"available": False, "pairs": len(pairs), "slope": None}
    from collections import Counter

    counts = Counter(channels)
    weights = np.array([1 / counts[c] for c in channels])
    weights /= weights.sum()
    delta = np.array([predictions[a] - predictions[b] for a, b in pairs])
    truth = np.array([float(records[a].target > records[b].target) for a, b in pairs])

    def loss(slope):
        probabilities = np.clip(expit(slope * delta), 1e-6, 1 - 1e-6)
        return -np.sum(
            weights * (truth * np.log(probabilities) + (1 - truth) * np.log1p(-probabilities))
        )

    fit = minimize_scalar(loss, bounds=(0, 20), method="bounded")
    return {
        "available": True,
        "pairs": len(pairs),
        "slope": float(fit.x),
        "channels": len(counts),
        "channel_weighting": "equal total weight per channel",
        "method": "symmetric Platt calibration; held-out calibration pairs within channel/14-day window",
    }


def probability_metrics(records: list[PostRecord], predictions: np.ndarray, calibration: dict):
    predictions = np.clip(predictions, 0, 50)
    pairs = comparable_pairs(records)
    if not calibration.get("available") or not pairs:
        return {"brier_score": None, "pairs": len(pairs)}
    probabilities = np.array(
        [expit(calibration["slope"] * (predictions[a] - predictions[b])) for a, b in pairs]
    )
    truth = np.array([float(records[a].target > records[b].target) for a, b in pairs])
    errors = (probabilities - truth) ** 2
    reliability = []
    bins = np.minimum((probabilities * 10).astype(int), 9)
    for bucket in range(10):
        ids = bins == bucket
        if ids.any():
            reliability.append(
                {
                    "count": int(ids.sum()),
                    "probability": float(probabilities[ids].mean()),
                    "observed": float(truth[ids].mean()),
                }
            )
    selective = []
    confidence = np.maximum(probabilities, 1 - probabilities)
    correct = (probabilities > 0.5) == truth
    for threshold in (0.5, 0.6, 0.65, 0.7, 0.8, 0.9):
        accepted = confidence > threshold
        selective.append(
            {
                "threshold": threshold,
                "accepted_pairs": int(accepted.sum()),
                "acceptance_rate": float(accepted.mean()),
                "accuracy": float(correct[accepted].mean()) if accepted.any() else None,
            }
        )
    return {
        "brier_score": float(errors.mean()),
        "pairs": len(pairs),
        "reliability": reliability,
        "selective_accuracy": selective,
    }
