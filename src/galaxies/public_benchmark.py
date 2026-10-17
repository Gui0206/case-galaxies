"""Real-data research with explicit calendar targets; never a production certification.

Public exports lack original video files and verified pre-publication content versions.
Keep these records outside PostRecord: neither synthetic timestamps nor sector mappings
are invented to satisfy the application's stricter data contract.
"""

import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer

from galaxies.data import ChannelHistory
from galaxies.estimators import PointEstimator, sample_weights
from galaxies.io import sha256_file, utcnow, write_json
from galaxies.metrics import paired_channel_bootstrap, regression_metrics
from galaxies.protocol import candidate_catalog, implementation_fingerprint, select_candidate
from galaxies.schemas import TrainConfig

SOURCES = {
    "kenjee": {
        "ref": "kenjee/ken-jee-youtube-data",
        "version": 2,
        "files": ["Aggregated_Metrics_By_Video.csv", "Video_Performance_Over_Time.csv"],
        "sha256": None,
    },
    "trending": {
        "ref": "datasnaek/youtube-new",
        "version": 115,
        "files": ["USvideos.csv", "CAvideos.csv", "GBvideos.csv"],
        "sha256": "9f59324bd23ebd95aa16f854de1e0a6299bacf988b30c5e7db5f1e4216d8deda",
    },
}


def download_public(root: Path, name: str) -> dict:
    """Download versioned, public research exports; never execute upstream code."""
    spec = SOURCES[name]
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "dataset.zip"
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        response = client.get(f"https://www.kaggle.com/api/v1/datasets/view/{spec['ref']}")
        response.raise_for_status()
        write_json(folder / "source_metadata.json", response.json())
        if not archive.exists():
            url = f"https://www.kaggle.com/api/v1/datasets/download/{spec['ref']}?datasetVersionNumber={spec['version']}"
            partial = archive.with_suffix(".partial")
            with client.stream("GET", url) as response, partial.open("wb") as handle:
                response.raise_for_status()
                for chunk in response.iter_bytes(1024 * 1024):
                    handle.write(chunk)
            partial.rename(archive)
    digest = sha256_file(archive)
    if spec["sha256"] and digest != spec["sha256"]:
        raise ValueError("O arquivo público mudou; audite a origem antes de continuar.")
    # Select known members, no archive-provided paths or scripts are executed.
    with zipfile.ZipFile(archive) as bundle:
        for name in spec["files"]:
            matches = [p for p in bundle.namelist() if p.endswith(name)]
            if len(matches) != 1:
                raise ValueError(f"Arquivo esperado ausente/ambíguo: {name}")
            (folder / name).write_bytes(bundle.read(matches[0]))
    manifest = {
        "source": f"https://www.kaggle.com/datasets/{spec['ref']}",
        "version": spec["version"],
        "archive_sha256": digest,
        "files": {n: sha256_file(folder / n) for n in spec["files"]},
    }
    write_json(folder / "download.json", manifest)
    return manifest


def _text(value):
    return "" if pd.isna(value) else str(value)


def load_kenjee(folder: Path) -> tuple[pd.DataFrame, dict]:
    aggregate = pd.read_csv(folder / SOURCES["kenjee"]["files"][0])
    aggregate.columns = aggregate.columns.str.replace("\xad", "")
    daily = pd.read_csv(folder / SOURCES["kenjee"]["files"][1])
    aggregate["published_at"] = pd.to_datetime(
        aggregate["Video publish time"].str.replace("Sept", "Sep"),
        format="%b %d, %Y",
        errors="coerce",
        utc=True,
    )
    daily["day"] = pd.to_datetime(
        daily["Date"].str.replace("Sept", "Sep"), format="%d %b %Y", utc=True
    )
    if daily.duplicated(["External Video ID", "day"]).any():
        raise ValueError("Observações diárias duplicadas; não some silenciosamente.")
    joined = daily.merge(
        aggregate[["Video", "published_at"]].dropna(),
        left_on="External Video ID",
        right_on="Video",
        validate="many_to_one",
    )
    joined["age"] = (joined.day - joined.published_at).dt.days
    rows = []
    for video, group in joined[joined.age.between(0, 6)].groupby("Video"):
        if set(group.age) != set(range(7)):
            continue  # Missing daily rows are NOT silently treated as zero views.
        group = group.sort_values("day")
        first = group.iloc[0]
        rows.append(
            {
                "creative_id": video,
                "channel_id": "ken-jee",
                "category": "Education",
                "published_at": first.published_at,
                "observed_at": first.published_at + pd.Timedelta(days=8),
                "caption": _text(first["Video Title"]),
                "duration_s": float(first["Video Length"]),
                "target": int(group.Views.sum()),
            }
        )
    return pd.DataFrame(rows), {
        "raw_videos": int(aggregate.published_at.notna().sum()),
        "complete_calendar_week": len(rows),
        "target": "views during publication calendar date and following six dates",
        "availability": "conservative publication date + 8 days, allowing 1 day reporting lag",
        "limitations": [
            "One creator; English data-science content, not the three Galaxies verticals.",
            "Calendar dates, not an exact 168-hour measurement; source timezone unspecified.",
            "Titles and lengths exported retrospectively; original versions cannot be verified.",
            "No original video, transcript, image embeddings, rubric or paid-distribution history.",
        ],
    }


def load_trending(folder: Path) -> tuple[pd.DataFrame, dict]:
    frames = []
    for region in ("US", "CA", "GB"):
        part = pd.read_csv(folder / f"{region}videos.csv")
        part["region"] = region
        frames.append(part)
    data = pd.concat(frames, ignore_index=True)
    data["published_at"] = pd.to_datetime(data.publish_time, utc=True)
    data["day"] = pd.to_datetime(data.trending_date, format="%y.%d.%m", utc=True)
    data["age"] = (data.day - data.published_at.dt.floor("D")).dt.days
    # Lexical country order is fixed, never choose the snapshot with the largest count.
    data = data.sort_values(["day", "region", "video_id"])
    first = data.drop_duplicates("video_id")
    targets = data[data.age == 7].drop_duplicates("video_id").set_index("video_id")
    rows = []
    for row in first[first.age.between(0, 1)].itertuples():
        if row.video_id not in targets.index or row.video_id == "#NAME?":
            continue
        target = targets.loc[row.video_id]
        if target.video_error_or_removed or not _text(row.channel_title):
            continue
        rows.append(
            {
                "creative_id": row.video_id,
                "channel_id": _text(row.channel_title).strip().casefold(),
                "category": str(row.category_id),
                "published_at": row.published_at,
                "observed_at": target.day + pd.Timedelta(days=1),
                "caption": _text(row.title),
                "duration_s": np.nan,
                "target": int(target.views),
            }
        )
    return pd.DataFrame(rows), {
        "raw_rows": len(data),
        "unique_videos": int(data.video_id.nunique()),
        "eligible_videos": len(rows),
        "target": "cumulative views in snapshot dated publication calendar date + 7 days",
        "availability": "end of snapshot calendar date (conservative)",
        "limitations": [
            "Selected on appearing in Trending early AND on calendar day 7; severe survivor/popularity bias.",
            "Snapshot time absent: target age approximately 6-8 days, not exact 168 hours.",
            "Title/category from first snapshot within one calendar day; pre-publication version unverified.",
            "Channel title is the available grouping identifier; renames/homonyms cannot be resolved.",
            "US/CA/GB snapshots deduplicated across regions; no channel size or current engagement features.",
            "No video/audio or contemporaneous thumbnail files; not an end-to-end multimodal test.",
        ],
    }


def prepare_public(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame = frame.sort_values(["published_at", "creative_id"]).reset_index(drop=True).copy()
    if frame.creative_id.duplicated().any():
        raise ValueError("IDs de vídeo duplicados.")
    if not np.isfinite(frame.target).all() or (frame.target < 0).any():
        raise ValueError("Alvos precisam ser contagens finitas não negativas.")
    if (frame.observed_at <= frame.published_at).any():
        raise ValueError("Resultado precisa ser posterior à publicação.")
    # Conservative same-title duplicate proxy within channel; report its limitations.
    normalized = frame.caption.map(lambda s: re.sub(r"\W+", " ", s.casefold()).strip())
    frame["content_group"] = [
        hashlib.sha256(f"{channel}:{title or video}".encode()).hexdigest()
        for channel, title, video in zip(
            frame.channel_id, normalized, frame.creative_id, strict=True
        )
    ]
    before = len(frame)
    frame = frame.drop_duplicates("content_group").reset_index(drop=True)
    return frame, {
        "rows": len(frame),
        "channels": int(frame.channel_id.nunique()),
        "same_channel_title_duplicates_removed": before - len(frame),
        "period_start": frame.published_at.min().isoformat(),
        "period_end": frame.published_at.max().isoformat(),
        "deduplication_limit": "ID plus normalized channel/title; no perceptual video hashes available",
    }


def history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Reuse the production point-in-time history implementation, without fake sectors."""
    records = [
        SimpleNamespace(
            channel_id=r.channel_id,
            sector=r.category,
            published_at=r.published_at,
            observed_at=r.observed_at,
            target=r.target,
        )
        for r in frame.itertuples()
    ]
    history = ChannelHistory(records)
    rows = []
    for row in frame.itertuples():
        h = history.get(row.channel_id, row.category, row.published_at)
        rows.append(
            {
                "log_channel_median": np.log1p(h["pooled_median"]),
                "log_raw_channel_median": np.log1p(h["channel_median"]),
                "log_channel_mean": np.log1p(h["channel_mean"]),
                "history_count": h["history_count"],
                "history_log_std": h["history_log_std"],
                "history_log_ewma": h["history_log_ewma"],
                "history_log_trend_per_day": h["history_log_trend_per_day"],
                "history_age_days": h["history_age_days"],
                "weekday_sin": np.sin(row.published_at.weekday() * 2 * np.pi / 7),
                "weekday_cos": np.cos(row.published_at.weekday() * 2 * np.pi / 7),
            }
        )
    return pd.DataFrame(rows, index=frame.index)


class PublicEncoder:
    """Adapt real available columns to the shared PointEstimator; train-only projections."""

    def __init__(self, with_content: bool):
        self.with_content = with_content
        self.vectorizer = HashingVectorizer(
            n_features=1024, alternate_sign=False, ngram_range=(1, 2)
        )
        self.svd = None

    def fit(self, frame, context):
        self.categories = sorted(frame.category.unique())
        if self.with_content:
            hashed = self.vectorizer.transform(frame.caption)
            if hashed.nnz:
                self.svd = TruncatedSVD(n_components=min(12, len(frame) - 1), random_state=42).fit(
                    hashed
                )
        matrix = self.transform(frame, context)
        self.medians = np.array([np.nanmedian(c) if np.isfinite(c).any() else 0 for c in matrix.T])
        filled = np.where(np.isfinite(matrix), matrix, self.medians)
        self.scales = np.maximum(np.std(filled, axis=0), 0.1)
        return matrix

    def transform(self, frame, context):
        columns = {name: context[name].to_numpy() for name in context.columns}
        for category in self.categories:
            columns[f"category_{category}"] = (frame.category == category).to_numpy(dtype=float)
        if self.with_content:
            columns["log_duration"] = np.log1p(frame.duration_s.to_numpy())
            columns["caption_words"] = frame.caption.str.split().str.len().to_numpy()
            columns["caption_chars"] = frame.caption.str.len().to_numpy()
            if self.svd:
                text = self.svd.transform(self.vectorizer.transform(frame.caption))
                columns.update({f"text_component_{i}": text[:, i] for i in range(text.shape[1])})
        self.names = list(columns)
        return np.column_stack(list(columns.values())).astype(float)

    def standardized(self, matrix):
        return (np.where(np.isfinite(matrix), matrix, self.medians) - self.medians) / self.scales


def _records(frame):
    return [
        SimpleNamespace(channel_id=r.channel_id, published_at=r.published_at, target=r.target)
        for r in frame.itertuples()
    ]


def _parts(frame, cuts, end=None):
    boundaries = pd.to_datetime(cuts, utc=True)
    result = {}
    for i, name in enumerate(("train", "validation", "calibration", "test")):
        mask = pd.Series(True, index=frame.index)
        if i:
            mask &= frame.published_at >= boundaries[i - 1]
        if i < 3:
            mask &= (frame.published_at < boundaries[i]) & (frame.observed_at < boundaries[i])
        if end is not None:
            mask &= frame.published_at < pd.Timestamp(end)
        result[name] = frame.index[mask].tolist()
    if any(
        len(result[k]) < n
        for k, n in (("train", 40), ("validation", 10), ("calibration", 10), ("test", 10))
    ):
        raise ValueError(
            f"Partições insuficientes após expurgo: { {k: len(v) for k, v in result.items()} }"
        )
    return result


def benchmark_plan(frame, audit, sources):
    dates = sorted(frame.published_at.unique())
    final_cuts = [pd.Timestamp(dates[int(len(dates) * p)]).isoformat() for p in (0.55, 0.70, 0.85)]
    development = frame[frame.observed_at < pd.Timestamp(final_cuts[-1])]
    dev_dates = sorted(development.published_at.unique())
    folds = []
    # Two expanding origins; publication windows are disjoint. Small real exports
    # cannot support the default production protocol's minimum sizes in 3 windows.
    for start, stop in ((0.70, 0.85), (0.85, 1.0)):
        cuts = [
            pd.Timestamp(dev_dates[int(len(dev_dates) * p)]).isoformat()
            for p in (start - 0.30, start - 0.15, start)
        ]
        end = (
            pd.Timestamp(dev_dates[int(len(dev_dates) * stop)]).isoformat()
            if stop < 1
            else final_cuts[-1]
        )
        parts = _parts(development, cuts, end)
        folds.append({"cuts": cuts, "end": end, "parts": parts})
    return {
        "protocol": "public-calendar-benchmark-v1",
        "created_at": utcnow().isoformat(),
        "implementation_sha256": _code_fingerprint(),
        "sources": sources,
        "audit": audit,
        "frame_sha256": _frame_hash(frame),
        "final_cuts": final_cuts,
        "final_parts": _parts(frame, final_cuts),
        "development_folds": folds,
        "candidates": {k: v.model_dump() for k, v in candidate_catalog(180).items()},
        "primary_existing_candidate": "catboost_residual",
        "selection": "same balanced rule as production protocol; two development origins; frozen before final",
        "metrics": "log MAE, count MAE, median multiplicative error, within-factor-2, same-channel/14-day pairs, 90% intervals",
        "forbidden_inputs": [
            "current video views/likes/comments/CTR/retention",
            "future channel totals",
            "row order as publication time",
        ],
        "eligible_for_production": False,
    }


def _frame_hash(frame):
    return hashlib.sha256(frame.to_json(orient="records", date_format="iso").encode()).hexdigest()


def _code_fingerprint():
    return hashlib.sha256(
        (implementation_fingerprint() + Path(__file__).read_text()).encode()
    ).hexdigest()


@dataclass
class Fit:
    config: TrainConfig
    encoder: PublicEncoder
    point: PointEstimator
    quantiles: CatBoostRegressor
    correction: float

    def predict(self, frame, context):
        x = self.encoder.transform(frame, context)
        raw = self.point.predict(x)
        quantiles = self.quantiles.predict(x) + self.point.offset(x)[:, None]
        low = np.minimum(np.min(quantiles, axis=1), raw) - self.correction
        high = np.maximum(np.max(quantiles, axis=1), raw) + self.correction
        return np.clip(raw, 0, 50), np.expm1(np.clip(low, 0, 50)), np.expm1(np.clip(high, 0, 50))


def _fit(frame, context, parts, config):
    ids = parts
    encoder = PublicEncoder(config.feature_blocks != ["context"])
    x = {"train": encoder.fit(frame.loc[ids["train"]], context.loc[ids["train"]])}
    for name in ("validation", "calibration"):
        x[name] = encoder.transform(frame.loc[ids[name]], context.loc[ids[name]])
    y = {name: np.log1p(frame.loc[ids[name], "target"].to_numpy()) for name in x}
    weights = {
        name: sample_weights(_records(frame.loc[ids[name]]), config.channel_weighting) for name in x
    }
    point = PointEstimator(config, encoder).fit(
        x["train"],
        y["train"],
        x["validation"],
        y["validation"],
        weights["train"],
        weights["validation"],
    )
    quantiles = CatBoostRegressor(
        iterations=config.iterations,
        depth=config.depth,
        learning_rate=config.learning_rate,
        random_seed=config.seed,
        loss_function="MultiQuantile:alpha=0.05,0.95",
        verbose=False,
        thread_count=2,
        allow_writing_files=False,
    )
    quantiles.fit(
        Pool(x["train"], y["train"] - point.offset(x["train"]), weight=weights["train"]),
        eval_set=Pool(
            x["validation"],
            y["validation"] - point.offset(x["validation"]),
            weight=weights["validation"],
        ),
        early_stopping_rounds=35,
        use_best_model=True,
    )
    raw = point.predict(x["calibration"])
    q = quantiles.predict(x["calibration"]) + point.offset(x["calibration"])[:, None]
    scores = np.maximum(
        np.minimum(q.min(axis=1), raw) - y["calibration"],
        y["calibration"] - np.maximum(q.max(axis=1), raw),
    )
    order = min(len(scores) - 1, int(np.ceil((len(scores) + 1) * 0.9)) - 1)
    correction = max(0.0, float(np.sort(scores)[order]))
    return Fit(config, encoder, point, quantiles, correction)


def _metrics(frame, prediction, low=None, high=None):
    result = regression_metrics(_records(frame), prediction, low, high)
    error = np.abs(np.log1p(frame.target.to_numpy()) - prediction)
    result["median_multiplicative_error_plus_one"] = float(np.exp(np.median(error)))
    result["fraction_within_factor_2_plus_one"] = float(np.mean(error <= np.log(2)))
    return result


def error_gain_uncertainty(frame, prediction, reference, seed=42):
    """Paired macro log-error differences; channels or 28-day blocks for one creator."""
    actual = np.log1p(frame.target.to_numpy())
    improvement = np.abs(actual - reference) - np.abs(actual - prediction)
    if frame.channel_id.nunique() >= 5:
        groups = frame.channel_id.to_numpy()
        unit = "channel"
    else:
        groups = (frame.published_at.astype("int64") // (28 * 86400 * 10**9)).to_numpy()
        unit = "28-day publication block; one creator, no between-channel inference"
    differences = pd.Series(improvement).groupby(groups).mean().to_numpy()
    result = {
        "resampling_unit": unit,
        "units": len(differences),
        "mean_macro_log_error_reduction": float(differences.mean()),
    }
    if len(differences) < 5:
        return {**result, "ci95_low": None, "ci95_high": None}
    rng = np.random.default_rng(seed)
    boot = rng.choice(differences, size=(2000, len(differences)), replace=True).mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return {**result, "ci95_low": float(low), "ci95_high": float(high)}


def develop(frame, folder: Path, plan):
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "plan.json", plan)
    context = history_features(frame)
    configs = {name: TrainConfig.model_validate(v) for name, v in plan["candidates"].items()}
    results = []
    for i, fold in enumerate(plan["development_folds"]):
        for name, config in configs.items():
            print(f"Desenvolvimento {i + 1}/2 · {name}", flush=True)
            fit = _fit(frame, context, fold["parts"], config)
            ids = fold["parts"]["test"]
            prediction, low, high = fit.predict(frame.loc[ids], context.loc[ids])
            results.append(
                {
                    "fold": i,
                    "candidate": name,
                    "metrics": _metrics(frame.loc[ids], prediction, low, high),
                }
            )
    selection = select_candidate(results, configs)
    report = {"results": results, "selection": selection, "final_test_used": False}
    write_json(folder / "development.json", report)
    write_json(
        folder / "selection.json",
        {
            "selection": selection,
            "plan_sha256": sha256_file(folder / "plan.json"),
            "development_sha256": sha256_file(folder / "development.json"),
        },
    )
    return report


def evaluate_final(frame, folder: Path):
    plan = json.loads((folder / "plan.json").read_text())
    frozen = json.loads((folder / "selection.json").read_text())
    if (
        _frame_hash(frame) != plan["frame_sha256"]
        or _code_fingerprint() != plan["implementation_sha256"]
    ):
        raise ValueError("Dados/código mudaram após o plano.")
    for name in ("plan", "development"):
        if sha256_file(folder / f"{name}.json") != frozen[f"{name}_sha256"]:
            raise ValueError("Artefatos de seleção alterados.")
    original = json.loads((folder / "development.json").read_text())["selection"]
    if frozen["selection"] != original:
        raise ValueError("Seleção foi alterada.")
    # One final opening per dataset and target in the public benchmark workspace.
    ledger = folder.parent / f"final-opened-{plan['frame_sha256']}.json"
    with ledger.open("x") as handle:
        json.dump({"started_at": utcnow().isoformat(), "folder": str(folder)}, handle)
    parts = plan["final_parts"]
    context = history_features(frame)
    test_ids = parts["test"]
    test = frame.loc[test_ids]
    names = list(
        dict.fromkeys(["catboost_residual", original["selected"], original["strongest_context"]])
    )
    predictions, metrics = {}, {}
    for name in names:
        print(f"Teste final congelado · {name}", flush=True)
        fit = _fit(frame, context, parts, TrainConfig.model_validate(plan["candidates"][name]))
        prediction, low, high = fit.predict(test, context.loc[test_ids])
        predictions[name] = prediction
        metrics[name] = _metrics(test, prediction, low, high)
        metrics[name]["by_history"] = {}
        for label, mask in {
            "no_history": context.loc[test_ids].history_count == 0,
            "one_or_more": context.loc[test_ids].history_count > 0,
            "at_least_five": context.loc[test_ids].history_count >= 5,
        }.items():
            if mask.any():
                metrics[name]["by_history"][label] = _metrics(
                    test[mask], prediction[mask], low[mask], high[mask]
                )
        model_folder = folder / name
        model_folder.mkdir()
        fit.point.save(model_folder)
        fit.quantiles.save_model(str(model_folder / "quantiles.cbm"))
        write_json(
            model_folder / "encoder.json",
            {
                "names": fit.encoder.names,
                "categories": fit.encoder.categories,
                "with_content": fit.encoder.with_content,
                "correction": fit.correction,
            },
        )
        np.savez_compressed(
            model_folder / "encoder.npz",
            medians=fit.encoder.medians,
            scales=fit.encoder.scales,
            text_components=fit.encoder.svd.components_ if fit.encoder.svd else [],
        )
    for name, column in (
        ("channel_median", "log_raw_channel_median"),
        ("pooled_median", "log_channel_median"),
    ):
        predictions[name] = context.loc[test_ids, column].to_numpy()
        metrics[name] = _metrics(test, predictions[name])
    selected = original["selected"]
    strong = original["strongest_context"]
    comparison = paired_channel_bootstrap(
        _records(test), predictions[selected], predictions[strong]
    )
    export = test[["creative_id", "channel_id", "published_at", "observed_at", "target"]].copy()
    export["history_count"] = context.loc[test_ids, "history_count"]
    for name, prediction in predictions.items():
        export[name] = np.expm1(prediction)
    export.to_csv(folder / "test_predictions.csv", index=False)
    report = {
        "evidence_level": "real_public_retrospective_metadata_only",
        "eligible_for_production": False,
        "selected": selected,
        "strongest_context": strong,
        "original_candidate": "catboost_residual",
        "partitions": {name: len(ids) for name, ids in parts.items()},
        "test_period": [test.published_at.min().isoformat(), test.published_at.max().isoformat()],
        "metrics": metrics,
        "content_pair_gain_vs_context": comparison,
        "log_error_gain_uncertainty_vs_context": error_gain_uncertainty(
            test, predictions[selected], predictions[strong]
        ),
        "log_error_gain_uncertainty_vs_channel_median": error_gain_uncertainty(
            test, predictions[selected], predictions["channel_median"]
        ),
        "count_error_reduction_vs_channel_median": 1
        - metrics[selected]["mae"] / metrics["channel_median"]["mae"],
        "log_error_reduction_vs_channel_median": 1
        - metrics[selected]["mae_log"] / metrics["channel_median"]["mae_log"],
        "log_error_reduction_vs_strong_context": 1
        - metrics[selected]["mae_log"] / metrics[strong]["mae_log"],
        "completed_at": utcnow().isoformat(),
        "limitations": plan["audit"]["source_audit"]["limitations"],
    }
    write_json(folder / "final.json", report)
    return report
