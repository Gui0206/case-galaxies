"""Prepublication reach study: point-in-time channel history and residual content models.

This research adapter deliberately does not promote a historical public sample to production.
It never uses early popularity, subscribers measured later, or the target video's engagement.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from galaxies.io import read_json, sha256_file, utcnow, write_json

ROOT = Path(".runtime/public/beyond-views")
RUN = Path(".runtime/reports/reach")
DAY = 86400
FINAL_START = "2016-08-20"
BASELINES = [
    "median_5",
    "median_20",
    "mean_20",
    "logmean_20",
    "shrunk_2",
    "shrunk_5",
    "median_50",
    "logmean_50",
]
CONTEXT = [
    "log_history_count",
    "history_age_days",
    "gap_days",
    "log_duration",
    "duration_relative",
    "median_5",
    "median_20",
    "mean_20",
    "logmean_20",
    "median_50",
    "logmean_50",
    "shrunk_2",
    "shrunk_5",
    "history_std",
    "history_q10",
    "history_q90",
    "history_trend",
    "category_prior",
    "recent_duration",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]
TEXT_STATS = ["title_length", "title_words", "title_digits", "title_question", "title_exclamation"]
CAT = ["category", "language", "definition"]
STYLE_TERMS = re.compile(
    r"\b(makeup|make up|maquiagem|maquillaje|skincare|skin care|cosmetics|cosmeticos|"
    r"lipstick|batom|eyeshadow|sombra|hair tutorial|penteado|nail art|unhas|"
    r"fashion|lookbook|outfits?|moda|try on|try-on|styling|wardrobe)\b",
    re.IGNORECASE,
)
CANDIDATES = {
    "context_mae4": {"bundle": "context", "depth": 4, "loss": "MAE"},
    "context_mae6": {"bundle": "context", "depth": 6, "loss": "MAE"},
    "context_rmse6": {"bundle": "context", "depth": 6, "loss": "RMSE"},
    "text_mae4": {"bundle": "text", "depth": 4, "loss": "MAE"},
    "text_mae6": {"bundle": "text", "depth": 6, "loss": "MAE"},
    "text_rmse6": {"bundle": "text", "depth": 6, "loss": "RMSE"},
    "retrieval_mae4": {"bundle": "retrieval", "depth": 4, "loss": "MAE"},
    "retrieval_mae6": {"bundle": "retrieval", "depth": 6, "loss": "MAE"},
    "retrieval_rmse6": {"bundle": "retrieval", "depth": 6, "loss": "RMSE"},
    "linear_text_10": {"bundle": "linear", "alpha": 10},
    "linear_text_100": {"bundle": "linear", "alpha": 100},
    "semantic_mae4": {"bundle": "semantic", "depth": 4, "loss": "MAE"},
    "semantic_mae6": {"bundle": "semantic", "depth": 6, "loss": "MAE"},
    "semantic_retrieval_mae4": {"bundle": "semantic_retrieval", "depth": 4, "loss": "MAE"},
    "semantic_retrieval_mae6": {"bundle": "semantic_retrieval", "depth": 6, "loss": "MAE"},
    "blend_text_retrieval": {"bundle": "blend", "weight": 0.5},
}
# Second development round: retained original candidates, improved anchoring and simple
# content-based historical estimators. The untouched final window is unchanged.
for _bundle in ("context", "retrieval", "semantic_retrieval"):
    for _depth in (4, 6):
        CANDIDATES[f"channel_{_bundle}_mae{_depth}"] = {
            "bundle": _bundle,
            "depth": _depth,
            "loss": "MAE",
            "anchor": "logmean_20",
        }
for _alpha in (100, 1000):
    CANDIDATES[f"channel_linear_{_alpha}"] = {
        "bundle": "linear",
        "alpha": _alpha,
        "anchor": "logmean_20",
    }
for _bundle in ("lexical_rule", "semantic_rule"):
    for _shrink in (0.25, 0.5, 0.75):
        CANDIDATES[f"{_bundle}_{int(_shrink * 100)}"] = {
            "bundle": _bundle,
            "shrink": _shrink,
            "anchor": "logmean_20",
        }


def epoch(value):
    return pd.Timestamp(value, tz="UTC").timestamp()


def reserved_channel(channel):
    return int(hashlib.sha256(channel.encode()).hexdigest()[:8], 16) % 10 == 0


def duration_seconds(value):
    match = re.fullmatch(r"P(?:(\d+)D)?T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?", value)
    if not match:
        raise ValueError("invalid_duration")
    d, h, m, s = (float(x or 0) for x in match.groups())
    result = d * DAY + h * 3600 + m * 60 + s
    if not 0 < result <= DAY:
        raise ValueError("invalid_duration")
    return result


def parse_record(record):
    """Require explicit daily counts; a missing date is never synthesized as zero."""
    info, insights = record["snippet"], record["insights"]
    published = pd.Timestamp(info["publishedAt"])
    if published.tzinfo is None:
        raise ValueError("missing_timezone")
    published = published.tz_convert("UTC")
    if not epoch("2016-07-01") <= published.timestamp() < epoch("2016-09-01"):
        raise ValueError("outside_source_period")
    if insights["startDate"] != published.date().isoformat():
        raise ValueError("reporting_start_not_publication_date")
    days = [int(x) for x in insights["days"].split(",")]
    views = [int(x) for x in insights["dailyView"].split(",")]
    if len(days) != len(views) or len(set(days)) != len(days) or days != sorted(days):
        raise ValueError("invalid_daily_series")
    daily = dict(zip(days, views, strict=True))
    if not all(day in daily for day in range(7)):
        raise ValueError("missing_week_day")
    if any(x < 0 for x in views):
        raise ValueError("negative_daily_count")
    title = info.get("title", "").strip()
    channel = info["channelId"]
    if not title or not channel or str(info["categoryId"]) not in {"2", "26"}:
        raise ValueError("invalid_metadata")
    target = sum(daily[d] for d in range(7))
    return {
        "creative_id": record["id"],
        "channel_id": channel,
        "published_at": published.timestamp(),
        "available_at": published.normalize().timestamp() + 9 * DAY,
        "title": title,
        "duration": duration_seconds(record["contentDetails"]["duration"]),
        "category": str(info["categoryId"]),
        "language": str(info.get("detectLang", "NA")),
        "definition": str(record["contentDetails"].get("definition", "unknown")),
        "target": target,
        "target_log": float(np.log1p(target)),
        "reserved_channel": reserved_channel(channel),
        "style_title_subset": str(info["categoryId"]) == "26" and bool(STYLE_TERMS.search(title)),
    }


def prepare(root=ROOT):
    if (root / "cohort.parquet").exists():
        raise ValueError("Prepared data already exists; do not overwrite the registered cohort")
    acquisition = read_json(root / "acquisition.json")
    records, counts = [], Counter()
    for source in acquisition["files"]:
        if source["file"] not in {"autos.json", "howto.json"}:
            raise ValueError("Unexpected source member")
        path = root / source["file"]
        if sha256_file(path) != source["sha256"]:
            raise ValueError("Source changed after acquisition")
        with path.open() as stream:
            for line in stream:
                counts["source_rows"] += 1
                try:
                    record = parse_record(json.loads(line))
                except (KeyError, ValueError, TypeError) as exc:
                    counts[f"excluded:{str(exc)[:80]}"] += 1
                    continue
                records.append(record)
    frame = pd.DataFrame(records).sort_values(["published_at", "creative_id"])
    counts["duplicate_ids"] = int(frame.creative_id.duplicated().sum())
    frame = frame.drop_duplicates("creative_id")
    normalized = frame.title.map(lambda s: " ".join(re.findall(r"\w+", s.casefold())))
    frame["duplicate_key"] = frame.channel_id + ":" + normalized
    counts["duplicate_channel_title"] = int(frame.duplicate_key.duplicated().sum())
    frame = (
        frame.drop_duplicates("duplicate_key").drop(columns="duplicate_key").reset_index(drop=True)
    )
    frame.to_parquet(root / "cohort.parquet", index=False)
    # This audit contains availability and counts only, never final outcome summaries.
    audit = {
        "created_at": utcnow(),
        "counts": dict(counts),
        "eligible_rows": len(frame),
        "channels": int(frame.channel_id.nunique()),
        "category_counts": frame.category.value_counts().to_dict(),
        "published_min": pd.to_datetime(frame.published_at.min(), unit="s", utc=True),
        "published_max": pd.to_datetime(frame.published_at.max(), unit="s", utc=True),
        "cohort_sha256": sha256_file(root / "cohort.parquet"),
        "acquisition_sha256": sha256_file(root / "acquisition.json"),
    }
    write_json(root / "audit.json", audit)
    return audit


def history_features(frame):
    """A strict event-time join; within-test history updates use ONLY already mature outcomes.

    Reserved channels cannot contribute labels to category priors or model fitting. Their
    diagnostic is a genuine no-local-history test, even if earlier source rows exist.
    """
    frame = frame.sort_values(["published_at", "creative_id"]).reset_index(drop=True).copy()
    ordered = np.argsort(frame.available_at.to_numpy(), kind="stable")
    available = frame.available_at.to_numpy()
    y = frame.target_log.to_numpy()
    histories, categories = defaultdict(list), defaultdict(list)
    rows, neighbors, pointer = [], [], 0
    for item in frame.itertuples(index=False):
        while pointer < len(frame) and available[ordered[pointer]] < item.published_at:
            j = int(ordered[pointer])
            previous = frame.iloc[j]
            if not previous.reserved_channel:
                histories[previous.channel_id].append(j)
                categories[previous.category].append(y[j])
            pointer += 1
        # Availability order and publication order agree under the nine-day convention,
        # except for same-day ties, for which the stable input order is preserved.
        prior = histories[item.channel_id][-50:]
        neighbors.append(prior.copy())
        cat = categories[item.category]
        # Bounded recent reference so a large early category cannot dominate forever.
        cat_prior = float(np.median(cat[-5000:])) if cat else 0.0
        values = y[prior]
        recent = values[-20:]
        med = float(np.median(recent)) if len(recent) else cat_prior
        logmean = float(np.mean(recent)) if len(recent) else cat_prior
        rawmean = float(np.log1p(np.mean(np.expm1(recent)))) if len(recent) else cat_prior
        n = len(recent)
        past = frame.iloc[prior]
        d = float(np.median(np.log1p(past.duration))) if n else np.log1p(item.duration)
        date = pd.Timestamp(item.published_at, unit="s", tz="UTC")
        row = {
            "history_count": len(histories[item.channel_id]),
            "log_history_count": np.log1p(len(histories[item.channel_id])),
            "history_age_days": (item.published_at - past.published_at.iloc[0]) / DAY if n else 0,
            "gap_days": (item.published_at - past.published_at.iloc[-1]) / DAY if n else 0,
            "log_duration": np.log1p(item.duration),
            "duration_relative": np.log1p(item.duration) - d,
            "median_5": float(np.median(values[-5:])) if n else cat_prior,
            "median_20": med,
            "logmean_20": logmean,
            "median_50": float(np.median(values)) if n else cat_prior,
            "logmean_50": float(np.mean(values)) if n else cat_prior,
            "mean_20": rawmean,
            "shrunk_2": (n * med + 2 * cat_prior) / (n + 2),
            "shrunk_5": (n * med + 5 * cat_prior) / (n + 5),
            "history_std": float(np.std(recent)) if n else 0,
            "history_q10": float(np.quantile(recent, 0.1)) if n else cat_prior,
            "history_q90": float(np.quantile(recent, 0.9)) if n else cat_prior,
            "history_trend": float(np.median(values[-3:]) - med) if n else 0,
            "category_prior": cat_prior,
            "recent_duration": d,
            "hour_sin": np.sin(date.hour * np.pi / 12),
            "hour_cos": np.cos(date.hour * np.pi / 12),
            "weekday_sin": np.sin(date.dayofweek * 2 * np.pi / 7),
            "weekday_cos": np.cos(date.dayofweek * 2 * np.pi / 7),
            "title_length": np.log1p(len(item.title)),
            "title_words": np.log1p(len(item.title.split())),
            "title_digits": sum(c.isdigit() for c in item.title) / len(item.title),
            "title_question": float("?" in item.title),
            "title_exclamation": float("!" in item.title),
        }
        rows.append(row)
    return pd.concat([frame, pd.DataFrame(rows)], axis=1), neighbors


def weights(frame):
    frequency = frame.channel_id.value_counts()
    result = frame.channel_id.map(lambda c: 1 / frequency[c]).to_numpy()
    return result / result.mean()


def macro_error(frame, prediction):
    errors = np.abs(frame.target_log.to_numpy() - prediction)
    return float(pd.Series(errors).groupby(frame.channel_id.to_numpy()).mean().mean())


class ReachEncoder:
    """Title vocabulary, IDF, SVD and scales fit on training rows only."""

    def fit_transform(self, frame, train, neighbors, semantic=None):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            max_features=30000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.vectorizer.fit(frame.iloc[train].title)
        text = self.vectorizer.transform(frame.title)
        dimensions = min(48, len(train) - 1, text.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=dimensions, n_iter=5, random_state=42)
        self.svd.fit(text[train])
        self.levels = {column: sorted(frame.iloc[train][column].unique()) for column in CAT}
        self.context_names = CONTEXT + [f"{c}:{v}" for c in CAT for v in self.levels[c]]
        self.scaler = StandardScaler().fit(self.context(frame)[train])
        self.text_scaler = StandardScaler().fit(frame[TEXT_STATS].to_numpy(dtype=float)[train])
        self.semantic_pca = None
        if semantic is not None:
            self.semantic_pca = PCA(
                n_components=min(32, len(train) - 1, semantic.shape[1]),
                svd_solver="randomized",
                random_state=42,
            ).fit(semantic[train])
        return self.transform(frame, neighbors, semantic, text)

    def context(self, frame):
        category = np.column_stack(
            [
                (frame[column] == value).to_numpy(dtype=float)
                for column in CAT
                for value in self.levels[column]
            ]
        )
        return np.column_stack([frame[CONTEXT].to_numpy(dtype=float), category])

    def transform(self, frame, neighbors, semantic=None, text=None):
        if text is None:
            text = self.vectorizer.transform(frame.title)
        dense = self.svd.transform(text)
        context = self.context(frame)
        stats = frame[TEXT_STATS].to_numpy(dtype=float)
        # Similarity is based on a train-only vocabulary; neighbor outcomes obey event time.
        anchor = frame.shrunk_2.to_numpy()
        retrieval = np.zeros((len(frame), 4))
        lexical_rule = np.zeros((len(frame), 1))
        rule_anchor = frame.logmean_20.to_numpy()
        y = frame.target_log.to_numpy()
        for i, prior in enumerate(neighbors):
            if not prior:
                continue
            similarity = (text[prior] @ text[i].T).toarray().ravel()
            order = np.argsort(-similarity, kind="stable")[:5]
            ids, sim = np.asarray(prior)[order], similarity[order]
            strength = sim**2
            if strength.sum() > 0:
                lexical_rule[i, 0] = np.dot(strength, y[ids] - rule_anchor[i]) / strength.sum()
            retrieval[i] = [
                np.dot(strength, y[ids] - anchor[i]) / (1 + strength.sum()),
                float(sim.max()),
                float(strength.sum()),
                np.dot(strength, np.log1p(frame.duration.to_numpy()[ids])) / (1 + strength.sum()),
            ]
        text_dense = np.column_stack([context, stats, dense])
        matrices = {
            "context": context,
            "text": text_dense,
            "retrieval": np.column_stack([text_dense, retrieval]),
            "lexical_rule": lexical_rule,
            "linear": sparse.hstack(
                [
                    sparse.csr_matrix(self.scaler.transform(context)),
                    sparse.csr_matrix(self.text_scaler.transform(stats)),
                    text,
                ],
                format="csr",
            ),
        }
        if self.semantic_pca is not None:
            if semantic is None:
                raise ValueError("Saved encoder requires the same frozen semantic representations")
            semantic_dense = self.semantic_pca.transform(semantic)
            semantic_neighbors = np.zeros((len(frame), 3))
            semantic_rule = np.zeros((len(frame), 1))
            for i, prior in enumerate(neighbors):
                if not prior:
                    continue
                similarity = np.maximum(0, semantic[prior] @ semantic[i])
                order = np.argsort(-similarity, kind="stable")[:5]
                ids, sim = np.asarray(prior)[order], similarity[order]
                strength = sim**4
                if strength.sum() > 0:
                    semantic_rule[i, 0] = np.dot(strength, y[ids] - rule_anchor[i]) / strength.sum()
                semantic_neighbors[i] = [
                    np.dot(strength, y[ids] - anchor[i]) / (1 + strength.sum()),
                    float(sim.max()),
                    float(strength.sum()),
                ]
            matrices["semantic"] = np.column_stack([context, stats, semantic_dense])
            matrices["semantic_retrieval"] = np.column_stack(
                [matrices["semantic"], semantic_neighbors]
            )
            matrices["semantic_rule"] = semantic_rule
        return matrices

    def save(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        write_json(
            folder / "encoder.json",
            {
                "vocabulary": self.vectorizer.vocabulary_,
                "parameters": {
                    "analyzer": "char_wb",
                    "ngram_range": [3, 5],
                    "sublinear_tf": True,
                },
                "levels": self.levels,
                "context_names": self.context_names,
            },
        )
        np.savez_compressed(
            folder / "encoder.npz",
            idf=self.vectorizer.idf_,
            components=self.svd.components_,
            mean=self.scaler.mean_,
            scale=self.scaler.scale_,
            text_mean=self.text_scaler.mean_,
            text_scale=self.text_scaler.scale_,
        )
        if self.semantic_pca is not None:
            np.savez_compressed(
                folder / "semantic_pca.npz",
                mean=self.semantic_pca.mean_,
                components=self.semantic_pca.components_,
            )

    @classmethod
    def load(cls, folder):
        info = read_json(folder / "encoder.json")
        arrays = np.load(folder / "encoder.npz", allow_pickle=False)
        encoder = cls()
        encoder.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            dtype=np.float32,
            vocabulary=info["vocabulary"],
        )
        encoder.vectorizer.idf_ = arrays["idf"]
        encoder.svd = TruncatedSVD(n_components=len(arrays["components"]))
        encoder.svd.components_ = arrays["components"]
        encoder.svd.n_features_in_ = arrays["components"].shape[1]
        encoder.levels, encoder.context_names = info["levels"], info["context_names"]
        for name, prefix in (("scaler", ""), ("text_scaler", "text_")):
            scaler = StandardScaler()
            scaler.mean_, scaler.scale_ = arrays[f"{prefix}mean"], arrays[f"{prefix}scale"]
            scaler.var_, scaler.n_features_in_ = scaler.scale_**2, len(scaler.mean_)
            setattr(encoder, name, scaler)
        encoder.semantic_pca = None
        if (folder / "semantic_pca.npz").exists():
            data = np.load(folder / "semantic_pca.npz", allow_pickle=False)
            encoder.semantic_pca = PCA(n_components=len(data["components"]))
            encoder.semantic_pca.components_, encoder.semantic_pca.mean_ = (
                data["components"],
                data["mean"],
            )
            encoder.semantic_pca.explained_variance_ = np.ones(len(data["components"]))
            encoder.semantic_pca.n_features_in_ = data["components"].shape[1]
        return encoder


def fit_candidate(name, matrices, frame, train, predict, folder=None):
    spec = CANDIDATES[name]
    if spec["bundle"] == "blend":
        raise ValueError("A blend must use its declared component predictions")
    x = matrices[spec["bundle"]]
    anchor = frame[spec.get("anchor", "shrunk_2")].to_numpy()
    if spec["bundle"].endswith("_rule"):
        if folder:
            write_json(folder / f"{name}.json", spec)
        return np.maximum(0, anchor[predict] + spec["shrink"] * x[predict, 0])
    residual = frame.target_log.to_numpy()[train] - anchor[train]
    w = weights(frame.iloc[train])
    if spec["bundle"] == "linear":
        model = Ridge(alpha=spec["alpha"], solver="lsqr").fit(x[train], residual, sample_weight=w)
        if folder:
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                folder / f"{name}.npz", coef=model.coef_, intercept=model.intercept_
            )
    else:
        model = CatBoostRegressor(
            iterations=500,
            depth=spec["depth"],
            learning_rate=0.035,
            loss_function=spec["loss"],
            l2_leaf_reg=10,
            random_seed=42,
            thread_count=4,
            verbose=False,
            allow_writing_files=False,
        ).fit(x[train], residual, sample_weight=w)
        if folder:
            folder.mkdir(parents=True, exist_ok=True)
            model.save_model(str(folder / f"{name}.cbm"))
    return np.maximum(0, anchor[predict] + model.predict(x[predict]))


def predict_saved_candidate(name, matrices, frame, rows, folder):
    """Execute the saved numerical model without fitting or reading outcome files."""
    spec = CANDIDATES[name]
    if spec["bundle"] == "blend":
        return 0.5 * predict_saved_candidate(
            "linear_text_100", matrices, frame, rows, folder
        ) + 0.5 * predict_saved_candidate("retrieval_mae6", matrices, frame, rows, folder)
    matrix = matrices[spec["bundle"]][rows]
    if spec["bundle"].endswith("_rule"):
        if read_json(folder / f"{name}.json") != spec:
            raise ValueError("Saved historical-comparable rule changed")
        return np.maximum(0, frame[spec["anchor"]].to_numpy()[rows] + spec["shrink"] * matrix[:, 0])
    if spec["bundle"] == "linear":
        values = np.load(folder / f"{name}.npz", allow_pickle=False)
        residual = np.asarray(matrix @ values["coef"]).ravel() + float(values["intercept"])
    else:
        residual = CatBoostRegressor().load_model(str(folder / f"{name}.cbm")).predict(matrix)
    return np.maximum(0, frame[spec.get("anchor", "shrunk_2")].to_numpy()[rows] + residual)


def fingerprint():
    digest = hashlib.sha256(Path(__file__).read_bytes())
    for library in ("numpy", "pandas", "scikit-learn", "catboost", "scipy"):
        digest.update(f"{library}:{importlib.metadata.version(library)}".encode())
    return digest.hexdigest()


def data_for_study(root=ROOT):
    audit = read_json(root / "audit.json")
    if audit["cohort_sha256"] != sha256_file(root / "cohort.parquet"):
        raise ValueError("Cohort changed after source audit")
    return history_features(pd.read_parquet(root / "cohort.parquet"))


def encode_titles(root=ROOT, weights_path=Path(".runtime/weights/language")):
    """Content-only frozen multilingual encoder; no engagement, descriptions or IDs as input."""
    import torch
    from sentence_transformers import SentenceTransformer

    frame = (
        pd.read_parquet(root / "cohort.parquet")
        .sort_values(["published_at", "creative_id"])
        .reset_index(drop=True)
    )
    identity = {
        "cohort_sha256": sha256_file(root / "cohort.parquet"),
        "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "revision": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
        "weights": {
            str(p.relative_to(weights_path)): sha256_file(p)
            for p in sorted(weights_path.rglob("*"))
            if p.is_file() and not any(x.startswith(".") for x in p.relative_to(weights_path).parts)
        },
        "sentence_transformers": importlib.metadata.version("sentence-transformers"),
        "torch": importlib.metadata.version("torch"),
        "normalize": True,
        "input": "title only",
    }
    if not identity["weights"]:
        raise ValueError("Provision the pinned language model using the documented weights command")
    manifest_path = root / "semantic.json"
    if manifest_path.exists():
        current = read_json(manifest_path)
        if current["identity"] != identity or current["sha256"] != sha256_file(
            root / "semantic.npy"
        ):
            raise ValueError("Semantic cache identity changed")
        return current
    progress_path = root / "semantic-progress.json"
    progress = (
        read_json(progress_path) if progress_path.exists() else {"identity": identity, "done": 0}
    )
    if progress["identity"] != identity:
        raise ValueError("Incomplete semantic cache belongs to a different source/model")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(str(weights_path), device=device, local_files_only=True)
    size = model.get_sentence_embedding_dimension()
    temporary = root / "semantic.partial.npy"
    matrix = np.lib.format.open_memmap(
        temporary,
        mode="r+" if temporary.exists() else "w+",
        dtype=np.float32,
        shape=(len(frame), size),
    )
    for start in range(progress["done"], len(frame), 2048):
        end = min(start + 2048, len(frame))
        matrix[start:end] = model.encode(
            frame.title.iloc[start:end].tolist(),
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        matrix.flush()
        progress["done"] = end
        write_json(progress_path, progress)
        print(f"Encoded titles {end}/{len(frame)}", flush=True)
    del matrix
    temporary.replace(root / "semantic.npy")
    manifest = {
        "identity": identity,
        "rows": len(frame),
        "dimensions": size,
        "sha256": sha256_file(root / "semantic.npy"),
        "created_at": utcnow(),
    }
    write_json(manifest_path, manifest)
    return manifest


def provision_language(weights_path=Path(".runtime/weights/language")):
    from huggingface_hub import snapshot_download

    snapshot_download(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "tokenizer*",
            "sentencepiece*",
            "*.txt",
            "README.md",
        ],
        ignore_patterns=["onnx/*", "openvino/*"],
        local_dir=weights_path,
        max_workers=2,
    )
    return {"language_weights": str(weights_path)}


def load_semantic(root, frame):
    if not (root / "semantic.json").exists():
        if any(v["bundle"].startswith("semantic") for v in CANDIDATES.values()):
            raise ValueError(
                "Run encode before development; semantic candidates require a frozen cache"
            )
        return None
    manifest = read_json(root / "semantic.json")
    if manifest["identity"]["cohort_sha256"] != sha256_file(root / "cohort.parquet") or manifest[
        "sha256"
    ] != sha256_file(root / "semantic.npy"):
        raise ValueError("Semantic cache does not match frozen cohort/hash")
    matrix = np.load(root / "semantic.npy", allow_pickle=False, mmap_mode="r")
    if matrix.shape != (len(frame), manifest["dimensions"]) or not np.isfinite(matrix).all():
        raise ValueError("Invalid semantic representations")
    return matrix


def indices(frame, start, stop=None, train=False, warm=True):
    mask = ~frame.reserved_channel
    if warm:
        mask &= frame.history_count >= 3
    if train:
        mask &= frame.available_at < epoch(start)
    else:
        mask &= frame.published_at >= epoch(start)
        if stop:
            mask &= frame.published_at < epoch(stop)
    return np.flatnonzero(mask)


def develop(root=ROOT, run=RUN):
    if run.exists():
        raise ValueError("Development requires a new directory; preserve previous evidence")
    frame, neighbors = data_for_study(root)
    semantic = load_semantic(root, frame)
    folds = []
    for start, stop in [("2016-07-25", "2016-08-01"), ("2016-08-01", "2016-08-08")]:
        train = indices(frame, start, train=True)
        validation = indices(frame, start, stop)
        if len(train) < 100 or len(validation) < 100:
            raise ValueError("Insufficient development data after maturity and history filters")
        folds.append(
            {
                "start": start,
                "stop": stop,
                "train": train.tolist(),
                "validation": validation.tolist(),
            }
        )
    plan = {
        "protocol": "prepublication-reach-v2",
        "created_at": utcnow(),
        "code_sha256": fingerprint(),
        "audit": read_json(root / "audit.json"),
        "semantic_sha256": sha256_file(root / "semantic.json") if semantic is not None else None,
        "folds": folds,
        "candidates": CANDIDATES,
        "baselines": BASELINES,
        "final_start": FINAL_START,
        "calibration_start": "2016-08-08",
        "calibration_stop": "2016-08-11",
        "selection": "Minimum macro-channel MAE in log over concatenated disjoint development folds; within 1% prefer lower depth, then error, then lexical name. Baseline and context selected independently.",
        "success": "At least 5% relative macro MAE reduction with positive 95% channel-bootstrap lower gain vs BOTH selected history and context; >=500 videos and >=50 channels. Ranking separately requires positive lower gain vs BOTH context and history, >=200 pairs and >=30 channels.",
        "production_eligible": False,
    }
    write_json(run / "plan.json", plan)
    saved, outcomes = [], []
    for fold in folds:
        train, valid = np.array(fold["train"]), np.array(fold["validation"])
        print(
            f"Development {fold['start']}: train={len(train)}, validation={len(valid)}", flush=True
        )
        encoder = ReachEncoder()
        matrices = encoder.fit_transform(frame, train, neighbors, semantic)
        part = frame.iloc[valid][["creative_id", "channel_id", "category", "target_log"]].copy()
        for name in BASELINES:
            part[name] = frame.iloc[valid][name].to_numpy()
        metrics = {}
        for name, spec in CANDIDATES.items():
            if spec["bundle"] == "blend":
                part[name] = 0.5 * part["linear_text_100"] + 0.5 * part["retrieval_mae6"]
            else:
                part[name] = fit_candidate(name, matrices, frame, train, valid)
            metrics[name] = macro_error(frame.iloc[valid], part[name].to_numpy())
            print(f"  {name}: macro MAE log={metrics[name]:.4f}", flush=True)
        saved.append(part)
        outcomes.append(
            {
                "start": fold["start"],
                "train": len(train),
                "validation": len(valid),
                "scores": metrics,
            }
        )
    predictions = pd.concat(saved, ignore_index=True)
    scores = {
        name: macro_error(predictions, predictions[name].to_numpy())
        for name in BASELINES + list(CANDIDATES)
    }

    def choose(names):
        best = min(scores[name] for name in names)
        options = [n for n in names if scores[n] <= best * 1.01]
        return min(options, key=lambda n: (CANDIDATES.get(n, {}).get("depth", 0), scores[n], n))

    selection = {
        "created_at": utcnow(),
        "baseline": min(BASELINES, key=scores.get),
        "context": choose([n for n, v in CANDIDATES.items() if v["bundle"] == "context"]),
        "content": choose([n for n, v in CANDIDATES.items() if v["bundle"] != "context"]),
        "scores": scores,
        "plan_sha256": sha256_file(run / "plan.json"),
    }
    selection["lexical_control"] = choose(
        [
            n
            for n, v in CANDIDATES.items()
            if v["bundle"] != "context" and not v["bundle"].startswith("semantic")
        ]
    )
    semantic_names = [n for n, v in CANDIDATES.items() if v["bundle"].startswith("semantic")]
    selection["semantic_control"] = choose(semantic_names) if semantic_names else None
    alternatives = {
        "history": selection["baseline"],
        "context": selection["context"],
        "content": selection["content"],
    }
    best_score = min(scores[name] for name in alternatives.values())
    selection["recommended_forecaster"] = next(
        label for label, name in alternatives.items() if scores[name] <= best_score * 1.01
    )
    predictions.to_csv(run / "development_predictions.csv", index=False)
    write_json(run / "development.json", {"folds": outcomes, "scores": scores})
    selection["development_sha256"] = sha256_file(run / "development.json")
    selection["predictions_sha256"] = sha256_file(run / "development_predictions.csv")
    write_json(run / "selection.json", selection)
    return selection


def paired_gain(frame, candidate, reference, seed=20261017):
    """Bootstrap independent channel clusters; positive means less absolute log error."""
    delta = np.abs(frame.target_log.to_numpy() - reference) - np.abs(
        frame.target_log.to_numpy() - candidate
    )
    groups = pd.Series(delta).groupby(frame.channel_id.to_numpy()).mean().to_numpy()
    rng = np.random.default_rng(seed)
    samples = np.concatenate(
        [
            rng.choice(groups, (min(100, 5000 - i), len(groups)), replace=True).mean(axis=1)
            for i in range(0, 5000, 100)
        ]
    )
    reference_error = macro_error(frame, reference)
    return {
        "macro_log_error_reduction": float(groups.mean()),
        "relative_reduction_pct": float(100 * groups.mean() / reference_error)
        if reference_error
        else None,
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "clusters": len(groups),
        "bootstrap_repetitions": 5000,
    }


def ranking(frame, predictions):
    """All same-channel pairs in fixed seven-day publication bins, capped at 500 per channel.

    Cap selection uses a stable ID hash, never targets or forecast errors. Exact outcome ties
    are excluded only after pair selection. Cluster means prevent prolific channels dominating.
    """
    data = frame.copy().reset_index(drop=True)
    data["week"] = ((data.published_at - epoch("2016-07-01")) // (7 * DAY)).astype(int)
    candidates = defaultdict(list)
    for (channel, _), group in data.groupby(["channel_id", "week"]):
        ids = group.index.to_numpy()
        if len(ids) >= 2:
            candidates[channel].append(ids)
    channels, totals = [], Counter()
    for channel, weeks in candidates.items():
        scores = defaultdict(list)
        sizes = np.array([len(ids) * (len(ids) - 1) // 2 for ids in weeks])
        ends = np.cumsum(sizes)
        # Sample integer pair slots without materializing O(n²) pairs for prolific channels.
        seed = int(hashlib.sha256(channel.encode()).hexdigest()[:16], 16)
        sampled_slots = np.random.default_rng(seed).choice(
            int(ends[-1]), min(500, int(ends[-1])), replace=False
        )
        for slot in sorted(sampled_slots):
            week = int(np.searchsorted(ends, slot, side="right"))
            local = slot - (ends[week - 1] if week else 0)
            ids = weeks[week]
            offsets = np.arange(len(ids))
            starts = offsets * (2 * len(ids) - offsets - 1) // 2
            offset = int(np.searchsorted(starts, local, side="right") - 1)
            left, right = int(ids[offset]), int(ids[offset + 1 + local - starts[offset]])
            actual = np.sign(data.target_log.iloc[left] - data.target_log.iloc[right])
            if not actual:
                totals["true_ties"] += 1
                continue
            totals["pairs"] += 1
            for name, values in predictions.items():
                direction = np.sign(values[left] - values[right])
                scores[name].append(0.5 if direction == 0 else float(direction == actual))
        if scores:
            channels.append(
                {"channel_id": channel, **{n: float(np.mean(v)) for n, v in scores.items()}}
            )
    if not channels:
        return {
            "pairs": 0,
            "channels": 0,
            "macro_accuracy": {},
            "content_vs_context_ci95": None,
            "content_vs_history_ci95": None,
        }
    groups = pd.DataFrame(channels)
    rng = np.random.default_rng(20261018)
    deltas = {name: (groups.content - groups[name]).to_numpy() for name in ("context", "history")}
    sampled = {name: [] for name in deltas}
    for _ in range(50):
        selected = rng.integers(0, len(groups), (100, len(groups)))
        for name, delta in deltas.items():
            sampled[name].extend(delta[selected].mean(axis=1))
    return {
        **dict(totals),
        "channels": len(groups),
        "macro_accuracy": {name: float(groups[name].mean()) for name in predictions},
        "content_vs_context_gain": float(deltas["context"].mean()),
        "content_vs_context_ci95": np.quantile(sampled["context"], [0.025, 0.975]).tolist(),
        "content_vs_history_gain": float(deltas["history"].mean()),
        "content_vs_history_ci95": np.quantile(sampled["history"], [0.025, 0.975]).tolist(),
    }


def prediction_metrics(frame, predicted, radius=None):
    y = frame.target_log.to_numpy()
    counts = frame.target.to_numpy()
    estimate = np.expm1(predicted)
    result = {
        "videos": len(frame),
        "channels": int(frame.channel_id.nunique()),
        "macro_mae_log": macro_error(frame, predicted),
        "mae_log": float(np.mean(np.abs(y - predicted))),
        "mae_views": float(np.mean(np.abs(counts - estimate))),
        "within_factor_two": float(np.mean(np.abs(y - predicted) <= np.log(2))),
        "median_multiplicative_error": float(np.exp(np.median(np.abs(y - predicted)))),
    }
    if radius is not None:
        low, high = np.maximum(0, predicted - radius), predicted + radius
        result.update(
            {
                "coverage90": float(np.mean((y >= low) & (y <= high))),
                "mean_interval_width_views": float(np.mean(np.expm1(high) - np.expm1(low))),
                "median_interval_width_views": float(np.median(np.expm1(high) - np.expm1(low))),
                "interval_score_log": float(
                    np.mean(high - low + 20 * np.maximum(low - y, 0) + 20 * np.maximum(y - high, 0))
                ),
            }
        )
    return result


def conformal_radius(errors, coverage=0.9):
    order = int(np.ceil((len(errors) + 1) * coverage))
    if order > len(errors):
        raise ValueError("Insufficient calibration sample")
    return float(np.sort(errors)[order - 1])


def verify_frozen(root, run):
    plan, selection = read_json(run / "plan.json"), read_json(run / "selection.json")
    expected = {
        "plan_sha256": run / "plan.json",
        "development_sha256": run / "development.json",
        "predictions_sha256": run / "development_predictions.csv",
    }
    if any(sha256_file(path) != selection[key] for key, path in expected.items()):
        raise ValueError("Frozen development artifacts changed")
    if plan["code_sha256"] != fingerprint():
        raise ValueError("Study code or dependencies changed after selection")
    if plan["audit"]["cohort_sha256"] != sha256_file(root / "cohort.parquet"):
        raise ValueError("Frozen cohort changed")
    if plan.get("semantic_sha256") and plan["semantic_sha256"] != sha256_file(
        root / "semantic.json"
    ):
        raise ValueError("Frozen semantic source changed")
    return plan, selection


def final(root=ROOT, run=RUN):
    plan, selection = verify_frozen(root, run)
    ledger = run.parent / f"reach-final-opened-{plan['audit']['cohort_sha256']}.json"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    # Claim before any final fitting or scoring. Failures do not silently reopen the holdout.
    with ledger.open("x") as out:
        json.dump(
            {
                "opened_at": utcnow().isoformat(),
                "selection_sha256": sha256_file(run / "selection.json"),
            },
            out,
        )
    frame, neighbors = data_for_study(root)
    semantic = load_semantic(root, frame)
    train = indices(frame, plan["calibration_start"], train=True)
    cal = indices(frame, plan["calibration_start"], plan["calibration_stop"])
    test = np.flatnonzero(frame.published_at >= epoch(FINAL_START))
    if max(frame.available_at.iloc[cal]) >= epoch(FINAL_START):
        raise ValueError("Calibration outcomes are not mature before final test")
    predict = np.concatenate([cal, test])
    encoder = ReachEncoder()
    matrices = encoder.fit_transform(frame, train, neighbors, semantic)
    encoder.save(run / "models")
    predictions = {"history": frame.iloc[predict][selection["baseline"]].to_numpy()}
    names = {selection["context"], selection["content"], selection["lexical_control"]}
    if selection["semantic_control"]:
        names.add(selection["semantic_control"])
    needs_blend = "blend_text_retrieval" in names
    if needs_blend:
        names |= {"linear_text_100", "retrieval_mae6"}
        names.remove("blend_text_retrieval")
    fit_predictions = {}
    for name in sorted(names):
        print("Final fit:", name, flush=True)
        fit_predictions[name] = fit_candidate(name, matrices, frame, train, predict, run / "models")
        # Verify persistence on calibration inputs; never refit or select using test errors.
        replay = predict_saved_candidate(name, matrices, frame, cal, run / "models")
        if not np.allclose(replay, fit_predictions[name][: len(cal)], atol=1e-6, rtol=1e-6):
            raise ValueError("Saved model does not reproduce its calibration forecasts")
    if needs_blend:
        fit_predictions["blend_text_retrieval"] = (
            0.5 * fit_predictions["linear_text_100"] + 0.5 * fit_predictions["retrieval_mae6"]
        )
    for label in ("context", "content"):
        predictions[label] = fit_predictions[selection[label]]
    predictions["lexical_control"] = fit_predictions[selection["lexical_control"]]
    if selection["semantic_control"]:
        predictions["semantic_control"] = fit_predictions[selection["semantic_control"]]
    radii = {}
    for name, values in predictions.items():
        errors = np.abs(frame.iloc[cal].target_log.to_numpy() - values[: len(cal)])
        radii[name] = {"global": conformal_radius(errors)}
        for category in frame.iloc[cal].category.unique():
            mask = frame.iloc[cal].category.to_numpy() == category
            if mask.sum() >= 100:
                radii[name][category] = conformal_radius(errors[mask])
    result = (
        frame.iloc[test][
            [
                "creative_id",
                "channel_id",
                "published_at",
                "category",
                "target",
                "target_log",
                "reserved_channel",
                "history_count",
                "style_title_subset",
            ]
        ]
        .copy()
        .reset_index(drop=True)
    )
    for name, values in predictions.items():
        result[name] = values[len(cal) :]
        result[f"{name}_radius"] = result.category.map(
            lambda c, n=name: radii[n].get(c, radii[n]["global"])
        )
    for name in BASELINES:
        result[f"baseline_{name}"] = frame.iloc[test][name].to_numpy()
    result.to_csv(run / "test_predictions.csv", index=False)
    masks = {
        "primary": ~result.reserved_channel & (result.history_count >= 3),
        "little_history": ~result.reserved_channel & (result.history_count < 3),
        "unseen_channels": result.reserved_channel,
    }
    for category in ("2", "26"):
        masks[f"category_{category}"] = masks["primary"] & (result.category == category)
    masks["style_title_subset"] = masks["primary"] & result.style_title_subset
    sections = {}
    for label, mask in masks.items():
        part = result[mask].reset_index(drop=True)
        if not len(part):
            sections[label] = {"videos": 0}
            continue
        forecasts = {name: part[name].to_numpy() for name in predictions}
        metrics = {
            name: prediction_metrics(part, values, part[f"{name}_radius"].to_numpy())
            for name, values in forecasts.items()
        }
        sections[label] = {
            "metrics": metrics,
            "all_history_baselines": {
                name: prediction_metrics(part, part[f"baseline_{name}"].to_numpy())
                for name in BASELINES
            },
            "content_vs_history": paired_gain(part, forecasts["content"], forecasts["history"]),
            "content_vs_context": paired_gain(part, forecasts["content"], forecasts["context"]),
            "context_vs_history": paired_gain(part, forecasts["context"], forecasts["history"]),
            "recommended_vs_history": paired_gain(
                part, forecasts[selection["recommended_forecaster"]], forecasts["history"]
            ),
            "ranking": ranking(part, forecasts),
        }
    primary = sections["primary"]
    comparison = [primary["content_vs_history"], primary["content_vs_context"]]
    size = primary["metrics"]["content"]
    gain = (
        size["videos"] >= 500
        and size["channels"] >= 50
        and all(x["relative_reduction_pct"] >= 5 and x["ci95"][0] > 0 for x in comparison)
    )
    ranks = primary["ranking"]
    ranking_gain = (
        ranks["pairs"] >= 200
        and ranks["channels"] >= 30
        and ranks["content_vs_context_ci95"][0] > 0
        and ranks["content_vs_history_ci95"][0] > 0
    )
    report = {
        "completed_at": utcnow(),
        "selection": selection,
        "train_videos": len(train),
        "calibration_videos": len(cal),
        "all_test_videos": len(test),
        "prediction_intervals": {"nominal_coverage": 0.9, "absolute_log_error_radius": radii},
        "sections": sections,
        "confirmed_error_gain": bool(gain),
        "confirmed_ranking_gain": bool(ranking_gain),
        "confirmed_recommended_vs_history": bool(
            size["videos"] >= 500
            and size["channels"] >= 50
            and primary["recommended_vs_history"]["relative_reduction_pct"] >= 5
            and primary["recommended_vs_history"]["ci95"][0] > 0
        ),
        "case_solved": False,
        "production_eligible": False,
        "limits": [
            "Retrospective 2016 sample; Twitter/availability/100-view selection; not current Shorts or all channel uploads.",
            "Seven calendar dates, not 168 hours; outcome availability is a conservative convention, not observed collection time.",
            "No original prepublication media/title versions, paid-distribution information, or controlled creative alternatives.",
            "Howto & Style includes non-fashion/non-beauty tutorials. No causal or current-client claim.",
        ],
        "test_predictions_sha256": sha256_file(run / "test_predictions.csv"),
        "selection_sha256": sha256_file(run / "selection.json"),
    }
    write_json(run / "final.json", report)
    return report
