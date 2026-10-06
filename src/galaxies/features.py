"""Train-only text/visual projections and point-in-time feature construction."""

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer

from galaxies.data import ChannelHistory
from galaxies.io import read_json, write_json
from galaxies.schemas import FEATURES, RUBRIC_FEATURES, Creative, PostRecord, PredictionContext

LABELS = {
    "log_channel_median": "Histórico recente do canal",
    "log_channel_mean": "Média histórica do canal",
    "log_subscribers": "Porte do canal",
    "history_count": "Histórico disponível",
    "duration_s": "Duração",
    "hook_clarity": "Clareza da abertura",
    "product_visibility": "Presença do produto",
    "demonstration": "Demonstração de uso",
    "benefit_clarity": "Clareza do benefício",
    "pacing": "Ritmo da edição",
    "speech_energy": "Energia da narração",
    "cta": "Chamada para ação",
    "onscreen_text_density": "Densidade de texto na tela",
    "brightness": "Luminosidade",
    "saturation": "Saturação",
    "frame_change": "Variação entre frames",
    "cuts_per_second": "Frequência aproximada de cortes",
    "audio_rms": "Energia do áudio",
    "silence_ratio": "Proporção de silêncio",
    "speech_words_per_second": "Densidade de fala",
    "ocr_word_count": "Texto identificado nos frames",
}


def label(name: str) -> str:
    if name.startswith("text_component"):
        return "Representação semântica do texto"
    if name.startswith("visual_component"):
        return "Representação visual do criativo"
    return LABELS.get(name, name.replace("_", " "))


def record_context(record: PostRecord) -> PredictionContext:
    return PredictionContext(
        channel_id=record.channel_id,
        sector=record.sector,
        as_of=record.published_at,
        planned_datetime=record.published_at,
        metric=record.metric,
        horizon_days=record.horizon_days,
        subscribers=record.subscribers,
        channel_stats_observed_at=record.channel_stats_observed_at,
        language=record.language,
        distribution=record.distribution,
    )


class FeatureEncoder:
    def __init__(self, blocks: list[str], seed: int = 42):
        self.blocks = blocks
        self.seed = seed
        self.vectorizer = HashingVectorizer(
            n_features=1024,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            strip_accents="unicode",
        )
        self.text_components = np.empty((0, 1024))
        self.visual_components = np.empty((0, 0))
        self.visual_mean = np.empty(0)
        self.visual_dim = 0
        self.names: list[str] = []
        self.medians = np.empty(0)
        self.scales = np.empty(0)
        self.low = np.empty(0)
        self.high = np.empty(0)

    def _basic(self, creative: Creative, context: PredictionContext, history: ChannelHistory):
        values = {}
        if "context" in self.blocks:
            hist = history.get(context.channel_id, context.sector, context.as_of)
            values.update(
                {
                    "log_channel_median": np.log1p(hist["pooled_median"]),
                    "log_channel_mean": np.log1p(hist["channel_mean"]),
                    "history_count": float(hist["history_count"]),
                    "history_log_std": hist["history_log_std"],
                    "log_subscribers": np.log1p(context.subscribers)
                    if context.subscribers is not None
                    else np.nan,
                    "hour_sin": np.sin(context.planned_datetime.hour * np.pi / 12),
                    "hour_cos": np.cos(context.planned_datetime.hour * np.pi / 12),
                    "weekday_sin": np.sin(context.planned_datetime.weekday() * 2 * np.pi / 7),
                    "weekday_cos": np.cos(context.planned_datetime.weekday() * 2 * np.pi / 7),
                    **{
                        f"sector_{s}": float(context.sector == s)
                        for s in ("beleza", "automotivo", "moda")
                    },
                }
            )
        if "text" in self.blocks:
            values["caption_words"] = len(creative.caption.split())
            values["transcript_words"] = len(creative.transcript.split())
        names = set()
        if "visual" in self.blocks:
            names |= FEATURES - RUBRIC_FEATURES
        if "rubric" in self.blocks:
            names |= RUBRIC_FEATURES
        for name in sorted(names):
            value = creative.features.get(name)
            values[name] = np.nan if value is None else value
        return values

    def fit(self, records: list[PostRecord], history: ChannelHistory) -> np.ndarray:
        if not self.blocks:
            raise ValueError("Selecione pelo menos um bloco de atributos.")
        if "text" in self.blocks:
            text = [f"{r.caption} {r.transcript}" for r in records]
            hashed = self.vectorizer.transform(text)
            if hashed.nnz:
                svd = TruncatedSVD(n_components=min(12, len(records) - 1), random_state=self.seed)
                svd.fit(hashed)
                self.text_components = svd.components_
        embeddings = [r.visual_embedding for r in records if r.visual_embedding]
        if "visual" in self.blocks and len(embeddings) >= 10:
            dimensions = {len(e) for e in embeddings}
            if len(dimensions) != 1:
                raise ValueError("Embeddings devem usar o mesmo encoder e dimensão.")
            self.visual_dim = dimensions.pop()
            pca = PCA(
                n_components=min(16, len(embeddings), self.visual_dim), random_state=self.seed
            )
            pca.fit(np.asarray(embeddings))
            self.visual_components, self.visual_mean = pca.components_, pca.mean_
        self.names = list(self._basic(records[0], record_context(records[0]), history))
        self.names += [f"text_component_{i}" for i in range(len(self.text_components))]
        self.names += [f"visual_component_{i}" for i in range(len(self.visual_components))]
        if self.visual_dim:
            self.names += ["visual_embedding_missing"]
        matrix = self.transform_records(records, history)
        # All-missing attributes remain explicit; they are not presented as observed zeroes.
        self.medians = np.array([np.nanmedian(c) if np.isfinite(c).any() else 0 for c in matrix.T])
        filled = np.where(np.isfinite(matrix), matrix, self.medians)
        self.scales = np.maximum(np.std(filled, axis=0), 0.1)
        self.low, self.high = np.quantile(filled, [0.01, 0.99], axis=0)
        return matrix

    def transform(
        self, creative: Creative, context: PredictionContext, history: ChannelHistory
    ) -> np.ndarray:
        values = list(self._basic(creative, context, history).values())
        if len(self.text_components):
            hashed = self.vectorizer.transform([f"{creative.caption} {creative.transcript}"])
            values.extend(np.asarray(hashed @ self.text_components.T).ravel())
        if self.visual_dim:
            if creative.visual_embedding and len(creative.visual_embedding) != self.visual_dim:
                raise ValueError("Dimensão do embedding incompatível com o modelo.")
            vector = (
                np.asarray(creative.visual_embedding)
                if creative.visual_embedding
                else self.visual_mean
            )
            values.extend((vector - self.visual_mean) @ self.visual_components.T)
            values.append(float(not creative.visual_embedding))
        return np.asarray(values, dtype=float)

    def transform_records(self, records: list[PostRecord], history: ChannelHistory) -> np.ndarray:
        return np.array([self.transform(r, record_context(r), history) for r in records])

    def standardized(self, matrix: np.ndarray) -> np.ndarray:
        return (np.where(np.isfinite(matrix), matrix, self.medians) - self.medians) / self.scales

    def out_of_domain(self, vector: np.ndarray) -> list[str]:
        reasons = []
        for i, name in enumerate(self.names):
            if name in FEATURES and np.isfinite(vector[i]):
                margin = max((self.high[i] - self.low[i]) * 0.5, 0.1)
                if vector[i] < self.low[i] - margin or vector[i] > self.high[i] + margin:
                    reasons.append(f"Atributo fora da faixa de treino: {label(name)}.")
        return reasons

    def save(self, folder: Path):
        write_json(
            folder / "encoder.json",
            {
                "blocks": self.blocks,
                "seed": self.seed,
                "names": self.names,
                "visual_dim": self.visual_dim,
            },
        )
        np.savez_compressed(
            folder / "encoder.npz",
            text=self.text_components,
            visual=self.visual_components,
            visual_mean=self.visual_mean,
            medians=self.medians,
            scales=self.scales,
            low=self.low,
            high=self.high,
        )

    @classmethod
    def load(cls, folder: Path) -> "FeatureEncoder":
        metadata = read_json(folder / "encoder.json")
        instance = cls(metadata["blocks"], metadata["seed"])
        instance.names, instance.visual_dim = metadata["names"], metadata["visual_dim"]
        with np.load(folder / "encoder.npz", allow_pickle=False) as arrays:
            for attribute, key in (
                ("text_components", "text"),
                ("visual_components", "visual"),
                ("visual_mean", "visual_mean"),
                ("medians", "medians"),
                ("scales", "scales"),
                ("low", "low"),
                ("high", "high"),
            ):
                setattr(instance, attribute, arrays[key])
        return instance
