"""Strict public contracts shared by CLI, Python, HTTP and MCP."""

import math
from datetime import datetime, timedelta
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

Sector = Literal["beleza", "automotivo", "moda"]
Metric = Literal["likes", "views"]

# Only pre-publication content attributes can enter the predictor.
FEATURES = {
    "duration_s",
    "brightness",
    "saturation",
    "frame_change",
    "cuts_per_second",
    "audio_rms",
    "silence_ratio",
    "speech_words_per_second",
    "ocr_word_count",
    "hook_clarity",
    "product_visibility",
    "demonstration",
    "onscreen_text_density",
    "pacing",
    "speech_energy",
    "benefit_clarity",
    "cta",
}
RUBRIC_FEATURES = {
    "hook_clarity",
    "product_visibility",
    "demonstration",
    "onscreen_text_density",
    "pacing",
    "speech_energy",
    "benefit_clarity",
    "cta",
}


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Evidence(Contract):
    attribute: str
    observation: str = Field(max_length=1000)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def chronological(self):
        if self.end_seconds < self.start_seconds:
            raise ValueError("O fim da evidência deve suceder seu início.")
        return self


class Creative(Contract):
    creative_id: str = Field(min_length=1, max_length=150)
    attributes_source: Literal["media_extraction", "human_annotation", "simulation"] = "human_annotation"
    caption: str = Field(default="", max_length=10000)
    transcript: str = Field(default="", max_length=30000)
    features: dict[str, float | None] = Field(default_factory=dict)
    visual_embedding: list[float] = Field(default_factory=list, max_length=2048)
    evidence: list[Evidence] = Field(default_factory=list, max_length=100)
    media_id: str | None = None

    @model_validator(mode="after")
    def allowed_features(self):
        unknown = set(self.features) - FEATURES
        if unknown:
            raise ValueError(f"Atributos não permitidos: {sorted(unknown)}")
        for name, value in self.features.items():
            if value is not None and not math.isfinite(value):
                raise ValueError(f"Atributo não finito: {name}")
            if name in RUBRIC_FEATURES and value is not None and not 0 <= value <= 10:
                raise ValueError(f"Rubrica fora de 0–10: {name}")
        return self


class PostRecord(Creative):
    channel_id: str = Field(min_length=1, max_length=150)
    sector: Sector
    published_at: AwareDatetime
    features_available_at: AwareDatetime
    observed_at: AwareDatetime
    metric: Metric = "likes"
    horizon_days: int = Field(default=7, ge=1, le=90)
    target: int = Field(ge=0)
    subscribers: int | None = Field(default=None, ge=0)
    channel_stats_observed_at: AwareDatetime | None = None
    content_hash: str = Field(min_length=1, max_length=128)
    content_group: str | None = None
    source: str = Field(min_length=1)
    usage_basis: str = Field(min_length=1)
    synthetic: bool = False
    language: str = "pt"
    distribution: Literal["organic", "paid", "unknown"] = "unknown"

    @model_validator(mode="after")
    def information_cutoff(self):
        if self.attributes_source == "simulation" and not self.synthetic:
            raise ValueError("Atributos simulados exigem synthetic=true.")
        if self.features_available_at > self.published_at:
            raise ValueError("Conteúdo/contexto não estava disponível antes da publicação.")
        if self.subscribers is not None and self.channel_stats_observed_at is None:
            raise ValueError("Inscritos exigem channel_stats_observed_at.")
        if self.channel_stats_observed_at and self.channel_stats_observed_at > self.published_at:
            raise ValueError("Estatísticas futuras do canal causam vazamento.")
        maturity = self.published_at + timedelta(days=self.horizon_days)
        if not maturity <= self.observed_at <= maturity + timedelta(hours=24):
            raise ValueError("Alvo deve ser observado no horizonte, com tolerância máxima de 24h.")
        return self


class PredictionContext(Contract):
    channel_id: str = Field(min_length=1, max_length=150)
    sector: Sector
    as_of: AwareDatetime
    planned_datetime: AwareDatetime
    metric: Metric = "likes"
    horizon_days: int = Field(default=7, ge=1, le=90)
    subscribers: int | None = Field(default=None, ge=0)
    channel_stats_observed_at: AwareDatetime | None = None
    language: str = "pt"
    distribution: Literal["organic", "paid", "unknown"] = "unknown"

    @model_validator(mode="after")
    def cutoff(self):
        if self.planned_datetime < self.as_of:
            raise ValueError("Publicação planejada deve ser igual ou posterior a as_of.")
        if self.subscribers is not None and self.channel_stats_observed_at is None:
            raise ValueError("Informe quando os inscritos foram observados.")
        if self.channel_stats_observed_at and self.channel_stats_observed_at > self.as_of:
            raise ValueError("Estatísticas do canal posteriores à decisão.")
        return self


class PredictRequest(Contract):
    creative: Creative
    context: PredictionContext
    model_id: str | None = None


class CompareRequest(Contract):
    creatives: list[Creative] = Field(min_length=2, max_length=10)
    context: PredictionContext
    model_id: str | None = None

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [c.creative_id for c in self.creatives]
        if len(set(ids)) != len(ids):
            raise ValueError("creative_id deve ser único na comparação.")
        return self


class TrainConfig(Contract):
    iterations: int = Field(default=240, ge=20, le=2000)
    depth: int = Field(default=5, ge=2, le=8)
    learning_rate: float = Field(default=0.06, gt=0, le=1)
    seed: int = 42
    coverage: float = Field(default=0.9, ge=0.5, le=0.99)
    feature_blocks: list[Literal["context", "text", "visual", "rubric"]] = Field(
        default_factory=lambda: ["context", "text", "visual", "rubric"]
    )
    held_out_channels: list[str] = Field(default_factory=list)
    min_train_rows: int = Field(default=60, ge=20)


class TrainRequest(Contract):
    dataset: str
    config: TrainConfig = Field(default_factory=TrainConfig)


class Driver(Contract):
    feature: str
    label: str
    contribution_log: float
    direction: Literal["positive", "negative"]


class Prediction(Contract):
    creative_id: str
    model_id: str
    metric: Metric
    horizon_days: int
    predicted_metric: float
    interval_low: float
    interval_high: float
    nominal_coverage: float
    relative_to_channel: float | None
    confidence_flag: Literal["supported", "limited", "insufficient"]
    confidence_reasons: list[str]
    evidence_level: Literal["synthetic_demo", "retrospective"]
    eligible_for_decision: bool
    top_drivers: list[Driver]
    explanation: str
    comparables: list[dict]
    extraction_evidence: list[Evidence]
    trace_id: str
    predicted_at: datetime
