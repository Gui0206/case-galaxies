"""Inference-only adapter for the frozen SMTPD experiment. Never fits or downloads.

The experimental models remain distinct from the synthetic demo registry. A real
prediction is not a production endorsement: the reserved test found no gain.
"""

from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from galaxies.io import read_json, sha256_file
from galaxies.reach_study import TEXT_STATS, ReachEncoder, history_features
from galaxies.smtpd_auxiliary import pure_features
from galaxies.smtpd_media_features import derived
from galaxies.smtpd_study import CONTEXT_RULES, HISTORY, enrich


class RealCreativeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    creative_id: str = Field(min_length=1)
    channel_id: str = Field(min_length=1)
    caption: str = Field(min_length=1)
    planned_datetime: AwareDatetime
    duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    category: Literal["2", "26"]
    media_path: str | None = None
    cached_video_id: str | None = None


class RealCompareRequest(BaseModel):
    creatives: list[RealCreativeRequest] = Field(min_length=2, max_length=10)


class RealPredictionService:
    def __init__(self, workspace=Path(".runtime")):
        self.workspace = Path(workspace).resolve()
        self.root = self.workspace / "public/smtpd"
        self.run = self.workspace / "reports/smtpd-final"
        self.models = self.run / "models"
        manifest_path = self.workspace / "real-service-manifest.json"
        if not manifest_path.is_file():
            raise ValueError(
                "Artefatos reais não preparados. Execute scripts/prepare_real_service.py com os arquivos locais existentes."
            )
        manifest = read_json(manifest_path)
        for name, digest in manifest["files"].items():
            path = (self.workspace / name).resolve()
            if not path.is_relative_to(self.workspace) or sha256_file(path) != digest:
                raise ValueError(f"Artefato real ausente ou alterado: {name}")
        self.manifest = manifest
        self.selection = read_json(self.run / "selection.json")
        if self.selection["content"] != "media/editing:tree3@0.25":
            raise ValueError("O adaptador exige a seleção SMTPD congelada, sem substituir modelos")
        self.report = read_json(self.run / "final.json")
        self.calibration = read_json(self.run / "calibration.json")
        self.cohort = pd.read_parquet(self.root / "cohort.parquet")
        self.semantic = np.load(self.root / "semantic.npy", allow_pickle=False)
        self.encoder = ReachEncoder.load(self.models / "base_encoder")
        self.within = dict(np.load(self.models / "base/within_semantic_100.npz"))
        self.aux = dict(np.load(self.models / "auxiliary/auxiliary_models.npz"))
        self.media_info = read_json(self.models / "media/editing-tree3/media_encoder.json")
        self.media_medians = np.load(self.models / "media/editing-tree3/media_encoder.npz")[
            "numeric_medians"
        ]
        self.index = read_json(self.root / "features/index.json")
        self.trees = {}
        for name, file in {
            "context": "base/context_d5_logmean_20.cbm",
            "ranking": "base/retrieval_d3_median_20.cbm",
            "editing": "media/editing-tree3/model.cbm",
        }.items():
            self.trees[name] = CatBoostRegressor().load_model(str(self.models / file))
        self._language = None
        self._analyzer = None

    def _embedding(self, text):
        matches = np.flatnonzero(self.cohort.title.eq(text).to_numpy())
        if len(matches):
            return self.semantic[matches[0]]
        if self._language is None:
            from sentence_transformers import SentenceTransformer

            identity = read_json(self.root / "semantic.json")["identity"]
            for name, digest in identity["weights"].items():
                local = self.workspace / "weights/language" / name
                if not local.is_file() or sha256_file(local) != digest:
                    raise ValueError(
                        "Pesos locais de linguagem ausentes ou alterados; nenhum download será iniciado"
                    )
            self._language = SentenceTransformer(
                str(self.workspace / "weights/language"), device="cpu", local_files_only=True
            )
        return self._language.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]

    def _inputs(self, request):
        date = request.planned_datetime.timestamp()
        if date < pd.Timestamp("2024-01-05", tz="UTC").timestamp():
            raise ValueError("Modelo final disponível apenas para decisões a partir de 05/01/2024")
        # Exclude the candidate's observed outcome even for retrospective demonstrations.
        keep = (self.cohort.published_at < date) & self.cohort.creative_id.ne(request.creative_id)
        past = self.cohort[keep].copy()
        candidate = pd.DataFrame(
            [
                {
                    "creative_id": request.creative_id,
                    "channel_id": request.channel_id,
                    "published_at": date,
                    "observed_at": date + 7 * 86400,
                    "available_at": date + 8 * 86400,
                    "title": request.caption.strip(),
                    "category": request.category,
                    "duration": request.duration_seconds,
                    "language": "unknown",
                    "definition": "unknown",
                    "target": 0,
                    "target_log": 0.0,
                    "reserved_channel": False,
                }
            ]
        )
        if not request.caption.strip():
            raise ValueError("Legenda vazia")
        frame, neighbors = history_features(pd.concat([past, candidate], ignore_index=True))
        frame = enrich(frame, neighbors)
        position = len(frame) - 1
        if frame.iloc[position].history_count < 3:
            raise ValueError(
                "Canal sem pelo menos três resultados maduros na base local; não estimar sem suporte"
            )
        vectors = np.vstack(
            [self.semantic[keep.to_numpy()], self._embedding(request.caption.strip())]
        )
        matrices = self.encoder.transform(frame, neighbors, vectors)
        extra = frame[CONTEXT_RULES + HISTORY[-2:]].to_numpy()
        matrices = {
            n: np.column_stack([matrices[n], extra])
            for n in ["context", "retrieval", "semantic_retrieval"]
        }
        pure = pure_features(frame, vectors)
        within_x = np.column_stack([pure[:, :15], self.encoder.semantic_pca.transform(vectors)])
        return frame, neighbors[position][-20:], matrices, pure, within_x

    def _media(self, request):
        if request.media_path and request.cached_video_id:
            raise ValueError("Escolha arquivo local ou vídeo em cache, não ambos")
        if request.cached_video_id:
            entry = self.index["rows"].get(request.cached_video_id)
            if not entry:
                return None, "Mídia ausente no cache; usada previsão textual congelada."
            path = (self.root / "features" / entry["file"]).resolve()
            if (
                path.parent != (self.root / "features").resolve()
                or sha256_file(path) != entry["sha256"]
            ):
                raise ValueError("Cache de mídia alterado")
            record = read_json(path)
            # Preserve the study's exact-duplicate fallback policy.
            duplicates = self.report["media_provenance"]["exact_duplicates_fallback_to_text"]
            if request.creative_id in duplicates:
                return None, "Duplicata exata: fallback textual preservado do estudo."
        elif request.media_path:
            path = (self.workspace / request.media_path).resolve()
            if not path.is_relative_to(self.workspace) or not path.is_file():
                raise ValueError(
                    "O vídeo deve ser um arquivo local dentro de .runtime; URLs não são baixadas"
                )
            # Uploading an existing video reuses its verified extraction, regardless of filename.
            digest = sha256_file(path)
            for video_id, entry in self.index["rows"].items():
                if entry.get("content_hash") == digest:
                    return self._media(
                        request.model_copy(update={"media_path": None, "cached_video_id": video_id})
                    )
            if request.duration_seconds > 180:
                return None, "Vídeo acima de 180 segundos: usada previsão textual congelada."
            if self._analyzer is None:
                from galaxies.deep_media import DeepMediaAnalyzer

                metadata = read_json(self.root / "features/extractor.json")
                analyzer = DeepMediaAnalyzer(
                    self.workspace / "weights",
                    self.workspace / "inference-media",
                    speech_backend=metadata["speech_backend"],
                )
                if analyzer.fingerprint != self.index["extractor_fingerprint"]:
                    raise ValueError("Extrator local difere do usado no estudo")
                self._analyzer = analyzer
            record = self._analyzer.analyze(path, request.caption)
        else:
            return None, "Sem mídia fornecida: usada previsão textual congelada."
        if abs(record["duration_s"] - request.duration_seconds) > max(
            1, request.duration_seconds * 0.02
        ):
            raise ValueError("Duração informada difere da mídia")
        return record, None

    def analyze_video(self, media_path: str, duration_seconds: float) -> dict:
        """Describe the supplied video only; no channel, forecast or performance score."""

        class MediaInput(BaseModel):
            media_path: str | None
            duration_seconds: float
            caption: str = ""
            creative_id: str = ""
            cached_video_id: str | None = None

        media = MediaInput(
            media_path=media_path,
            duration_seconds=duration_seconds,
            caption="",
            creative_id="",
            cached_video_id=None,
        )
        record, warning = self._media(media)
        if record is None:
            raise ValueError(warning or "Não foi possível analisar o vídeo")
        return {
            "mode": "descriptive_analysis",
            "performance_prediction_available": False,
            "duration_seconds": record["duration_s"],
            "sampled_frames": len(record["frame_times"]),
            "transcript": record["transcript"],
            "language": record["language_detected"],
            "editing": record["editing"]["features"],
            "speech": record["speech_features"],
            "limitations": [
                "Sem histórico do seu canal, esta análise não estima visualizações nem compara desempenho.",
                "Transcrição e detecção de cortes são automáticas e podem conter erros.",
                "Quadros amostrados geram representações visuais; não identificamos produtos, emoções ou qualidade criativa.",
                "Ritmo, luminosidade e fala descrevem o arquivo. Não são notas de qualidade nem provas de alcance.",
            ],
        }

    def predict(self, request: RealCreativeRequest) -> dict:
        frame, prior, x, pure, within_x = self._inputs(request)
        row = frame.iloc[-1]
        anchor = float(row.logmean_20)
        context = max(0.0, anchor + float(self.trees["context"].predict(x["context"][-1:])[0]))
        normalized = (within_x - self.within["mean"]) / self.within["scale"]
        delta = (normalized[-1] - normalized[prior].mean(axis=0)) * self.within["coef"]
        within = max(0.0, anchor + float(delta.sum()))
        text = 0.5 * context + 0.5 * within
        record, warning = self._media(request)
        editing_x = None
        predicted = text
        if record is not None:
            measurements = derived(record, "editing")
            values = np.array(
                [measurements.get(k, np.nan) for k in self.media_info["keys"]["editing"]]
            )
            values = np.where(np.isnan(values), self.media_medians, values)
            editing_x = np.r_[x["semantic_retrieval"][-1], values]
            editing = max(0.0, anchor + float(self.trees["editing"].predict(editing_x[None])[0]))
            predicted = 0.75 * text + 0.25 * editing
        aux_x = (pure - self.aux["mean"]) / self.aux["scale"]
        aux_delta = (aux_x[-1] - aux_x[prior].mean(axis=0)) @ self.aux["aux_context0_ridge100"]
        context_reference = 0.5 * context + 0.5 * max(0.0, anchor + aux_delta)
        ranking = max(
            0.0, float(row.median_20) + float(self.trees["ranking"].predict(x["retrieval"][-1:])[0])
        )
        radius = self.calibration["radii"]["content"]
        names = self.encoder.context_names + CONTEXT_RULES + HISTORY[-2:]
        drivers = []
        components = [("context", x["context"][-1], names, 0.375 if record else 0.5)]
        if editing_x is not None:
            media_names = (
                self.encoder.context_names
                + TEXT_STATS
                + [f"semantic_{i}" for i in range(32)]
                + [
                    "semantic_similarity_adjustment",
                    "semantic_similarity_max",
                    "semantic_similarity_weight",
                ]
                + CONTEXT_RULES
                + HISTORY[-2:]
                + self.media_info["keys"]["editing"]
            )
            components.append(("editing", editing_x, media_names, 0.25))
        constant = 0.0
        for name, vector, labels, weight in components:
            attribution = self.trees[name].get_feature_importance(
                Pool(vector[None]), type="ShapValues"
            )[0]
            constant += weight * float(attribution[-1])
            drivers += [
                {"component": name, "feature": feature, "contribution_log": float(weight * value)}
                for feature, value in zip(labels, attribution[:-1], strict=True)
            ]
        within_names = (
            [
                "log_duration",
                "duration_le_180",
                "duration_hinge_30",
                "duration_hinge_60",
                "duration_hinge_180",
                "duration_hinge_600",
                "hour_sin",
                "hour_cos",
                "weekday_sin",
                "weekday_cos",
            ]
            + TEXT_STATS
            + [f"semantic_{i}" for i in range(32)]
        )
        weight = 0.375 if record else 0.5
        drivers += [
            {
                "component": "within_channel_text",
                "feature": name,
                "contribution_log": float(weight * value),
            }
            for name, value in zip(within_names, delta, strict=True)
        ]
        adjustment = predicted - anchor - constant - sum(d["contribution_log"] for d in drivers)
        reasons = [
            "Ganho sobre histórico e contexto não demonstrado no teste reservado.",
            "Base de 2023–2024, selecionada, com histórico incompleto; não validada para campanhas atuais.",
            "Categoria Howto & Style é mais ampla que moda e beleza.",
        ]
        if warning:
            reasons.append(warning)
        comparables = frame.iloc[prior[-5:]]
        return {
            "creative_id": request.creative_id,
            "channel_id": request.channel_id,
            "model_id": "smtpd-frozen-2024",
            "evidence_level": "real_data_experimental",
            "metric": "views",
            "horizon_days": 7,
            "predicted_metric": float(np.expm1(predicted)),
            "interval_low": float(np.expm1(max(0, predicted - radius))),
            "interval_high": float(np.expm1(predicted + radius)),
            "interval_nominal_coverage": 0.9,
            "interval_test_coverage": self.report["primary_metrics"]["content"]["coverage90"],
            "historical_baseline": float(np.expm1(anchor)),
            "context_baseline": float(np.expm1(context_reference)),
            "relative_to_channel": float(np.expm1(predicted) / np.expm1(anchor))
            if anchor > 0
            else None,
            "ranking_score_log": ranking,
            "confidence_flag": "low",
            "eligible_for_decision": False,
            "confidence_reasons": reasons,
            "history_count": int(row.history_count),
            "history_available_before": request.planned_datetime.isoformat(),
            "latest_history_publication": pd.Timestamp(
                comparables.published_at.max(), unit="s", tz="UTC"
            ).isoformat(),
            "media_used_for_prediction": record is not None,
            "observed_content": None
            if record is None
            else {
                "duration_seconds": record["duration_s"],
                "semantic_frames": len(record["frame_times"]),
                "transcript": record["transcript"],
                "language": record["language_detected"],
                "editing": record["editing"]["features"],
                "note": "Imagem e fala foram extraídas; a correção multimodal escolhida usa apenas atributos de edição.",
            },
            "top_drivers": sorted(drivers, key=lambda d: abs(d["contribution_log"]), reverse=True)[
                :8
            ],
            "attribution": {
                "scale": "log(1+views)",
                "baseline_log": anchor,
                "model_constant_log": constant,
                "feature_sum_log": sum(d["contribution_log"] for d in drivers),
                "clipping_adjustment_log": adjustment,
                "prediction_log": predicted,
            },
            "explanation": f"O histórico fornece a referência de {np.expm1(anchor):.0f} views. O modelo ajusta essa referência para {np.expm1(predicted):.0f} views em sete dias, com intervalo de {np.expm1(max(0, predicted - radius)):.0f} a {np.expm1(predicted + radius):.0f}. As contribuições descrevem o cálculo, não efeitos causais. Não houve ganho comprovado sobre o histórico.",
            "recent_history": [
                {
                    "creative_id": p.creative_id,
                    "caption": p.title,
                    "views_7d": int(p.target),
                    "published_at": pd.Timestamp(p.published_at, unit="s", tz="UTC").isoformat(),
                    "available_at": pd.Timestamp(p.available_at, unit="s", tz="UTC").isoformat(),
                }
                for p in comparables.itertuples()
            ],
        }

    def replay(self, creative_id: str) -> dict:
        matches = self.cohort[self.cohort.creative_id.eq(creative_id)]
        if len(matches) != 1:
            raise ValueError("Vídeo não encontrado na base local")
        row = matches.iloc[0]
        result = self.predict(
            RealCreativeRequest(
                creative_id=creative_id,
                channel_id=row.channel_id,
                caption=row.title,
                planned_datetime=datetime.fromtimestamp(row.published_at, tz=UTC),
                duration_seconds=row.duration,
                category=row.category,
                cached_video_id=creative_id,
            )
        )
        result["demonstration"] = {
            "kind": "retrospective_replay",
            "observed_views_7d": int(row.target),
            "note": "Resultado observado anexado somente depois da inferência; não é entrada do modelo.",
        }
        return result

    def compare(self, request: RealCompareRequest) -> dict:
        if (
            len({c.channel_id for c in request.creatives}) != 1
            or len({c.planned_datetime for c in request.creatives}) != 1
            or len({c.category for c in request.creatives}) != 1
        ):
            raise ValueError("Comparar apenas o mesmo canal, categoria e horário planejado")
        if len({c.creative_id for c in request.creatives}) != len(request.creatives):
            raise ValueError("IDs dos criativos devem ser únicos")
        results = [self.predict(c) for c in request.creatives]
        return {
            "decision": "insufficient_evidence",
            "recommended_id": None,
            "reason": "Ordenação indicativa; superioridade preditiva não confirmada. Pontuação não é probabilidade de viralizar.",
            "ranking": sorted(results, key=lambda r: r["ranking_score_log"], reverse=True),
        }


@lru_cache(maxsize=2)
def real_service(workspace=Path(".runtime")):
    return RealPredictionService(workspace)


def predict_real(request: RealCreativeRequest, workspace=Path(".runtime")):
    return real_service(workspace).predict(request)
