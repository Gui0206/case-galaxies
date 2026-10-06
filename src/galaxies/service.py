"""Single inference implementation for Python, API and MCP, orchestrated with LangGraph."""

import uuid
from datetime import datetime
from typing import TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph
from scipy.special import expit

from galaxies.config import Settings
from galaxies.io import utcnow, write_json
from galaxies.model import ModelBundle
from galaxies.registry import Registry
from galaxies.schemas import (
    FEATURES,
    CompareRequest,
    Creative,
    Prediction,
    PredictionContext,
    PredictRequest,
)


class State(TypedDict, total=False):
    request: PredictRequest
    bundle: ModelBundle
    vector: np.ndarray
    reasons: list[str]
    flags: list[str]
    prediction: Prediction
    trace_id: str


def _validate(state: State) -> dict:
    bundle, request = state["bundle"], state["request"]
    report, context = bundle.report, request.context
    if (context.metric, context.horizon_days) != (report["metric"], report["horizon_days"]):
        raise ValueError("Métrica/horizonte incompatíveis com o modelo selecionado.")
    if context.as_of < datetime.fromisoformat(report["inference_not_before"]):
        raise ValueError("Modelo ainda não estaria disponível em as_of; use o backtest para o passado.")
    reasons, flags = [], []
    if report["evidence_level"] == "synthetic_demo":
        reasons.append("Modelo treinado em dados sintéticos; saída apenas para demonstração.")
        flags.append("synthetic")
    elif not report["eligible_for_decision"]:
        reasons.append("Modelo de pesquisa ainda não passou pelos critérios de decisão.")
        flags.append("unapproved")
    if context.sector not in report["sectors"]:
        reasons.append("Setor ausente no treinamento.")
        flags.append("out_of_domain")
    if context.language not in report["languages"]:
        reasons.append("Idioma ausente no treinamento.")
        flags.append("out_of_domain")
    if context.distribution not in report["distribution"]:
        reasons.append("Regime de distribuição diferente do treinamento.")
        flags.append("out_of_domain")
    return {"reasons": reasons, "flags": flags}


def _features(state: State) -> dict:
    request, bundle = state["request"], state["bundle"]
    vector = bundle.encoder.transform(request.creative, request.context, bundle.history)
    reasons, flags = state["reasons"].copy(), state["flags"].copy()
    history = bundle.history.get(request.context.channel_id, request.context.sector, request.context.as_of)
    if history["history_count"] < 5:
        reasons.append("Canal com menos de cinco resultados históricos disponíveis.")
        flags.append("sparse_history")
    content_names = [n for n in bundle.encoder.names if n in FEATURES]
    available = sum(request.creative.features.get(n) is not None for n in content_names)
    if content_names and available / len(content_names) < 0.5:
        reasons.append("Mais da metade dos atributos multimodais está ausente.")
        flags.append("missing_features")
    outliers = bundle.encoder.out_of_domain(vector)
    reasons.extend(outliers)
    if outliers:
        flags.append("out_of_domain")
    return {"vector": vector, "reasons": reasons, "flags": flags}


def _predict(state: State) -> dict:
    request, bundle, vector = state["request"], state["bundle"], state["vector"]
    center, low, high = bundle.predict(vector.reshape(1, -1))
    predicted = float(np.expm1(np.clip(center[0], 0, 50)))
    history = bundle.history.get(request.context.channel_id, request.context.sector, request.context.as_of)
    drivers = bundle.explain(vector)
    flags = state["flags"]
    severe = any(f in flags for f in ("synthetic", "unapproved", "out_of_domain", "missing_features"))
    confidence = "insufficient" if severe else "limited" if flags else "supported"
    eligible = bool(bundle.report["eligible_for_decision"] and not severe)
    positive = [d.label.lower() for d in drivers if d.direction == "positive"][:2]
    negative = [d.label.lower() for d in drivers if d.direction == "negative"][:2]
    explanation = (
        f"Estimativa de {predicted:,.0f} {request.context.metric} em {request.context.horizon_days} dias. "
        f"Intervalo nominal de {bundle.report['config']['coverage']:.0%}: {low[0]:,.0f} a {high[0]:,.0f}. "
    )
    if positive:
        explanation += f"Contribuições positivas no modelo: {', '.join(positive)}. "
    if negative:
        explanation += f"Contribuições negativas no modelo: {', '.join(negative)}. "
    explanation += "Contribuições são associações do modelo, não efeitos causais de alterações."
    prediction = Prediction(
        creative_id=request.creative.creative_id, model_id=bundle.report["model_id"],
        metric=request.context.metric, horizon_days=request.context.horizon_days,
        predicted_metric=predicted, interval_low=float(low[0]), interval_high=float(high[0]),
        nominal_coverage=bundle.report["config"]["coverage"],
        relative_to_channel=predicted / history["channel_median"] if history["history_count"] and history["channel_median"] > 0 else None,
        confidence_flag=confidence, confidence_reasons=state["reasons"],
        evidence_level=bundle.report["evidence_level"], eligible_for_decision=eligible,
        top_drivers=drivers, explanation=explanation,
        comparables=bundle.comparables(vector, request.context.sector, request.context.as_of, request.creative.creative_id),
        extraction_evidence=request.creative.evidence, trace_id=state["trace_id"], predicted_at=utcnow())
    return {"prediction": prediction}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("validate", _validate)
    graph.add_node("features", _features)
    graph.add_node("predict_and_explain", _predict)
    graph.add_edge(START, "validate")
    graph.add_edge("validate", "features")
    graph.add_edge("features", "predict_and_explain")
    graph.add_edge("predict_and_explain", END)
    return graph.compile()


class PredictionService:
    def __init__(self, settings: Settings):
        self.settings = settings.prepare()
        self.registry = Registry(settings.workspace / "models")
        self.graph = build_graph()
        self._cached_id = None
        self._bundle = None

    def bundle(self, model_id: str | None):
        selected = model_id or self.registry.active()
        if selected != self._cached_id or self._bundle is None:
            self._bundle = self.registry.load(selected)
            self._cached_id = selected
        return self._bundle

    def predict(self, request: PredictRequest) -> Prediction:
        trace_id = uuid.uuid4().hex
        bundle = self.bundle(request.model_id)
        result = self.graph.invoke({"request": request, "bundle": bundle, "trace_id": trace_id})
        prediction = result["prediction"]
        write_json(self.settings.workspace / "runs" / f"{trace_id}.json", {
            "request": request.model_dump(mode="json"), "prediction": prediction.model_dump(mode="json"),
            "features": {name: float(value) if np.isfinite(value) else None
                         for name, value in zip(bundle.encoder.names, result["vector"], strict=True)},
            "stages": ["validate", "features", "predict_and_explain"],
        })
        return prediction

    def compare(self, request: CompareRequest) -> dict:
        # Resolve the active model once so a promotion cannot split a comparison across models.
        bundle = self.bundle(request.model_id)
        model_id = bundle.report["model_id"]
        predictions = [self.predict(PredictRequest(creative=c, context=request.context, model_id=model_id))
                       for c in request.creatives]
        predictions.sort(key=lambda p: p.predicted_metric, reverse=True)
        calibration = bundle.report["pair_calibration"]
        pairs = []
        for i, a in enumerate(predictions):
            for b in predictions[i + 1:]:
                probability = float(expit(calibration["slope"] * (
                    np.log1p(a.predicted_metric) - np.log1p(b.predicted_metric)))) if calibration["available"] else None
                pairs.append({"a": a.creative_id, "b": b.creative_id,
                              "probability_a_better": probability,
                              "scope": "historical within-channel pairs; not randomized campaigns"})
        best_probability = pairs[0]["probability_a_better"] if pairs else None
        if bundle.report["evidence_level"] == "synthetic_demo":
            decision, recommended = "demonstration_only", None
            reason = "Ranking ilustrativo de um modelo sintético. Não use para decidir uma campanha real."
        elif not all(p.eligible_for_decision for p in predictions):
            decision, recommended = "insufficient_evidence", None
            reason = "Há limitações no modelo ou nas entradas; não há evidência suficiente para recomendar."
        elif best_probability is None or best_probability < 0.65:
            decision, recommended = "technical_tie", None
            reason = "A separação entre os primeiros criativos não atingiu o limiar operacional de 65%."
        else:
            decision, recommended = "recommendation", predictions[0].creative_id
            reason = "O primeiro criativo superou o limiar operacional de comparação. Valide a escolha no contexto da campanha."
        return {"model_id": model_id, "decision": decision, "recommended_id": recommended,
                "reason": reason, "ranking": [p.model_dump(mode="json") for p in predictions],
                "pairwise": pairs, "probability_threshold": 0.65,
                "evidence_level": bundle.report["evidence_level"]}


def predict(creative: Creative, context: PredictionContext, *, settings: Settings | None = None,
            model_id: str | None = None) -> Prediction:
    """Documented Python integration point. Identical implementation to API and MCP."""
    return PredictionService(settings or Settings()).predict(
        PredictRequest(creative=creative, context=context, model_id=model_id))
