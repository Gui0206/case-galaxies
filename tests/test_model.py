import shutil
from datetime import timedelta

import numpy as np
import pytest
from catboost import Pool

from galaxies.demo import demo_comparison
from galaxies.io import read_json
from galaxies.metrics import calibrate_pairs, comparable_pairs
from galaxies.model import ModelBundle, conformal_adjustment
from galaxies.registry import Registry
from galaxies.schemas import PredictRequest
from galaxies.service import PredictionService


def test_finite_sample_conformal_quantile():
    assert conformal_adjustment(np.arange(1, 20), 0.9) == 18
    with pytest.raises(ValueError, match="insuficiente"):
        conformal_adjustment(np.array([1, 2]), 0.99)


def test_artifacts_and_shap_are_consistent(trained):
    settings, report = trained
    bundle = ModelBundle(settings.workspace / "models" / report["model_id"])
    request = demo_comparison()
    vector = bundle.encoder.transform(request.creatives[0], request.context, bundle.history)
    center, lo, hi = bundle.predict(vector.reshape(1, -1))
    shap = bundle.point.get_feature_importance(Pool(vector.reshape(1, -1)), type="ShapValues")[0]
    assert np.sum(shap) == pytest.approx(center[0], abs=1e-7)
    assert 0 <= lo[0] <= np.expm1(center[0]) <= hi[0]
    assert report["metrics"]["mae_log"] < report["baseline"]["mae_log"]
    assert report["eligible_for_decision"] is False


def test_synthetic_model_never_recommends_a_real_campaign(trained):
    settings, _ = trained
    service = PredictionService(settings)
    result = service.compare(demo_comparison())
    assert result["decision"] == "demonstration_only"
    assert result["recommended_id"] is None
    assert all(not p["eligible_for_decision"] for p in result["ranking"])
    values = [p["predicted_metric"] for p in result["ranking"]]
    assert values == sorted(values, reverse=True)
    trace = read_json(settings.workspace / "runs" / f"{result['ranking'][0]['trace_id']}.json")
    assert trace["stages"] == ["validate", "features", "predict_and_explain"]


def test_model_cannot_predict_its_own_past(trained):
    settings, _ = trained
    request = demo_comparison()
    past = request.context.as_of - timedelta(days=1000)
    ctx = request.context.model_copy(update={"as_of": past, "channel_stats_observed_at": past})
    with pytest.raises(ValueError, match="disponível em as_of"):
        PredictionService(settings).predict(
            PredictRequest(creative=request.creatives[0], context=ctx)
        )


def test_explicit_demo_promotion_required(trained):
    settings, report = trained
    with pytest.raises(ValueError, match="sintético"):
        Registry(settings.workspace / "models").promote(report["model_id"])


def test_tampering_is_detected_before_load(tmp_path, trained):
    settings, report = trained
    target = tmp_path / report["model_id"]
    shutil.copytree(settings.workspace / "models" / report["model_id"], target)
    (target / "report.json").write_text("{}")
    with pytest.raises(ValueError, match="alterado"):
        ModelBundle(target)


def test_calibration_uses_only_comparable_pairs(records):
    pairs = comparable_pairs(records)
    assert all(records[a].channel_id == records[b].channel_id for a, b in pairs)
    pred = np.log1p([r.target for r in records])
    result = calibrate_pairs(records, pred)
    assert result["available"] and result["slope"] > 0


def test_empty_content_is_flagged(trained):
    settings, _ = trained
    request = demo_comparison()
    request.creatives[0].features = {}
    result = PredictionService(settings).compare(request)
    prediction = next(
        p for p in result["ranking"] if p["creative_id"] == request.creatives[0].creative_id
    )
    assert any("ausente" in reason for reason in prediction["confidence_reasons"])
