"""Integration regression against published outputs; never fits or reevaluates models."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from galaxies.config import Settings
from galaxies.real_service import RealCompareRequest, RealCreativeRequest, RealPredictionService

WORKSPACE = Path(__file__).resolve().parents[1] / ".runtime"


@pytest.fixture(scope="module")
def saved_service():
    if not (WORKSPACE / "real-service-manifest.json").exists():
        pytest.skip("Existing local research artifacts are not distributed in Git")
    with (
        patch("catboost.CatBoostRegressor.fit", side_effect=AssertionError("Training forbidden")),
        patch(
            "galaxies.reach_study.ReachEncoder.fit_transform",
            side_effect=AssertionError("Training forbidden"),
        ),
        patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")),
    ):
        yield RealPredictionService(WORKSPACE)


def test_saved_prediction_parity(saved_service):
    published = pd.read_csv(WORKSPACE / "reports/smtpd-final/test_predictions.csv")
    for row in published.iloc[[0, 1, 2, 100, 250, 514]].itertuples():
        result = saved_service.replay(row.creative_id)
        for output, column in [
            ("predicted_metric", "content"),
            ("historical_baseline", "history"),
            ("context_baseline", "context"),
        ]:
            assert np.log1p(result[output]) == pytest.approx(getattr(row, column), abs=1e-10)
        assert result["ranking_score_log"] == pytest.approx(row.ranking_content, abs=1e-10)
        attr = result["attribution"]
        assert sum(
            attr[k]
            for k in [
                "baseline_log",
                "model_constant_log",
                "feature_sum_log",
                "clipping_adjustment_log",
            ]
        ) == pytest.approx(attr["prediction_log"], abs=1e-10)
        assert result["eligible_for_decision"] is False


def test_own_and_future_outcomes_are_not_inputs(saved_service):
    original = saved_service.cohort
    reference = saved_service.replay("a4TPwpWGN2I")
    modified = original.copy()
    cutoff = original.loc[original.creative_id.eq("a4TPwpWGN2I"), "published_at"].iloc[0]
    modified.loc[modified.published_at >= cutoff, ["target", "target_log"]] = [1e12, np.log1p(1e12)]
    try:
        saved_service.cohort = modified
        result = saved_service.replay("a4TPwpWGN2I")
        assert result["predicted_metric"] == reference["predicted_metric"]
        assert result["historical_baseline"] == reference["historical_baseline"]
    finally:
        saved_service.cohort = original


def test_http_real_replay(saved_service):
    from galaxies.api import create_app

    with patch("galaxies.api.real_service", return_value=saved_service):
        with TestClient(create_app(Settings(workspace=WORKSPACE))) as client:
            response = client.get("/api/real/replay/a4TPwpWGN2I")
            assert response.status_code == 200
            assert response.json()["evidence_level"] == "real_data_experimental"


def test_comparison_and_unsupported_channel(saved_service):
    row = saved_service.cohort.loc[saved_service.cohort.creative_id.eq("a4TPwpWGN2I")].iloc[0]
    base = RealCreativeRequest(
        creative_id="new-a",
        channel_id=row.channel_id,
        caption=row.title,
        planned_datetime=pd.Timestamp(row.published_at, unit="s", tz="UTC").to_pydatetime(),
        duration_seconds=row.duration,
        category=row.category,
    )
    result = saved_service.compare(
        RealCompareRequest(creatives=[base, base.model_copy(update={"creative_id": "new-b"})])
    )
    assert result["recommended_id"] is None
    assert len(result["ranking"]) == 2
    with pytest.raises(ValueError, match="três resultados"):
        saved_service.predict(base.model_copy(update={"channel_id": "unknown-channel"}))
    with pytest.raises(ValueError, match="mesmo canal"):
        saved_service.compare(
            RealCompareRequest(creatives=[base, base.model_copy(update={"channel_id": "other"})])
        )


def test_mcp_exposes_real_tools_without_loading_models(tmp_path):
    import asyncio

    from galaxies.mcp_server import create_mcp

    mcp, jobs = create_mcp(Settings(workspace=tmp_path))
    try:
        names = {t.name for t in asyncio.run(mcp.list_tools())}
        assert {"predict_real_creative", "compare_real_creatives", "replay_real_creative"} <= names
    finally:
        jobs.close()


def test_mcp_real_call(saved_service):
    import asyncio

    from galaxies.mcp_server import create_mcp

    with patch("galaxies.mcp_server.real_service", return_value=saved_service):
        mcp, jobs = create_mcp(Settings(workspace=WORKSPACE))
        try:
            content, structured = asyncio.run(
                mcp.call_tool("replay_real_creative", {"creative_id": "a4TPwpWGN2I"})
            )
            assert structured["model_id"] == "smtpd-frozen-2024"
            assert structured["media_used_for_prediction"] is True
        finally:
            jobs.close()


def test_uploaded_video_reuses_verified_extraction(saved_service):
    source = WORKSPACE / "public/smtpd/videos/a4TPwpWGN2I.mp4"
    row = saved_service.cohort.loc[saved_service.cohort.creative_id.eq("a4TPwpWGN2I")].iloc[0]
    request = RealCreativeRequest(
        creative_id="upload-check",
        channel_id=row.channel_id,
        caption=row.title,
        planned_datetime="2024-01-05T00:00:00Z",
        duration_seconds=row.duration,
        category=row.category,
        media_path=str(source.relative_to(WORKSPACE)),
    )
    with patch(
        "galaxies.deep_media.DeepMediaAnalyzer", side_effect=AssertionError("Reuse the extraction")
    ):
        record, warning = saved_service._media(request)
    assert record is not None
    assert warning is None


def test_video_metadata_and_real_home(tmp_path):
    from galaxies.api import create_app

    with TestClient(create_app(Settings(workspace=tmp_path))) as client:
        assert "real.js" in client.get("/").text
        assert "app.js" in client.get("/synthetic").text
        upload = client.post(
            "/api/media", files={"file": ("broken.mp4", b"invalid video", "video/mp4")}
        )
        assert upload.status_code == 201
        media_id = upload.json()["media_id"]
        assert client.get(f"/api/real/media/{media_id}").status_code == 422
