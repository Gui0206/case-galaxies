import pytest

from galaxies.config import Settings
from galaxies.io import write_json
from galaxies.quality import extraction_agreement, operational_report
from galaxies.schemas import Creative


@pytest.mark.parametrize(
    "feature,value",
    [
        ("cta", 0.5),
        ("demonstration", 8),
        ("brightness", 3),
        ("duration_s", 0),
        ("ocr_word_count", -1),
    ],
)
def test_invalid_feature_scales_are_rejected(feature, value):
    with pytest.raises(ValueError):
        Creative(creative_id="invalid", features={feature: value})


def test_extractor_agreement_excludes_missing_values_and_preserves_zero():
    human = [
        {"creative_id": "a", "features": {"cta": 0, "hook_clarity": 8}},
        {"creative_id": "b", "features": {"cta": 1, "hook_clarity": 2}},
        {"creative_id": "c", "features": {"cta": 1}},
    ]
    machine = [
        {"creative_id": "a", "features": {"cta": 0, "hook_clarity": 6}},
        {"creative_id": "b", "features": {"cta": 0, "hook_clarity": None}},
    ]
    result = extraction_agreement(human, machine)["by_feature"]
    assert result["cta"]["n"] == 2
    assert result["cta"]["agreement"] == 0.5
    assert result["cta"]["kappa"] == 0
    assert result["hook_clarity"] == {"n": 1, "mae": 2.0}


def test_extractor_agreement_rejects_nonfinite_values():
    reference = [{"creative_id": "a", "features": {"duration_s": 1}}]
    extracted = [{"creative_id": "a", "features": {"duration_s": float("nan")}}]
    with pytest.raises(ValueError, match="finitos"):
        extraction_agreement(reference, extracted)


def test_monitor_separates_evidence_and_counts_missing_attributes(tmp_path):
    settings = Settings(workspace=tmp_path).prepare()
    for name, level, duration in [("a", "synthetic_demo", 20), ("b", "retrospective", 80)]:
        write_json(
            settings.workspace / "runs" / f"{name}.json",
            {
                "prediction": {"confidence_flag": "insufficient", "evidence_level": level},
                "features": {"hook_clarity": None, "cta": 0.0},
                "duration_ms": duration,
            },
        )
    result = operational_report(settings)
    assert result["predictions"] == 2
    assert result["evidence_levels"] == {"synthetic_demo": 1, "retrospective": 1}
    assert result["missing_feature_counts"] == {"hook_clarity": 2}
    assert result["latency_ms_p50"] == 50
    assert result["prospective"]["registered"] == 0
