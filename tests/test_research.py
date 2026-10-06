import numpy as np
import pytest
from catboost import Pool

from galaxies.data import validate_split
from galaxies.demo import demo_comparison
from galaxies.io import read_json
from galaxies.metrics import interval_score
from galaxies.model import ModelBundle, train
from galaxies.protocol import make_plan, select_candidate
from galaxies.research import _shuffle_content, benchmark, final_evaluation
from galaxies.schemas import TrainConfig


@pytest.mark.parametrize("family", ["catboost", "ridge", "lightgbm", "blend"])
@pytest.mark.parametrize("target_mode", ["absolute", "channel_residual"])
def test_every_family_round_trips_with_additive_attributions(
    tmp_path, records, family, target_mode
):
    report = train(
        records,
        tmp_path,
        TrainConfig(
            family=family, target_mode=target_mode, iterations=25, channel_weighting="balanced"
        ),
    )
    bundle = ModelBundle(tmp_path / report["model_id"])
    scenario = demo_comparison()
    vector = bundle.encoder.transform(scenario.creatives[0], scenario.context, bundle.history)
    center, low, high = bundle.predict(vector.reshape(1, -1))
    contributions = bundle.point.get_feature_importance(
        Pool(vector.reshape(1, -1)), type="ShapValues"
    )
    assert contributions.sum() == pytest.approx(center[0], abs=1e-5)
    assert 0 <= low[0] <= np.expm1(center[0]) <= high[0]
    restored = ModelBundle(tmp_path / report["model_id"])
    np.testing.assert_allclose(bundle.predict(vector[None])[0], restored.predict(vector[None])[0])
    assert report["selection_verified"] is False
    assert report["eligible_for_decision"] is False


def test_rolling_protocol_never_uses_final_labels_or_overlapping_test_ids(records):
    plan, development, splits = make_plan(records)
    final_ids = set(plan["protected_final_ids"])
    assert not final_ids & {r.creative_id for r in development}
    all_tests = []
    for split in splits:
        validate_split(split, records)
        all_tests.extend(r.creative_id for r in split["parts"]["test"])
    assert len(all_tests) == len(set(all_tests))
    perturbed = [
        r.model_copy(update={"target": r.target * 10000}) if r.creative_id in final_ids else r
        for r in records
    ]
    changed, changed_development, _ = make_plan(perturbed)
    assert changed["folds"] == plan["folds"]
    assert changed_development == development
    assert changed["dataset_sha256"] != plan["dataset_sha256"]


def test_partition_validator_rejects_label_unavailable_at_cutoff(records):
    _, _, splits = make_plan(records)
    split = splits[0]
    altered = split["parts"]["train"][0].model_copy(
        update={"observed_at": split["parts"]["test"][0].published_at}
    )
    split["parts"]["train"][0] = altered
    with pytest.raises(ValueError, match="temporal"):
        validate_split(split, records)


def test_selection_does_not_accept_incomplete_competitors():
    rows = [
        {"candidate": "context", "fold": 0, "metrics": {}},
        {"candidate": "other", "fold": 1, "metrics": {}},
    ]
    with pytest.raises(ValueError, match="mesmos folds"):
        select_candidate(rows, {"context": TrainConfig()})


def test_interval_score_penalizes_uninformative_coverage():
    truth = np.zeros(10)
    assert interval_score(truth, truth - 1, truth + 1) < interval_score(
        truth, truth - 10, truth + 10
    )
    assert interval_score(truth, truth + 2, truth + 3) > interval_score(truth, truth - 1, truth + 1)


def test_simplicity_cannot_win_at_the_cost_of_much_worse_count_forecasts():
    configs = {
        "context": TrainConfig(feature_blocks=["context"]),
        "linear": TrainConfig(family="ridge"),
        "trees": TrainConfig(),
    }
    rows = [
        {
            "candidate": name,
            "fold": fold,
            "metrics": {
                "macro_channel_pair_accuracy": accuracy,
                "macro_channel_mae_log": error,
                "interval_score_log": 2.0,
            },
        }
        for fold in range(3)
        for name, accuracy, error in [
            ("context", 0.5, 0.6),
            ("linear", 0.8, 0.5),
            ("trees", 0.8, 0.3),
        ]
    ]
    selection = select_candidate(rows, configs)
    assert selection["selected"] == "trees"
    assert selection["macro_mae_log_ceiling"] == pytest.approx(0.33)


def test_content_permutation_preserves_context_and_outcomes(records):
    shuffled = _shuffle_content(records, 77)
    assert any(a.features != b.features for a, b in zip(records, shuffled, strict=True))
    for a, b in zip(records, shuffled, strict=True):
        assert (a.channel_id, a.published_at, a.observed_at, a.target) == (
            b.channel_id,
            b.published_at,
            b.observed_at,
            b.target,
        )
        assert any(
            source.features == b.features for source in records if source.channel_id == b.channel_id
        )


def test_selection_to_final_evaluation_is_auditable_and_single_use(tmp_path, records, monkeypatch):
    configs = {
        "context_ridge": TrainConfig(family="ridge", feature_blocks=["context"], iterations=20),
        "ridge": TrainConfig(family="ridge", iterations=20),
    }
    monkeypatch.setattr("galaxies.research.candidate_catalog", lambda *_: configs)
    model_root, report_root = tmp_path / "models", tmp_path / "reports"
    result = benchmark(records, model_root, report_root, iterations=20, folds=2)
    folder = report_root / result["run_id"]
    assert result["final_test_used"] is False
    assert read_json(folder / "selection.json")["selected"] == result["selection"]["selected"]
    with monkeypatch.context() as changed_code:
        changed_code.setattr("galaxies.research.implementation_fingerprint", lambda: "changed")
        with pytest.raises(ValueError, match="Código/dependências"):
            final_evaluation(records, folder, model_root)
    modified = [records[0].model_copy(update={"target": records[0].target + 1}), *records[1:]]
    with pytest.raises(ValueError, match="Dataset diferente"):
        final_evaluation(modified, folder, model_root)
    outcome = final_evaluation(records, folder, model_root)
    assert outcome["eligible_for_decision"] is False
    assert outcome["checks"]["real_data"] is False
    bundle = ModelBundle(model_root / outcome["model_id"])
    assert bundle.report["selection_verified"] is True
    assert bundle.report["incremental_content_gain"] == outcome["content_gain"]
    with pytest.raises(ValueError, match="já foi aberto"):
        final_evaluation(records, folder, model_root)
