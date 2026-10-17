"""Leakage, cohort, uncertainty and final-opening invariants for the new reach study."""

import json

import numpy as np
import pandas as pd
import pytest

from galaxies import reach_study as study
from galaxies.io import sha256_file, write_json


def source_record(day=1, channel="channel", video="video"):
    return {
        "id": video,
        "snippet": {
            "publishedAt": f"2016-07-{day:02}T12:00:00Z",
            "title": "A new car review",
            "channelId": channel,
            "categoryId": "2",
            "detectLang": "en",
        },
        "contentDetails": {"duration": "PT2M30S", "definition": "hd"},
        "insights": {
            "startDate": f"2016-07-{day:02}",
            "days": "0,1,2,3,4,5,6,7",
            "dailyView": "1,2,3,4,5,6,7,999999",
        },
    }


def test_explicit_calendar_week_excludes_later_popularity():
    item = source_record()
    row = study.parse_record(item)
    assert row["target"] == 28
    assert row["available_at"] == study.epoch("2016-07-10")
    assert row["duration"] == 150
    item["insights"]["days"] = "0,1,2,3,4,5,7,8"
    with pytest.raises(ValueError, match="missing_week_day"):
        study.parse_record(item)
    item = source_record()
    item["insights"]["startDate"] = "2016-07-02"
    with pytest.raises(ValueError, match="reporting_start"):
        study.parse_record(item)


def test_history_excludes_unavailable_and_reserved_channel_outcomes():
    known = next(f"c{i}" for i in range(100) if not study.reserved_channel(f"c{i}"))
    hidden = next(f"c{i}" for i in range(100) if study.reserved_channel(f"c{i}"))
    rows = [
        study.parse_record(source_record(day, channel, f"{channel}-{day}"))
        for channel in (known, hidden)
        for day in (1, 2, 10, 11)
    ]
    for row in rows:
        if row["reserved_channel"]:
            row["target_log"] = 1000
    frame, neighbors = study.history_features(pd.DataFrame(rows))
    query = frame[
        (frame.channel_id == known) & (frame.published_at == study.epoch("2016-07-10") + 43200)
    ].iloc[0]
    assert query.history_count == 1
    assert query.category_prior == pytest.approx(np.log1p(28))
    assert (frame[frame.reserved_channel].history_count == 0).all()
    for i, prior in enumerate(neighbors):
        assert all(frame.available_at.iloc[j] < frame.published_at.iloc[i] for j in prior)
        assert i not in prior
    before = frame[study.CONTEXT].copy()
    # Later outcomes, even extreme ones, cannot change a prior forecast's context.
    future = pd.DataFrame(rows)
    future.loc[future.published_at == future.published_at.max(), "target_log"] = 1234
    after, _ = study.history_features(future)
    pd.testing.assert_frame_equal(before, after[study.CONTEXT])


def test_text_representation_is_fit_only_on_train(tmp_path):
    channel = next(f"c{i}" for i in range(100) if not study.reserved_channel(f"c{i}"))
    rows = [study.parse_record(source_record(i + 1, channel, f"v{i}")) for i in range(16)]
    for i, row in enumerate(rows[:12]):
        row["title"] = f"A new car review model {i % 3}"
    for row in rows[12:]:
        row["title"] = "zqxvfutureonly"
    frame, neighbors = study.history_features(pd.DataFrame(rows))
    encoder = study.ReachEncoder()
    semantic = np.random.default_rng(42).normal(size=(16, 8)).astype(np.float32)
    semantic /= np.linalg.norm(semantic, axis=1, keepdims=True)
    matrices = encoder.fit_transform(frame, np.arange(12), neighbors, semantic)
    assert "zqx" not in encoder.vectorizer.vocabulary_
    assert matrices["context"].shape[0] == 16
    assert matrices["retrieval"].shape[1] == matrices["text"].shape[1] + 4
    encoder.save(tmp_path)
    restored = study.ReachEncoder.load(tmp_path).transform(frame, neighbors, semantic)
    for name, matrix in matrices.items():
        a = matrix.toarray() if hasattr(matrix, "toarray") else matrix
        b = restored[name].toarray() if hasattr(restored[name], "toarray") else restored[name]
        np.testing.assert_allclose(a, b, atol=1e-6)


def test_pair_ranking_counts_prediction_ties_and_caps_by_channel():
    n = 60
    frame = pd.DataFrame(
        {
            "creative_id": [f"v{i}" for i in range(n)],
            "channel_id": "a",
            "published_at": study.epoch("2016-08-20") + np.arange(n),
            "target_log": np.arange(n),
        }
    )
    predictions = {"content": np.arange(n), "context": np.zeros(n), "history": -np.arange(n)}
    result = study.ranking(frame, predictions)
    assert result["pairs"] == 500
    assert result["macro_accuracy"] == {"content": 1, "context": 0.5, "history": 0}
    assert result["content_vs_context_ci95"] == [0.5, 0.5]


def test_channel_bootstrap_and_finite_sample_calibration():
    frame = pd.DataFrame({"channel_id": ["a", "a", "b"], "target_log": [1.0, 1.0, 1.0]})
    candidate, reference = np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 4.0])
    result = study.paired_gain(frame, candidate, reference)
    assert result["macro_log_error_reduction"] == 2  # equal channel weight, not row mean 5/3
    assert result["ci95"][0] > 0
    assert study.conformal_radius(np.arange(19)) == 17
    with pytest.raises(ValueError, match="Insufficient"):
        study.conformal_radius(np.arange(3))


def test_complete_frozen_study_and_single_final_opening(tmp_path, monkeypatch):
    root, run = tmp_path / "data", tmp_path / "reports" / "study"
    root.mkdir()
    channels = [f"channel-{i}" for i in range(40)]
    rows = []
    for c, channel in enumerate(channels):
        for day in range(62):
            published = pd.Timestamp(
                study.epoch("2016-07-01") + day * study.DAY + 43200, unit="s", tz="UTC"
            )
            item = source_record(channel=channel, video=f"{c}-{day}")
            item["snippet"]["publishedAt"] = published.isoformat()
            item["snippet"]["title"] = f"Review car model {day % 4} number {day}"
            item["insights"]["startDate"] = published.date().isoformat()
            item["insights"]["dailyView"] = ",".join([str(10 + c + day % 4)] * 8)
            rows.append(study.parse_record(item))
    frame = pd.DataFrame(rows)
    frame.to_parquet(root / "cohort.parquet", index=False)
    write_json(root / "audit.json", {"cohort_sha256": sha256_file(root / "cohort.parquet")})
    monkeypatch.setattr(
        study, "CANDIDATES", {k: study.CANDIDATES[k] for k in ["context_mae4", "text_mae4"]}
    )
    real_model = study.CatBoostRegressor

    def small_model(**kwargs):
        kwargs["iterations"] = 3
        return real_model(**kwargs)

    monkeypatch.setattr(study, "CatBoostRegressor", small_model)
    selection = study.develop(root, run)
    assert selection["context"] == "context_mae4"
    with pytest.raises(ValueError, match="new directory"):
        study.develop(root, run)
    report = study.final(root, run)
    assert report["production_eligible"] is False
    assert report["sections"]["unseen_channels"]["metrics"]["content"]["videos"] > 0
    with pytest.raises(FileExistsError):
        study.final(root, run)
    value = json.loads((run / "development.json").read_text())
    value["changed"] = True
    (run / "development.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="artifacts changed"):
        study.verify_frozen(root, run)


def test_sparse_linear_forecast_survives_native_persistence(tmp_path):
    channel = next(f"c{i}" for i in range(100) if not study.reserved_channel(f"c{i}"))
    rows = [study.parse_record(source_record(i + 1, channel, f"v{i}")) for i in range(18)]
    for i, row in enumerate(rows):
        row["title"] = f"A new car model {i % 3}"
        row["target_log"] += i % 3
    frame, neighbors = study.history_features(pd.DataFrame(rows))
    train, predict = np.arange(12), np.arange(12, 18)
    matrices = study.ReachEncoder().fit_transform(frame, train, neighbors)
    before = study.fit_candidate("linear_text_10", matrices, frame, train, predict, tmp_path)
    after = study.predict_saved_candidate("linear_text_10", matrices, frame, predict, tmp_path)
    np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)


def test_comparable_rule_uses_only_mature_labels_and_persists(tmp_path):
    channel = next(f"c{i}" for i in range(100) if not study.reserved_channel(f"c{i}"))
    rows = [study.parse_record(source_record(i + 1, channel, f"v{i}")) for i in range(24)]
    for i, row in enumerate(rows):
        row["title"] = "Car review sedan" if i % 2 else "Car review truck"
        row["target_log"] = 2 + i % 2
    frame, neighbors = study.history_features(pd.DataFrame(rows))
    train, predict = np.arange(12), np.arange(18, 24)
    encoder = study.ReachEncoder()
    matrices = encoder.fit_transform(frame, train, neighbors)
    name = "lexical_rule_50"
    before = study.fit_candidate(name, matrices, frame, train, predict, tmp_path)
    after = study.predict_saved_candidate(name, matrices, frame, predict, tmp_path)
    np.testing.assert_allclose(before, after)
    # All six query outcomes are immature for every query in this period.
    changed = frame.copy()
    changed.loc[predict, "target_log"] = 100
    perturbed = encoder.transform(changed, neighbors)
    np.testing.assert_allclose(matrices["lexical_rule"], perturbed["lexical_rule"])
    declaration = tmp_path / f"{name}.json"
    declaration.write_text("{}")
    with pytest.raises(ValueError, match="rule changed"):
        study.predict_saved_candidate(name, matrices, frame, predict, tmp_path)
