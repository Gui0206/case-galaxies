"""Public-export provenance, time isolation, and one-use evaluation guards."""

import json

import numpy as np
import pandas as pd
import pytest

from galaxies.public_benchmark import (
    PublicEncoder,
    benchmark_plan,
    develop,
    error_gain_uncertainty,
    evaluate_final,
    history_features,
    load_kenjee,
    load_trending,
    prepare_public,
)


def real_shape_frame(n=160):
    """Artificial test fixture; never an empirical research result."""
    dates = pd.date_range("2020-01-01", periods=n, freq="4D", tz="UTC")
    return pd.DataFrame(
        {
            "creative_id": [f"fixture-{i}" for i in range(n)],
            "channel_id": [f"channel-{i % 2}" for i in range(n)],
            "category": ["Education"] * n,
            "published_at": dates,
            "observed_at": dates + pd.Timedelta(days=8),
            "caption": [f"topic {i % 9} unique {i}" for i in range(n)],
            "duration_s": np.arange(n) + 30.0,
            "target": (np.arange(n) * 17 % 90 + 20),
        }
    )


def test_incomplete_calendar_weeks_are_excluded_not_zero_filled(tmp_path):
    pd.DataFrame(
        {
            "Video": ["complete", "missing", "Total"],
            "Video publish time": ["Sep 1, 2020", "Sep 1, 2020", None],
        }
    ).to_csv(tmp_path / "Aggregated_Metrics_By_Video.csv", index=False)
    rows = []
    for video, days in [("complete", range(7)), ("missing", range(6))]:
        for day in days:
            rows.append(
                {
                    "External Video ID": video,
                    "Date": f"{day + 1} Sept 2020",
                    "Video Title": video,
                    "Video Length": 120,
                    "Views": 10,
                    "Average View Percentage": 99,
                    "Video Likes Added": 800,
                }
            )
    pd.DataFrame(rows).to_csv(tmp_path / "Video_Performance_Over_Time.csv", index=False)
    frame, _ = load_kenjee(tmp_path)
    assert frame.creative_id.tolist() == ["complete"]
    assert frame.target.iloc[0] == 70
    assert frame.observed_at.iloc[0] == pd.Timestamp("2020-09-09", tz="UTC")
    assert "Average View Percentage" not in frame.columns


def test_trending_uses_first_title_and_fixed_day_target_deduplicated(tmp_path):
    base = {
        "video_id": "video-one",
        "publish_time": "2018-01-01T12:00:00Z",
        "channel_title": "Channel",
        "category_id": 27,
        "video_error_or_removed": False,
    }
    rows = [
        {**base, "trending_date": "18.02.01", "title": "Original", "views": 10},
        {**base, "trending_date": "18.08.01", "title": "Edited later", "views": 700},
        {**base, "trending_date": "18.09.01", "title": "Edited later", "views": 999999},
    ]
    for region in ("US", "CA", "GB"):
        pd.DataFrame(rows).to_csv(tmp_path / f"{region}videos.csv", index=False)
    frame, audit = load_trending(tmp_path)
    assert len(frame) == 1
    assert frame.iloc[0].caption == "Original"
    assert frame.iloc[0].target == 700
    assert frame.iloc[0].observed_at == pd.Timestamp("2018-01-09", tz="UTC")
    assert audit["raw_rows"] == 9


def test_history_excludes_own_unmatured_and_future_outcomes():
    frame = real_shape_frame(12)
    old = history_features(frame)
    assert old.history_count.iloc[0] == 0
    assert old.history_count.iloc[2] == 0  # At exactly maturity, strict < excludes it.
    frame.loc[8:, "target"] = 10000000
    new = history_features(frame)
    pd.testing.assert_frame_equal(old.iloc[:9], new.iloc[:9])


def test_final_targets_cannot_change_development_or_encoder():
    frame, audit = prepare_public(real_shape_frame())
    plan = benchmark_plan(frame, audit, {})
    ids = plan["final_parts"]["test"]
    changed = frame.copy()
    changed.loc[ids, "target"] *= 10000
    new_plan = benchmark_plan(changed, audit, {})
    assert plan["development_folds"] == new_plan["development_folds"]
    last = plan["development_folds"][-1]["parts"]
    dev_ids = sum(last.values(), [])
    old, new = history_features(frame), history_features(changed)
    pd.testing.assert_frame_equal(old.loc[dev_ids], new.loc[dev_ids])
    encoder = PublicEncoder(True)
    encoder.fit(frame.loc[last["train"]], old.loc[last["train"]])
    components = encoder.svd.components_.copy()
    changed.loc[ids, "caption"] = "unseen words changed after training"
    encoder.transform(changed.loc[ids], new.loc[ids])
    np.testing.assert_array_equal(components, encoder.svd.components_)


def test_same_title_duplicates_do_not_cross_partitions():
    frame = real_shape_frame(20)
    frame.loc[18, ["caption", "channel_id"]] = frame.loc[0, ["caption", "channel_id"]]
    output, audit = prepare_public(frame)
    assert len(output) == 19
    assert "fixture-18" not in output.creative_id.tolist()
    assert audit["same_channel_title_duplicates_removed"] == 1


def test_real_benchmark_freezes_source_and_opens_final_once(tmp_path, monkeypatch):
    frame, audit = prepare_public(real_shape_frame())
    audit["source_audit"] = {"limitations": ["unit-test fixture, not real evidence"]}
    plan = benchmark_plan(frame, audit, {})
    plan["candidates"] = {
        k: {**v, "iterations": 20}
        for k, v in plan["candidates"].items()
        if k in ("catboost_residual", "context_catboost")
    }
    folder = tmp_path / "run"
    develop(frame, folder, plan)
    with monkeypatch.context() as context:
        context.setattr("galaxies.public_benchmark._code_fingerprint", lambda: "changed")
        with pytest.raises(ValueError, match="mudaram"):
            evaluate_final(frame, folder)
    result = evaluate_final(frame, folder)
    assert result["eligible_for_production"] is False
    assert result["metrics"]["channel_median"]["rows"] == len(plan["final_parts"]["test"])
    with pytest.raises(FileExistsError):
        evaluate_final(frame, folder)
    assert json.loads((folder / "final.json").read_text())["selected"] == result["selected"]


def test_error_uncertainty_is_paired_and_uses_blocks_for_one_creator():
    frame = real_shape_frame()
    frame["channel_id"] = "one"
    exact = np.log1p(frame.target.to_numpy())
    result = error_gain_uncertainty(frame, exact, exact + 1)
    assert result["ci95_low"] == pytest.approx(1.0)
    assert result["resampling_unit"].startswith("28-day")
