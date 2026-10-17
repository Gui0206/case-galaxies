"""Temporal isolation and modality ablation contracts; fixtures are artificial."""

import json

import numpy as np
import pandas as pd
import pytest

from galaxies.io import write_json
from galaxies.retention_benchmark import (
    FusionEncoder,
    develop,
    epoch,
    evaluate_final,
    make_plan,
    paired_gain,
    validate_media_record,
)
from galaxies.shortvideo import temporal_context


def test_history_waits_until_viewing_is_finished_and_ignores_own_outcome():
    events = pd.DataFrame(
        {
            "pid": [1, 2, 3, 4],
            "author_id": [1, 1, 1, 1],
            "root_id": [2] * 4,
            "exposed_time": [0, 8, 10, 30],
            "outcome_available": [3, 14, 12, 35],
            "retention": [0.3, 1.0, 0.2, 0.8],
        }
    )
    rows = pd.DataFrame({"author_id": [1], "root_id": [2], "first_seen": [10]}, index=[3])
    context = temporal_context(events, rows)
    assert context.loc[3, "channel_prior_events"] == 1
    assert context.loc[3, "history_raw"] == 0.3
    future = events.copy()
    future.loc[1:, "retention"] = 0
    pd.testing.assert_frame_equal(context, temporal_context(future, rows))


def fixture(n=420):
    rng = np.random.default_rng(23)
    starts = epoch("2022-09-16") + np.arange(n) * 1200
    frame = pd.DataFrame(
        {
            "creative_id": [str(i) for i in range(n)],
            "channel_id": [str(i % 60) for i in range(n)],
            "first_seen": starts,
            "available_at": starts + 86400,
            "duration": rng.uniform(10, 90, n),
            "root_id": np.arange(n) % 3,
            "exposures": 15,
            "channel_prior_mean": 0.5,
            "channel_prior_std": 0.1,
            "channel_prior_events": 20,
            "category_prior_mean": 0.5,
            "category_prior_std": 0.1,
            "category_prior_events": 100,
            "history_raw": 0.5,
            "history_shrunk": 0.5,
            "global_prior_mean": 0.5,
            "target": rng.uniform(0.1, 0.9, n),
        }
    )
    records = {}
    for video in frame.creative_id:
        duration = float(frame.loc[int(video), "duration"])
        records[video] = {
            "content_hash": video,
            "extractor_fingerprint": "fixture",
            "duration_s": duration,
            "frame_times": [0.0, duration - 0.1],
            "transcript_segments": [],
            **{
                key: rng.normal(size=6).tolist()
                for key in (
                    "visual_mean",
                    "visual_opening",
                    "visual_variation",
                    "caption_embedding",
                    "speech_embedding",
                )
            },
            "speech_features": {"has_audio": 1, "audio_rms": 0.1},
            "editing": {
                "features": {"shot_count": 3, "motion_mean": 0.2},
                "decoded_duration_s": duration,
                "decoded_frames": int(duration * 30),
                "cut_times": [1.0, 2.0],
            },
        }
    return frame, records


def test_incomplete_decoding_or_non_finite_content_cannot_enter_the_study():
    _, features = fixture(1)
    record = features["0"]
    assert validate_media_record(record)["visual_mean"] == 6
    record["editing"]["decoded_duration_s"] = 2
    with pytest.raises(ValueError, match="vídeo completo"):
        validate_media_record(record)
    record["editing"]["decoded_duration_s"] = record["duration_s"]
    record["speech_embedding"][0] = float("nan")
    with pytest.raises(ValueError, match="semântica"):
        validate_media_record(record)


def test_temporal_plan_purges_immature_labels_and_does_not_use_final_targets():
    frame, _ = fixture()
    first = make_plan(frame, {})
    final = first["final"]["test"]
    frame.loc[final, "target"] = 999
    second = make_plan(frame, {})
    assert first["folds"] == second["folds"]
    assert first["final"] == second["final"]
    seen = set()
    for fold in first["folds"]:
        assert (frame.loc[fold["train"], "available_at"] < fold["decision_start"]).all()
        assert not (seen & set(fold["validation"]))
        seen.update(fold["validation"])
    assert not (seen & set(final))


def test_encoder_is_fit_only_on_train_and_ablation_really_removes_modalities():
    frame, features = fixture(100)
    encoder = FusionEncoder("full")
    encoder.fit_transform(frame.iloc[:70], features)
    before = {name: pca.components_.copy() for name, pca in encoder.projections.items()}
    changed = {key: dict(value) for key, value in features.items()}
    for video in frame.iloc[70:].creative_id:
        changed[video]["visual_mean"] = [1000] * 6
    encoder.transform(frame.iloc[70:], changed)
    for name, pca in encoder.projections.items():
        np.testing.assert_array_equal(before[name], pca.components_)
    context = FusionEncoder("context")
    context.fit_transform(frame.iloc[:70], features)
    np.testing.assert_array_equal(
        context.transform(frame.iloc[70:], features), context.transform(frame.iloc[70:], changed)
    )
    assert not context.projections


def test_bootstrap_measures_paired_gain_in_percentage_points():
    frame, _ = fixture(100)
    frame["target"] = 0.4
    result = paired_gain(frame, np.full(100, 0.4), np.full(100, 0.5))
    assert result["mean_error_reduction_pp"] == pytest.approx(10)
    assert result["ci95_low"] == pytest.approx(10)
    assert result["ci95_high"] == pytest.approx(10)


def test_frozen_final_is_single_use_and_source_changes_are_rejected(tmp_path, monkeypatch):
    import galaxies.retention_benchmark as benchmark

    monkeypatch.setattr(benchmark, "CANDIDATES", {"ridge_10": {"family": "ridge", "alpha": 10}})
    root = tmp_path / "data"
    feature_folder = root / "features"
    feature_folder.mkdir(parents=True)
    frame, records = fixture()
    frame.to_parquet(root / "cohort.parquet", index=False)
    write_json(root / "acquisition.json", {"artificial_unit_test": True})
    write_json(feature_folder / "extractor.json", {"fingerprint": "fixture"})
    index = {
        "extractor_fingerprint": "fixture",
        "rows": {},
        "failures": {},
        "extraction_complete": True,
    }
    for video, record in records.items():
        write_json(feature_folder / f"{video}.json", record)
        index["rows"][video] = {"file": f"{video}.json", "content_hash": video}
    write_json(feature_folder / "index.json", index)
    folder = tmp_path / "run"
    develop(root, folder)
    with monkeypatch.context() as patch:
        patch.setattr(benchmark, "code_hash", lambda: "changed")
        with pytest.raises(ValueError, match="Código"):
            evaluate_final(root, folder)
    report = evaluate_final(root, folder)
    assert report["eligible_for_production"] is False
    assert report["metrics"]["full"]["videos"] == report["test_videos"]
    with pytest.raises(FileExistsError):
        evaluate_final(root, folder)
    assert json.loads((folder / "final.json").read_text())["test_videos"] > 50
