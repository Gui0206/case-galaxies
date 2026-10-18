"""Contracts that protect a genuinely external evaluation from false improvements."""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from galaxies import transfer_study as study
from galaxies.reach_study import DAY, epoch


def snapshot(video="abcdefghijk", day=7, region="BR", views=80, **changes):
    row = {
        "video_id": video,
        "title": "Original title",
        "publishedAt": "2023-01-01T00:00:00Z",
        "channelId": "channel-one",
        "categoryId": 26,
        "trending_date": f"2023-01-{day + 1:02d}T00:00:00Z",
        "view_count": views,
        "region": region,
    }
    return row | changes


def test_fixed_age_deduplication_and_first_observed_title():
    records = [
        snapshot(day=1, views=7),
        snapshot(day=6, views=60),
        snapshot(day=7, views=80, title="Later title"),
        snapshot(day=7, region="US", views=80),
        snapshot(day=8, views=900),
        snapshot(video="missingday7", day=6),
        snapshot(video="missingday7", day=8),
    ]
    frame, audit = study.build_cohort(pd.DataFrame(records))
    assert len(frame) == 1
    assert frame.target.iloc[0] == 80
    assert frame.title.iloc[0] == "Original title"
    assert frame.available_at.iloc[0] == epoch("2023-01-09")
    assert frame.metadata_observed_at.iloc[0] == epoch("2023-01-02")
    assert audit["videos_with_title_changes"] == 1


def test_conflicting_global_counts_and_changed_identity_are_not_cherry_picked():
    records = [
        snapshot(),
        snapshot(region="US", views=90),
        snapshot(video="identitybad", day=1),
        snapshot(video="identitybad", channelId="different-channel"),
        snapshot(video="validvideo1"),
    ]
    frame, audit = study.build_cohort(pd.DataFrame(records))
    assert frame.creative_id.tolist() == ["validvideo1"]
    assert audit["conflicting_target_counts"] == 1
    assert audit["inconsistent_identity_or_publication"] == 1


def test_pair_sampling_is_label_independent_and_matches_enumeration():
    frame = pd.DataFrame(
        {
            "published_at": epoch("2016-07-01") + np.arange(10) * DAY,
            "channel_id": ["a"] * 5 + ["b"] * 5,
            "target_log": np.arange(10, dtype=float),
        }
    )
    left, right, _ = study.paired_indices(frame)
    expected = set(combinations(range(5), 2)) | {(5, 6)} | set(combinations(range(7, 10), 2))
    assert set(zip(left, right, strict=True)) == expected
    capped = study.paired_indices(frame, cap=2)
    changed = frame.copy()
    changed.target_log = -1e9
    for before, after in zip(capped, study.paired_indices(changed, cap=2), strict=True):
        np.testing.assert_array_equal(before, after)
    assert len(capped[0]) == 4


def test_pair_model_learns_differences_not_channel_scale(tmp_path):
    rng = np.random.default_rng(321)
    x = rng.normal(size=(90, 3))
    channel = np.repeat(np.arange(9), 10)
    frame = pd.DataFrame(
        {
            "published_at": epoch("2016-07-01") + np.tile(np.arange(10), 9) * 3600,
            "channel_id": [str(c) for c in channel],
            "target_log": channel * 100 + x[:, 0] * 2,
        }
    )
    score = study.fit_ranker(
        "pair_linear_1.0",
        {"linear": sparse.csr_matrix(x)},
        frame,
        np.arange(60),
        np.arange(60, 90),
        tmp_path,
    )
    metric = study.ranking_scores(frame.iloc[60:], {"model": score})
    assert metric["model"] > 0.9
    coef = np.load(tmp_path / "pair_linear_1.0.npz", allow_pickle=False)["coef"]
    np.testing.assert_allclose(x[60:] @ coef, score)
    assert abs(coef[0]) > abs(coef[1]) + abs(coef[2])


def test_missing_target_modalities_are_removed_from_both_domains():
    class Encoder:
        context_names = ["median_20", "log_duration", "category:2", "language:en", "definition:hd"]

    matrices = {
        "context": np.arange(10).reshape(2, 5),
        "text": np.arange(14).reshape(2, 7),
        "retrieval": np.arange(22).reshape(2, 11),
        "semantic": np.arange(14).reshape(2, 7),
        "semantic_retrieval": np.arange(20).reshape(2, 10),
        "linear": sparse.csr_matrix(np.arange(16).reshape(2, 8)),
    }
    selected = study.model_matrices(Encoder(), matrices)
    np.testing.assert_equal(selected["context"], matrices["context"][:, [0, 2]])
    np.testing.assert_equal(selected["lexical"][:, -3:], matrices["retrieval"][:, -4:-1])
    np.testing.assert_equal(
        selected["linear"].toarray(), matrices["linear"][:, [0, 2, 5, 6, 7]].toarray()
    )


def test_external_opening_cannot_be_repeated_after_an_interruption(monkeypatch, tmp_path):
    (tmp_path / "selection.json").write_text("{}")
    monkeypatch.setattr(study, "verify_frozen", lambda *a: ({}, {}))

    def fail():
        raise RuntimeError("interruption before any metrics")

    monkeypatch.setattr(study, "source_data", fail)
    with pytest.raises(RuntimeError, match="interruption"):
        study.final(tmp_path, tmp_path)
    with pytest.raises(FileExistsError):
        study.final(tmp_path, tmp_path)


def test_analogies_cannot_read_own_or_future_outcomes():
    frame = pd.DataFrame(
        {
            "channel_id": ["a", "b", "c", "a"],
            "published_at": [0, 10, 20, 30],
            "available_at": [5, 15, 100, 35],
            "target_log": [3.0, 4.0, 1000.0, 5.0],
            "logmean_20": [2.0] * 4,
            "history_count": [4] * 4,
        }
    )
    semantic = np.array([[1.0, 0.0]] * 4, dtype=np.float32)

    def compute(data):
        return study.add_analogies(
            {"semantic": np.zeros((4, 1))},
            data,
            semantic,
            [[], [], [], [0]],
            data.iloc[:3],
            semantic[:3],
            np.array([3]),
        )["analogy"]

    original = compute(frame)
    changed = frame.copy()
    changed.loc[2:, "target_log"] = -10000
    np.testing.assert_allclose(original, compute(changed))
    assert original[3, 1] > 0  # Local mature relative performance.
    assert original[3, 5] > 0  # Other-channel mature relative performance.
