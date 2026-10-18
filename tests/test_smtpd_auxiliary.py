import numpy as np
import pandas as pd

from galaxies.reach_study import DAY, TEXT_STATS, history_features
from galaxies.smtpd_auxiliary import pure_features, sufficient_statistics


def test_within_channel_statistics_cannot_learn_channel_size():
    frame = pd.DataFrame(
        {
            "channel_id": ["a"] * 6 + ["b"] * 6,
            "published_at": np.tile(np.arange(6) * DAY, 2),
            "target_log": [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15],
        }
    )
    x = np.tile(np.arange(6), 2)[:, None]
    gram, rhs, count = sufficient_statistics(x, frame, np.arange(12))
    altered = frame.copy()
    altered.loc[altered.channel_id.eq("b"), "target_log"] += 1000
    g2, h2, c2 = sufficient_statistics(x, altered, np.arange(12))
    np.testing.assert_allclose(gram, g2)
    np.testing.assert_allclose(rhs, h2)
    assert count == c2 == 12


def test_auxiliary_pure_features_match_primary_prepublication_definition():
    frame = pd.DataFrame(
        {
            "creative_id": ["a", "b"],
            "channel_id": ["c", "c"],
            "published_at": [1_700_000_000, 1_701_000_000],
            "available_at": [1_700_700_000, 1_701_700_000],
            "target_log": [2, 4],
            "duration": [30, 200],
            "title": ["3 makeups?", "Demo!"],
            "category": ["26", "26"],
            "reserved_channel": [False, False],
        }
    )
    frame, _ = history_features(frame)
    semantic = np.array([[1.0, 2.0], [3.0, 4.0]])
    x = pure_features(frame, semantic)
    np.testing.assert_allclose(
        x[:, 6:10], frame[["hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]]
    )
    np.testing.assert_allclose(x[:, 10:15], frame[TEXT_STATS])
    np.testing.assert_allclose(x[:, 15:], semantic)
