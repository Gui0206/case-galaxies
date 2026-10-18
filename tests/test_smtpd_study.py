import numpy as np
import pandas as pd

from galaxies.reach_study import DAY, epoch, history_features
from galaxies.smtpd_study import enrich, partitions, weighted_median, within_prediction


def sample_frame():
    rows = []
    for channel in range(3):
        for i in range(9):
            date = epoch("2023-09-01") + i * 9 * DAY
            rows.append(
                {
                    "creative_id": f"{channel}-{i}",
                    "channel_id": str(channel),
                    "published_at": date,
                    "available_at": date + 8 * DAY,
                    "target_log": channel * 3 + i / 10,
                    "duration": 20 + i,
                    "title": "Example video",
                    "category": "2",
                    "reserved_channel": False,
                }
            )
    return history_features(pd.DataFrame(rows))


def test_future_labels_cannot_change_matched_history_or_past_prediction():
    frame, neighbors = sample_frame()
    train = np.flatnonzero(frame.available_at < epoch("2023-10-10"))
    target = np.flatnonzero(
        (frame.published_at >= epoch("2023-10-10")) & (frame.published_at < epoch("2023-10-20"))
    )
    x = np.log1p(frame[["duration"]].to_numpy())
    before = within_prediction(x, frame, train, neighbors, target, 10)
    altered = frame.copy()
    altered.loc[altered.published_at >= epoch("2023-10-10"), "target_log"] = 1000
    after = within_prediction(x, altered, train, neighbors, target, 10)
    np.testing.assert_allclose(before, after)
    a, b = enrich(frame, neighbors), enrich(altered, neighbors)
    np.testing.assert_allclose(a.loc[target, "duration_05"], b.loc[target, "duration_05"])


def test_final_train_excludes_calibration_even_if_labels_mature():
    frame = pd.DataFrame(
        {
            "published_at": [
                epoch(x) for x in ["2023-12-22", "2023-12-23", "2023-12-27", "2024-01-05"]
            ],
            "available_at": [
                epoch(x) for x in ["2023-12-30", "2023-12-31", "2024-01-04", "2024-01-13"]
            ],
            "history_count": [3, 3, 3, 3],
        }
    )
    all_train, warm, test = partitions(frame, "2024-01-05", final_train=True)
    assert all_train.tolist() == warm.tolist() == [0]
    assert test.tolist() == [3]


def test_weighted_median_resists_one_extreme_and_uses_weights():
    assert weighted_median([1, 2, 1000], [1, 1, 1]) == 2
    assert weighted_median([1, 2, 1000], [10, 1, 1]) == 1
