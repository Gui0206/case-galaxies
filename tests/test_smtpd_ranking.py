import numpy as np
import pandas as pd

from galaxies.reach_study import epoch
from galaxies.smtpd_ranking import training_pairs


def test_ranking_pairs_preserve_channel_week_maturity_and_equal_channel_weight():
    day = 86400
    date = epoch("2023-11-03")
    frame = pd.DataFrame(
        {
            "creative_id": list("abcdefgh"),
            "channel_id": ["a", "a", "a", "a", "b", "b", "a", "a"],
            "published_at": [
                date,
                date + day,
                date + 2 * day,
                date + 3 * day,
                date,
                date + day,
                date + 9 * day,
                date + 9 * day,
            ],
            "target_log": [1, 2, 3, 4, 1, 2, 10, 20],
        }
    )
    chosen, pairs, weights = training_pairs(frame, np.arange(7))
    assert len(pairs) == 7
    a, b = pairs.T
    assert (chosen.iloc[a].channel_id.to_numpy() == chosen.iloc[b].channel_id.to_numpy()).all()
    assert (chosen.iloc[a].week.to_numpy() == chosen.iloc[b].week.to_numpy()).all()
    assert (chosen.iloc[a].target_log.to_numpy() > chosen.iloc[b].target_log.to_numpy()).all()
    assert "h" not in chosen.creative_id.to_list()
    sums = pd.Series(weights).groupby(chosen.iloc[a].channel_id.to_numpy()).sum()
    np.testing.assert_allclose(sums.iloc[0], sums.iloc[1])
