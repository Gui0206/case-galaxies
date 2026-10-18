import numpy as np
import pandas as pd

from galaxies import modern_adaptation as study
from galaxies.reach_study import epoch


def test_training_rejects_unmatured_and_without_history():
    boundary = epoch("2022-05-01")
    frame = pd.DataFrame(
        {
            "reserved_channel": [False] * 4,
            "available_at": [boundary - 1, boundary, boundary - 1, boundary - 1],
            "history_count": [1, 3, 0, 4],
        }
    )
    np.testing.assert_array_equal(study.training_rows(frame, "2022-05-01"), [0, 3])
    assert epoch(study.CALIBRATION[1]) + 9 * 86400 < epoch(study.TEST_START)


def test_selection_balances_channels_instead_of_upload_frequency():
    data = pd.DataFrame({"channel_id": ["large"] * 20 + ["small"], "target_log": np.zeros(21)})
    for n in study.BASELINES:
        data[n] = 3.0
    for m in study.MODELS:
        for a in study.SHRINKAGE:
            data[study.blended_name(m, a)] = 3.0
    data["lexical_mae3@1"] = [0] * 20 + [4]
    data["semantic_mae3@1"] = [1] * 21
    selected = study.select(data)
    assert selected["content"] == "semantic_mae3@1"
