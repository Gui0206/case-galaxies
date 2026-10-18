import pandas as pd
import pytest

from galaxies.io import sha256_file, write_json
from galaxies.reach_study import epoch
from galaxies.smtpd_final import final, validate_development, verify


def test_reserved_test_cannot_be_opened_a_second_time(tmp_path, monkeypatch):
    (tmp_path / "evaluation-opened.json").write_text("{}")
    (tmp_path / "selection.json").write_text("{}")
    monkeypatch.setattr("galaxies.smtpd_final.verify", lambda _: {})

    def reject(*args, **kwargs):
        raise AssertionError("No data may be loaded after repeated opening")

    monkeypatch.setattr("galaxies.smtpd_final.data", reject)
    with pytest.raises(FileExistsError):
        final(tmp_path)


def test_frozen_evaluation_rejects_code_or_input_changes(tmp_path, monkeypatch):
    predictions = tmp_path / "development.csv"
    predictions.write_text("creative_id,prediction\na,1\n")
    write_json(
        tmp_path / "selection.json",
        {"code": {"a": 1}, "inputs": {"b": 2}, "predictions_sha256": sha256_file(predictions)},
    )
    monkeypatch.setattr("galaxies.smtpd_final.code_identity", lambda: {"a": 1})
    monkeypatch.setattr("galaxies.smtpd_final.inputs", lambda: {"b": 2})
    assert verify(tmp_path)["code"] == {"a": 1}
    monkeypatch.setattr("galaxies.smtpd_final.inputs", lambda: {"b": 3})
    with pytest.raises(ValueError, match="changed"):
        verify(tmp_path)


def test_development_cannot_silently_include_a_2024_video():
    frame = pd.DataFrame(
        {
            "creative_id": ["valid", "heldout"],
            "published_at": [epoch("2023-11-18"), epoch("2024-01-10")],
            "available_at": [epoch("2023-11-26"), epoch("2024-01-18")],
            "history_count": [3, 3],
            "target_log": [1.0, 2.0],
        }
    )
    good = frame.iloc[[0]].set_index("creative_id")
    validate_development(good, frame)
    with pytest.raises(ValueError, match="unexpected"):
        validate_development(frame.set_index("creative_id"), frame)
