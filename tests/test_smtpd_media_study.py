import numpy as np
import pytest

from galaxies.smtpd_media_study import MediaEncoder, predict_base


def record(value):
    return {
        "duration_s": 12.0,
        "caption_embedding": [value, 3, 4, 5],
        "transcript_segments": [
            {"start": 0.0, "end": 3.0, "text": "example words", "avg_logprob": -0.2}
        ],
        "speech_features": {"has_audio": 1, "has_transcript": int(value > 0), "audio_rms": value},
        "editing": {
            "features": {"cuts_per_second": value, "shot_count": 4},
            "cut_times": [1, 2, 3],
        },
        "visual_mean": [value, 1, 2, 3],
        "visual_opening": [value, 2, 4, 1],
        "visual_variation": [value, 3, 1, 2],
        "speech_embedding": [value, 3, 4, 5],
    }


@pytest.fixture(autouse=True)
def mocked_prototypes(monkeypatch):
    monkeypatch.setattr("galaxies.smtpd_media_features.prototypes", lambda: np.eye(20, 4))


def test_media_encoder_never_refits_on_unseen_video_and_keeps_absent_audio_explicit():
    training = [record(i / 10) for i in range(6)]
    encoder = MediaEncoder("full").fit(training)
    original = encoder.transform(training)
    means = {n: p.mean_.copy() for n, p in encoder.projections.items()}
    unseen = record(100)
    unseen["speech_features"] = {"has_audio": 0, "has_transcript": 0}
    output = encoder.transform([unseen])
    assert np.isfinite(output).all()
    np.testing.assert_array_equal(original, encoder.transform(training))
    for name, expected in means.items():
        np.testing.assert_array_equal(encoder.projections[name].mean_, expected)


def test_text_control_adds_no_media_dimensions():
    encoder = MediaEncoder("text_control").fit([record(1), record(2)])
    assert encoder.transform([record(3)]).shape == (1, 0)


def test_ensemble_prediction_reproduces_selected_weight(monkeypatch):
    def fake(name, *args):
        return np.array([1.0, 3.0]) if name == "context" else np.array([5.0, 7.0])

    monkeypatch.setattr("galaxies.smtpd_media_study.selected_prediction", fake)
    selection = {
        "combinations": {
            "ensemble": {"context": "context", "content": "content", "content_weight": 0.25}
        }
    }
    value = predict_base("ensemble", selection, None, None, None, None, None, None)
    np.testing.assert_allclose(value, [2, 4])
