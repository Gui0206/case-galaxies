"""Extraction checks use original generated media, never benchmark labels."""

import importlib.util
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

from galaxies.deep_media import audio_statistics, frame_times
from galaxies.video_structure import shot_statistics


def test_frame_sampling_covers_opening_and_end_without_invalid_timestamps():
    for duration in (0.05, 1.0, 30, 180):
        times = frame_times(duration)
        assert times == sorted(set(times))
        assert times[0] == 0
        assert all(0 <= t < duration for t in times)
        assert duration - times[-1] <= 0.21
    assert all(t in frame_times(60) for t in (0.5, 1.5, 2.5))
    with pytest.raises(ValueError):
        frame_times(0)


def test_audio_silence_is_measured_in_windows_not_zero_crossing_samples():
    samples = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
    signal = audio_statistics(samples)
    silent = audio_statistics(np.zeros(16000, dtype=np.float32))
    assert signal["audio_rms"] > 0.7
    assert signal["silence_window_ratio"] == 0
    assert silent["silence_window_ratio"] == 1
    assert signal["audio_zero_crossing"] > silent["audio_zero_crossing"]


def test_shot_statistics_exclude_outside_and_duplicate_boundaries():
    result = shot_statistics([-2, 0, 1, 1, 3, 5, 8], 5)
    assert result["shot_count"] == 3
    assert result["cuts_per_second"] == 0.4
    assert result["first_shot_s"] == 1
    assert result["shot_mean_s"] == pytest.approx(5 / 3)


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or importlib.util.find_spec("scenedetect") is None,
    reason="FFmpeg and semantic extra required",
)
def test_editing_worker_decodes_all_frames_and_detects_a_real_cut(tmp_path):
    video = tmp_path / "two-shots.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=320x180:r=30:d=2",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x180:r=30:d=2",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
        capture_output=True,
    )
    result = subprocess.run(
        [sys.executable, "-m", "galaxies.video_structure", str(video)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    data = json.loads(result.stdout)
    assert data["decoded_frames"] == 120
    assert data["features"]["shot_count"] == 2
    assert data["cut_times"][0] == pytest.approx(2, abs=0.05)
