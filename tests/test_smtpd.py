import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from galaxies.smtpd import reconstruct_views, recover_cohort


def test_paper_score_is_reversible_and_corruption_is_rejected():
    counts = np.array([0, 1, 7, 9876543])
    score = np.log2(counts / 7 + 1)
    np.testing.assert_array_equal(reconstruct_views(score), counts)
    for invalid in [np.nan, np.inf, -1, np.log2(0.5 / 7 + 1)]:
        with pytest.raises(ValueError):
            reconstruct_views([invalid])


def test_metadata_parser_reads_only_identity_dates_and_duration():
    spec = importlib.util.spec_from_file_location("audit_smtpd", Path("scripts/audit_smtpd.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    html = """var ytInitialPlayerResponse = {"videoDetails":{"videoId":"abcdefghijk","channelId":"UCaaaaaaaaaaaaaaaaaaaaaa","viewCount":"999999","lengthSeconds":"30"},"microformat":{"playerMicroformatRenderer":{"publishDate":"2024-01-01T00:00:00Z"}},"playabilityStatus":{"status":"OK"}}; malicious();"""
    result = module.parse_metadata(html, "abcdefghijk")
    assert result["published_at"] == "2024-01-01T00:00:00Z"
    assert "viewCount" not in result
    with pytest.raises(ValueError, match="identity"):
        module.parse_metadata(html, "wrong-id")


def test_cohort_requires_recovered_identity_and_never_uses_source_normalization():
    source = pd.DataFrame(
        {
            "video_id": ["abcdefghijk", "bcdefghijkl", "cdefghijklm"],
            "title": ["a", "b", "c"],
            "category": ["Autos  Vehicles"] * 3,
            "Day 7": np.log2(np.array([7, 14, 21]) / 7 + 1),
            "video_duration": [-0.5, -0.4, -0.3],
            "user_subscribers": [100, 200, 300],
        }
    )
    meta = pd.DataFrame(
        {
            "video_id": source.video_id,
            "published_at": ["2024-01-01T00:00:00Z"] * 3,
            "channel_id": ["UCaaaaaaaaaaaaaaaaaaaaaa"] * 3,
            "duration": [30, 40, 50],
            "is_live": [False, True, False],
            "playability": ["OK", "OK", "LOGIN_REQUIRED"],
            "viewCount": [999] * 3,
        }
    )
    cohort, audit = recover_cohort(source, meta)
    assert len(cohort) == 1
    assert cohort.duration.iloc[0] == 30
    assert cohort.target.iloc[0] == 7
    assert cohort.available_at.iloc[0] - cohort.published_at.iloc[0] == 8 * 86400
    assert "user_subscribers" not in cohort and "viewCount" not in cohort
    assert audit["metadata_unusable"] == 2
