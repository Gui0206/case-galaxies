"""Maturity accounting only; no API requests or model training."""

import importlib.util
import sqlite3
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "channel_pilot", Path(__file__).parents[1] / "scripts/channel_pilot.py"
)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_maturity_is_not_48_hours_or_late_observation():
    db = sqlite3.connect(":memory:")
    db.executescript(
        "CREATE TABLE videos(id,channel_id,published_at,is_live); CREATE TABLE observations(video_id,observed_at,views);"
    )
    for vid, age, live in [
        ("early", 48 * 3600, 0),
        ("valid", 168 * 3600, 0),
        ("late", 172 * 3600, 0),
        ("live", 168 * 3600, 1),
    ]:
        db.execute("INSERT INTO videos VALUES (?,?,?,?)", (vid, "channel", 0, live))
        db.execute("INSERT INTO observations VALUES (?,?,?)", (vid, age, 100))
    db.execute("INSERT INTO observations VALUES (?,?,?)", ("valid", 169 * 3600, 150))
    result = pilot.summarize(db, 173 * 3600, {"ends_at": 200 * 3600})
    assert result["mature_7d"] == 1
    assert result["mature_by_channel"] == {"channel": 1}
    assert result["missed_7d_windows"] == 2
    assert not result["completed"]
