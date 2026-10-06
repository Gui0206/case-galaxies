import asyncio
import json
import os
import shutil
import subprocess
import sys
from datetime import timedelta

import pytest

from galaxies.config import Settings
from galaxies.io import utcnow
from galaxies.jobs import Jobs, Store
from galaxies.media import extract_media
from galaxies.prospective import record_outcome, report
from galaxies.schemas import TrainConfig


def test_persistent_training_job_completes(trained):
    settings, _ = trained
    manager = Jobs(settings)
    job = manager.submit("test.jsonl", TrainConfig(iterations=25))
    manager.close()
    finished = manager.store.job(job["id"])
    assert finished["status"] == "completed", finished["error"]
    assert finished["result"]["evidence_level"] == "synthetic_demo"
    assert manager.store.job(job["id"])["result"]["model_id"].startswith("model-")


def test_live_peer_job_is_not_marked_interrupted(tmp_path):
    settings = Settings(workspace=tmp_path).prepare()
    manager = Jobs(settings)
    with manager.store.connect() as db:
        db.execute(
            "INSERT INTO jobs (id,status,payload,created_at,updated_at,owner_pid) VALUES ('peer','running','{}','now','now',?)",
            (os.getpid(),),
        )
    manager.recover()
    assert manager.store.job("peer")["status"] == "running"
    manager.close()


def test_prospective_outcome_is_append_only_and_horizon_aligned(tmp_path):
    store = Store(Settings(workspace=tmp_path).prepare())
    published = utcnow() - timedelta(days=8)
    prediction = {"predicted_metric": 100, "interval_low": 70, "interval_high": 150}
    request = {"context": {"planned_datetime": published.isoformat(), "horizon_days": 7}}
    with store.connect() as db:
        db.execute(
            "INSERT INTO prospective VALUES ('p',?,?,?,NULL)",
            (
                json.dumps(prediction),
                json.dumps(request),
                (published - timedelta(hours=2)).isoformat(),
            ),
        )
    value = record_outcome(store, "p", 110, published + timedelta(days=7), published)
    assert value["covered"] is True
    assert report(store)["matured"] == 1
    with pytest.raises(ValueError, match="imutável"):
        record_outcome(store, "p", 120, published + timedelta(days=7), published)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is an optional system dependency")
def test_real_video_extraction_and_cache(tmp_path):
    settings = Settings(workspace=tmp_path / "workspace").prepare()
    video = tmp_path / "original.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=12",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000",
            "-t",
            "2",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            str(video),
        ],
        check=True,
    )
    result = extract_media(video, settings)
    assert result["features"]["duration_s"] == pytest.approx(2, abs=0.1)
    assert result["features"]["audio_rms"] > 0
    assert len(result["frame_times"]) >= 5
    assert "hook_clarity" not in result["features"]
    assert extract_media(video, settings)["cache_hit"] is True


def test_official_mcp_stdio_round_trip(trained):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def scenario():
        settings, _ = trained
        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "galaxies.mcp_server"],
            env={**os.environ, "GALAXIES_WORKSPACE": str(settings.workspace)},
        )
        async with stdio_client(server) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                inventory = await session.list_tools()
                names = {tool.name for tool in inventory.tools}
                assert {
                    "predict_creative",
                    "compare_creatives",
                    "train_model",
                    "get_training_status",
                } <= names
                result = await session.call_tool("get_model_report", {})
                assert not result.isError
                assert result.structuredContent["evidence_level"] == "synthetic_demo"

    asyncio.run(scenario())
