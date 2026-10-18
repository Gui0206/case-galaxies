"""Bounded public-metadata pilot: no ML, training, video downloads, or LLM calls."""

import fcntl
import json
import os
import sqlite3
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from galaxies.youtube import YouTubeCollector

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ".runtime/channel-pilot"
DAY = 86400


def epoch(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def summarize(db, now, config):
    mature = db.execute(
        """SELECT v.id, v.channel_id, MIN(o.observed_at) - v.published_at AS age
        FROM videos v JOIN observations o ON v.id=o.video_id
        WHERE o.views IS NOT NULL AND o.observed_at-v.published_at BETWEEN ? AND ?
        AND v.is_live=0 GROUP BY v.id""",
        (7 * DAY, 7 * DAY + 3 * 3600),
    ).fetchall()
    counts = {}
    for _, channel, _ in mature:
        counts[channel] = counts.get(channel, 0) + 1
    tracked = db.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    missed = db.execute(
        """SELECT COUNT(*) FROM videos v WHERE v.is_live=0 AND v.published_at < ?
        AND NOT EXISTS (SELECT 1 FROM observations o WHERE o.video_id=v.id
        AND o.views IS NOT NULL AND o.observed_at-v.published_at BETWEEN ? AND ?)""",
        (now - 7 * DAY - 3 * 3600, 7 * DAY, 7 * DAY + 3 * 3600),
    ).fetchone()[0]
    return {
        "tracked_videos": tracked,
        "mature_7d": len(mature),
        "mature_by_channel": counts,
        "missed_7d_windows": missed,
        "completed": now >= config["ends_at"],
        "horizon": "observed age 168–171 hours; exact timestamps retained, never relabeled as exact 168h",
        "no_training": True,
        "no_media_downloads": True,
    }


def notify(message):
    # Fixed local notification text only; no external messaging or credential content.
    result = subprocess.run(
        [
            "osascript",
            "-e",
            "display notification " + json.dumps(message) + ' with title "Piloto Galaxies"',
        ],
        capture_output=True,
    )
    return result.returncode == 0


def run():
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / "run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        config = json.loads((DATA / "config.json").read_text())
        old = (
            json.loads((DATA / "status.json").read_text())
            if (DATA / "status.json").exists()
            else {}
        )
        if old.get("completed"):
            return
        now = datetime.now(UTC).timestamp()
        db = sqlite3.connect(DATA / "observations.sqlite")
        db.executescript("""CREATE TABLE IF NOT EXISTS videos (
          id TEXT PRIMARY KEY, channel_id TEXT, published_at REAL, first_seen REAL,
          title_at_discovery TEXT, duration TEXT, is_live INTEGER);
          CREATE TABLE IF NOT EXISTS observations (
          video_id TEXT, observed_at REAL, views INTEGER, likes INTEGER, title TEXT,
          availability TEXT, PRIMARY KEY(video_id,observed_at));
          CREATE TABLE IF NOT EXISTS runs (observed_at REAL, success INTEGER, error TEXT);
        """)
        status = dict(old)
        try:
            if now < config["ends_at"]:
                load_dotenv(ROOT / ".env")
                client = YouTubeCollector(os.environ.get("YOUTUBE_API_KEY", ""))
                candidates = set()
                if now <= config["enrollment_ends_at"] + 3 * 3600:
                    for channel in config["channels"]:
                        page = client._get(
                            "playlistItems",
                            {
                                "part": "contentDetails",
                                "playlistId": channel["uploads"],
                                "maxResults": 50,
                            },
                        )
                        items = page.get("items", [])
                        if len(items) == 50 and all(
                            epoch(
                                x["contentDetails"].get("videoPublishedAt", "1970-01-01T00:00:00Z")
                            )
                            >= now - 2 * 3600
                            for x in items
                        ):
                            raise ValueError(
                                "Publicação acima do limite do piloto; revisar cobertura antes de ampliar"
                            )
                        for item in items:
                            detail = item["contentDetails"]
                            published = epoch(
                                detail.get("videoPublishedAt", "1970-01-01T00:00:00Z")
                            )
                            if (
                                config["started_at"]
                                <= published
                                <= min(now, config["enrollment_ends_at"])
                            ):
                                candidates.add(detail["videoId"])
                candidates.update(
                    r[0]
                    for r in db.execute(
                        "SELECT id FROM videos WHERE published_at >= ? AND is_live=0",
                        (now - 7 * DAY - 3 * 3600,),
                    )
                )
                ids = sorted(candidates)
                if len(ids) > 500:
                    raise ValueError(
                        "Limite de 500 vídeos ativos atingido; piloto não será ampliado automaticamente"
                    )
                for start in range(0, len(ids), 50):
                    batch = ids[start : start + 50]
                    result = client._get(
                        "videos",
                        {
                            "part": "snippet,statistics,contentDetails,liveStreamingDetails",
                            "id": ",".join(batch),
                        },
                    )
                    observed = datetime.now(UTC).timestamp()
                    found = set()
                    for item in result.get("items", []):
                        vid = item["id"]
                        found.add(vid)
                        snippet = item["snippet"]
                        stats = item.get("statistics", {})
                        published = epoch(snippet["publishedAt"])
                        if not config["started_at"] <= published <= config["enrollment_ends_at"]:
                            continue
                        live = (
                            bool(item.get("liveStreamingDetails"))
                            or snippet.get("liveBroadcastContent", "none") != "none"
                        )
                        db.execute(
                            "INSERT OR IGNORE INTO videos VALUES (?,?,?,?,?,?,?)",
                            (
                                vid,
                                snippet["channelId"],
                                published,
                                observed,
                                snippet["title"],
                                item.get("contentDetails", {}).get("duration"),
                                int(live),
                            ),
                        )
                        db.execute(
                            "INSERT OR IGNORE INTO observations VALUES (?,?,?,?,?,?)",
                            (
                                vid,
                                observed,
                                int(stats["viewCount"]) if "viewCount" in stats else None,
                                int(stats["likeCount"]) if "likeCount" in stats else None,
                                snippet["title"],
                                "returned",
                            ),
                        )
                    for vid in set(batch) - found:
                        db.execute(
                            "INSERT OR IGNORE INTO observations VALUES (?,?,?,?,?,?)",
                            (vid, observed, None, None, None, "not_returned"),
                        )
                db.commit()
            status.update(summarize(db, now, config))
            status.update(
                last_success_at=datetime.now(UTC).isoformat(), consecutive_failures=0, error=None
            )
            db.execute("INSERT INTO runs VALUES (?,?,?)", (now, 1, None))
        except Exception as exc:
            # Avoid traceback/HTTP URLs containing credentials; errors from our collector are sanitized.
            safe = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
            status.update(error=safe, consecutive_failures=old.get("consecutive_failures", 0) + 1)
            db.execute("INSERT INTO runs VALUES (?,?,?)", (now, 0, safe))
        finally:
            db.commit()
            db.close()
        status["checked_at"] = datetime.now(UTC).isoformat()
        events = status.setdefault("events", [])
        for key, condition, message in [
            (
                "capture_complete",
                now >= config["enrollment_ends_at"],
                f"Coleta inicial encerrada: {status.get('tracked_videos', 0)} vídeos registrados. Relatório local disponível; o horizonte de sete dias ainda precisa amadurecer.",
            ),
            (
                "first_mature",
                status.get("mature_7d", 0) > 0,
                "Primeiros resultados próximos de sete dias registrados. Ainda não demonstram ganho do modelo.",
            ),
            (
                "failure",
                status.get("consecutive_failures", 0) >= 3,
                "A coleta Galaxies falhou três vezes seguidas. Os registros foram preservados; é necessária uma verificação.",
            ),
            (
                "complete",
                status.get("completed", False),
                "Piloto Galaxies encerrado. Dados e lacunas estão registrados; nenhum treino foi realizado.",
            ),
        ]:
            if condition and not any(e["kind"] == key for e in events):
                events.append(
                    {
                        "kind": key,
                        "at": status["checked_at"],
                        "message": message,
                        "local_notification_requested": notify(message),
                    }
                )
        save(DATA / "status.json", status)
        (DATA / "summary.md").write_text(
            "# Piloto Galaxies — status automático\n\n"
            + f"Atualizado em {status['checked_at']} (UTC).\n\n"
            + f"- Vídeos registrados: {status.get('tracked_videos', 0)}\n"
            + f"- Resultados observados entre 168 e 171 horas: {status.get('mature_7d', 0)}\n"
            + f"- Janelas de sete dias perdidas: {status.get('missed_7d_windows', 0)}\n"
            + f"- Falhas consecutivas: {status.get('consecutive_failures', 0)}\n"
            + f"- Encerrado: {status.get('completed', False)}\n\n"
            + "Piloto de coleta, sem treino ou avaliação de ganho. Dados apenas de publicações novas; "
            + "não recupera histórico anterior nem contém vídeos completos. Zero vídeos também é um resultado possível.\n"
        )
        print(
            json.dumps(
                {
                    k: status.get(k)
                    for k in [
                        "checked_at",
                        "tracked_videos",
                        "mature_7d",
                        "missed_7d_windows",
                        "error",
                        "completed",
                    ]
                }
            )
        )


if __name__ == "__main__":
    run()
