"""Persistent local jobs and immutable prospective predictions, backed by SQLite."""

import json
import os
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from galaxies.config import Settings
from galaxies.data import load_records
from galaxies.io import sha256_file, utcnow
from galaxies.model import train
from galaxies.schemas import TrainConfig


class Store:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.path = settings.workspace / "state.sqlite3"
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, status TEXT NOT NULL, payload TEXT NOT NULL,
                    result TEXT, error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS prospective (
                    id TEXT PRIMARY KEY, prediction TEXT NOT NULL, request TEXT NOT NULL,
                    registered_at TEXT NOT NULL, outcome TEXT
                );
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(jobs)")}
            if "owner_pid" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN owner_pid INTEGER")

    def connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def job(self, job_id: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise ValueError("Execução não encontrada.")
        result = dict(row)
        for key in ("payload", "result"):
            result[key] = json.loads(result[key]) if result[key] else None
        return result

    def list_jobs(self) -> list[dict]:
        with self.connect() as db:
            ids = db.execute("SELECT id FROM jobs ORDER BY created_at DESC LIMIT 30").fetchall()
        return [self.job(row["id"]) for row in ids]


class Jobs:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = Store(settings)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="galaxies-train")
        self.lock = threading.Lock()
        self.futures = []

    def recover(self):
        """Mark jobs abandoned by dead processes; preserve live HTTP/MCP peers."""
        with self.store.connect() as db:
            rows = db.execute(
                "SELECT id, owner_pid FROM jobs WHERE status IN ('queued','running')"
            ).fetchall()
            for row in rows:
                try:
                    if row["owner_pid"]:
                        os.kill(row["owner_pid"], 0)
                        continue
                except ProcessLookupError:
                    pass
                except PermissionError:
                    continue
                db.execute(
                    "UPDATE jobs SET status='interrupted', error=?, updated_at=? WHERE id=?",
                    (
                        "Processo encerrado. Use retry para reiniciar com o mesmo contrato.",
                        utcnow().isoformat(),
                        row["id"],
                    ),
                )

    def submit(self, dataset: str, config: TrainConfig) -> dict:
        self.settings.contained(dataset)
        payload = {"dataset": dataset, "config": config.model_dump()}
        with self.lock:
            with self.store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                busy = db.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')"
                ).fetchone()[0]
                if busy >= 4:
                    raise ValueError("Fila de treino cheia; aguarde uma execução terminar.")
                job_id, now = uuid.uuid4().hex, utcnow().isoformat()
                db.execute(
                    "INSERT INTO jobs (id, status, payload, created_at, updated_at, owner_pid) VALUES (?, 'queued', ?, ?, ?, ?)",
                    (job_id, json.dumps(payload), now, now, os.getpid()),
                )
            self.futures.append(self.executor.submit(self._run, job_id, payload))
        return self.store.job(job_id)

    def _run(self, job_id: str, payload: dict):
        while True:
            self.recover()
            with self.store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                active = db.execute("SELECT COUNT(*) FROM jobs WHERE status='running'").fetchone()[
                    0
                ]
                if not active:
                    db.execute(
                        "UPDATE jobs SET status='running', updated_at=? WHERE id=?",
                        (utcnow().isoformat(), job_id),
                    )
                    break
            time.sleep(0.5)
        try:
            path = self.settings.contained(payload["dataset"])
            report = train(
                load_records(path),
                self.settings.workspace / "models",
                TrainConfig.model_validate(payload["config"]),
                sha256_file(path),
            )
            with self.store.connect() as db:
                db.execute(
                    "UPDATE jobs SET status='completed', result=?, updated_at=? WHERE id=?",
                    (json.dumps(report), utcnow().isoformat(), job_id),
                )
        except Exception as exc:
            with self.store.connect() as db:
                db.execute(
                    "UPDATE jobs SET status='failed', error=?, updated_at=? WHERE id=?",
                    (f"{type(exc).__name__}: {str(exc)[:1000]}", utcnow().isoformat(), job_id),
                )

    def retry(self, job_id: str):
        job = self.store.job(job_id)
        if job["status"] not in ("failed", "interrupted"):
            raise ValueError("Somente execuções falhas ou interrompidas podem ser reiniciadas.")
        return self.submit(
            job["payload"]["dataset"], TrainConfig.model_validate(job["payload"]["config"])
        )

    def close(self):
        self.executor.shutdown(wait=True)
