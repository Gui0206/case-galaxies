"""Register predictions before publication; append outcomes after the target horizon."""

import json
import uuid
from datetime import datetime, timedelta

import numpy as np

from galaxies.io import utcnow
from galaxies.jobs import Store
from galaxies.schemas import PredictRequest
from galaxies.service import PredictionService


def register(service: PredictionService, request: PredictRequest) -> dict:
    now = utcnow()
    if request.context.planned_datetime <= now:
        raise ValueError("Registro prospectivo exige publicação futura.")
    if abs((now - request.context.as_of).total_seconds()) > 300:
        raise ValueError("as_of deve corresponder ao momento atual (tolerância de cinco minutos).")
    prediction = service.predict(request)
    if prediction.evidence_level == "synthetic_demo" or request.creative.attributes_source == "simulation":
        raise ValueError("Registro prospectivo real não aceita modelo sintético.")
    record_id = uuid.uuid4().hex
    store = Store(service.settings)
    with store.connect() as db:
        db.execute(
            "INSERT INTO prospective VALUES (?, ?, ?, ?, NULL)",
            (record_id, prediction.model_dump_json(), request.model_dump_json(), now.isoformat()),
        )
    return {
        "id": record_id,
        "registered_at": now.isoformat(),
        "prediction": prediction.model_dump(mode="json"),
    }


def record_outcome(
    store: Store, record_id: str, target: int, observed_at: datetime, published_at: datetime
) -> dict:
    if target < 0:
        raise ValueError("Métrica observada não pode ser negativa.")
    if observed_at.tzinfo is None or published_at.tzinfo is None:
        raise ValueError("Datas precisam incluir fuso horário.")
    with store.connect() as db:
        row = db.execute("SELECT * FROM prospective WHERE id=?", (record_id,)).fetchone()
        if row is None:
            raise ValueError("Registro prospectivo não encontrado.")
        if row["outcome"]:
            raise ValueError("Resultado já registrado; o registro é imutável.")
        request, prediction = json.loads(row["request"]), json.loads(row["prediction"])
        registered = datetime.fromisoformat(row["registered_at"])
        if published_at <= registered:
            raise ValueError("A publicação precisa ocorrer depois do registro da previsão.")
        maturity = published_at + timedelta(days=request["context"]["horizon_days"])
        if not maturity <= observed_at <= maturity + timedelta(hours=24) or observed_at > utcnow():
            raise ValueError("Resultado fora do horizonte ou ainda no futuro.")
        planned = datetime.fromisoformat(request["context"]["planned_datetime"])
        outcome = {
            "target": target,
            "observed_at": observed_at.isoformat(),
            "published_at": published_at.isoformat(),
            "schedule_deviation_seconds": (published_at - planned).total_seconds(),
            "mae_log": float(abs(np.log1p(target) - np.log1p(prediction["predicted_metric"]))),
            "covered": prediction["interval_low"] <= target <= prediction["interval_high"],
        }
        cursor = db.execute(
            "UPDATE prospective SET outcome=? WHERE id=? AND outcome IS NULL",
            (json.dumps(outcome), record_id),
        )
        if cursor.rowcount != 1:
            raise ValueError("Resultado já registrado por outra execução.")
    return outcome


def report(store: Store) -> dict:
    with store.connect() as db:
        rows = db.execute(
            "SELECT id, registered_at, outcome FROM prospective ORDER BY registered_at"
        ).fetchall()
    outcomes = [json.loads(r["outcome"]) for r in rows if r["outcome"]]
    return {
        "registered": len(rows),
        "matured": len(outcomes),
        "mae_log": float(np.mean([o["mae_log"] for o in outcomes])) if outcomes else None,
        "coverage": float(np.mean([o["covered"] for o in outcomes])) if outcomes else None,
        "schedule_changed": sum(abs(o["schedule_deviation_seconds"]) > 3600 for o in outcomes),
        "records": [
            {
                "id": r["id"],
                "registered_at": r["registered_at"],
                "outcome": json.loads(r["outcome"]) if r["outcome"] else None,
            }
            for r in rows
        ],
    }
