"""Local HTTP application and visual workbench. All data access stays in the workspace."""

import hmac
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import AwareDatetime, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from galaxies.config import Settings
from galaxies.data import audit, load_records
from galaxies.demo import demo_comparison
from galaxies.jobs import Jobs, Store
from galaxies.media import capabilities, extract_media
from galaxies.prospective import record_outcome, register
from galaxies.prospective import report as prospective_report
from galaxies.real_service import RealCompareRequest, RealCreativeRequest, real_service
from galaxies.schemas import CompareRequest, Contract, PredictRequest, TrainRequest
from galaxies.service import PredictionService


class ExtractionOptions(Contract):
    audio: bool = False
    vision: bool = False
    ocr: bool = False
    vlm: bool = False


class Promotion(Contract):
    demo: bool = False


class Outcome(Contract):
    target: int = Field(ge=0)
    observed_at: AwareDatetime
    published_at: AwareDatetime


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = (settings or Settings()).prepare()
    if settings.host not in ("127.0.0.1", "localhost", "::1") and not settings.api_token:
        raise ValueError("Exposição fora de loopback exige GALAXIES_API_TOKEN.")
    service = PredictionService(settings)
    jobs = Jobs(settings)

    @asynccontextmanager
    async def lifespan(app):
        jobs.recover()
        yield
        jobs.close()

    app = FastAPI(
        title="Galaxies · Creative Intelligence",
        version="0.1.0",
        lifespan=lifespan,
        description="Previsões pré-publicação com evidências. Demonstração sintética é identificada em todas as saídas.",
    )
    app.state.service, app.state.jobs = service, jobs
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver", settings.host],
    )

    @app.middleware("http")
    async def browser_boundary(request: Request, call_next):
        origin = request.headers.get("origin")
        if (
            request.method not in ("GET", "HEAD", "OPTIONS")
            and origin
            and origin != str(request.base_url).rstrip("/")
        ):
            return JSONResponse({"detail": "Origem não permitida."}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; style-src-attr 'unsafe-inline'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'"
        )
        if request.url.path in ("/docs", "/redoc"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; img-src 'self' data: https://fastapi.tiangolo.com; connect-src 'self'; frame-ancestors 'none'"
            )
        return response

    def authorize(request: Request):
        if settings.api_token:
            candidate = request.headers.get("authorization", "").removeprefix("Bearer ")
            if not hmac.compare_digest(candidate, settings.api_token):
                raise HTTPException(status_code=401, detail="Token de acesso necessário.")

    protected = [Depends(authorize)]

    @app.exception_handler(ValueError)
    async def validation_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.post("/api/real/predict", dependencies=protected)
    def predict_real(request: RealCreativeRequest):
        return real_service(settings.workspace).predict(request)

    @app.post("/api/real/compare", dependencies=protected)
    def compare_real(request: RealCompareRequest):
        return real_service(settings.workspace).compare(request)

    @app.get("/api/real/replay/{creative_id}", dependencies=protected)
    def replay_real(creative_id: str):
        return real_service(settings.workspace).replay(creative_id)

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/api/status", dependencies=protected)
    def status():
        selected = service.registry.active()
        return {
            "active_model_id": selected,
            "capabilities": capabilities(settings),
            "model_count": len(service.registry.list()),
            "authenticated": bool(settings.api_token),
        }

    @app.get("/api/models", dependencies=protected)
    def models():
        return {"active_model_id": service.registry.active(), "items": service.registry.list()}

    @app.get("/api/models/{model_id}", dependencies=protected)
    def model_report(model_id: str):
        from galaxies.io import read_json

        return read_json(service.registry.folder(model_id) / "report.json")

    @app.get("/api/models/{model_id}/predictions", dependencies=protected)
    def test_predictions(model_id: str):
        from galaxies.io import read_json

        return {"items": read_json(service.registry.folder(model_id) / "test_predictions.json")}

    @app.get("/api/traces/{trace_id}", dependencies=protected)
    def trace(trace_id: str):
        from galaxies.io import read_json

        if not re.fullmatch(r"[a-f0-9]{32}", trace_id):
            raise ValueError("Identificador de execução inválido.")
        return read_json(settings.contained(f"{trace_id}.json", "runs"))

    @app.post("/api/models/{model_id}/promote", dependencies=protected)
    def promote(model_id: str, body: Promotion):
        return service.registry.promote(model_id, demo=body.demo)

    @app.get("/api/datasets", dependencies=protected)
    def datasets():
        return {
            "items": [
                {"name": p.name, "bytes": p.stat().st_size}
                for p in sorted((settings.workspace / "datasets").iterdir())
                if p.suffix in (".jsonl", ".json", ".parquet")
            ]
        }

    @app.get("/api/datasets/{name}/audit", dependencies=protected)
    def dataset_audit(name: str):
        return audit(load_records(settings.contained(name)))

    @app.post("/api/datasets", dependencies=protected, status_code=201)
    async def upload_dataset(file: UploadFile = File(...)):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in (".jsonl", ".json", ".parquet"):
            raise ValueError("Envie JSONL, JSON ou Parquet no contrato documentado.")
        name = f"dataset-{uuid.uuid4().hex[:12]}{suffix}"
        destination = settings.workspace / "datasets" / name
        size = 0
        try:
            with destination.open("xb") as out:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > settings.max_upload_mb * 1024 * 1024:
                        raise ValueError("Base excede o limite de upload.")
                    out.write(chunk)
            result = audit(load_records(destination))
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        finally:
            await file.close()
        return {"name": name, "audit": result}

    @app.post("/api/train", dependencies=protected, status_code=202)
    def train_job(request: TrainRequest):
        return jobs.submit(request.dataset, request.config)

    @app.get("/api/jobs", dependencies=protected)
    def list_jobs():
        return {"items": jobs.store.list_jobs()}

    @app.get("/api/jobs/{job_id}", dependencies=protected)
    def get_job(job_id: str):
        return jobs.store.job(job_id)

    @app.post("/api/jobs/{job_id}/retry", dependencies=protected, status_code=202)
    def retry(job_id: str):
        return jobs.retry(job_id)

    @app.post("/api/predict", dependencies=protected)
    def predict(request: PredictRequest):
        return service.predict(request)

    @app.post("/api/compare", dependencies=protected)
    def compare(request: CompareRequest):
        return service.compare(request)

    @app.get("/api/demo/scenario", dependencies=protected)
    def scenario():
        return demo_comparison()

    @app.post("/api/media", dependencies=protected, status_code=201)
    async def upload(file: UploadFile = File(...)):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in (".mp4", ".mov", ".webm", ".m4v"):
            raise ValueError("Envie MP4, MOV, WebM ou M4V.")
        media_id = uuid.uuid4().hex + suffix
        destination = settings.workspace / "media" / media_id
        size = 0
        try:
            with destination.open("xb") as out:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > settings.max_upload_mb * 1024 * 1024:
                        raise ValueError("Arquivo excede o limite de upload.")
                    out.write(chunk)
            if size == 0:
                raise ValueError("Arquivo vazio.")
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        finally:
            await file.close()
        return {
            "media_id": media_id,
            "bytes": size,
            "filename": Path(file.filename or "video").name,
        }

    @app.post("/api/media/{media_id}/extract", dependencies=protected)
    def extract(media_id: str, options: ExtractionOptions):
        if not re.fullmatch(r"[a-f0-9]{32}\.(mp4|mov|webm|m4v)", media_id):
            raise ValueError("Identificador de mídia inválido.")
        result = extract_media(
            settings.contained(media_id, "media"), settings, **options.model_dump()
        )
        return {"media_id": media_id, **result}

    @app.post("/api/prospective", dependencies=protected, status_code=201)
    def prospective(request: PredictRequest):
        return register(service, request)

    @app.post("/api/prospective/{record_id}/outcome", dependencies=protected)
    def outcome(record_id: str, body: Outcome):
        return record_outcome(
            Store(settings), record_id, body.target, body.observed_at, body.published_at
        )

    @app.get("/api/prospective", dependencies=protected)
    def prospective_results():
        return prospective_report(Store(settings))

    static = Path(__file__).parent / "web"
    if static.exists():
        app.mount("/assets", StaticFiles(directory=static), name="assets")

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(static / "index.html")

    return app
