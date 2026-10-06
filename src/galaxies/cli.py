"""Command-line workflows; `galaxies --help` is the executable command reference."""

import json
import os
import shutil
from pathlib import Path

import typer
from rich.console import Console

from galaxies.config import Settings
from galaxies.data import assemble_observations, audit, load_records, save_records
from galaxies.demo import demo_comparison, make_demo
from galaxies.io import read_json, sha256_file, write_json
from galaxies.model import train as train_model
from galaxies.registry import Registry
from galaxies.schemas import CompareRequest, PredictRequest, TrainConfig

app = typer.Typer(no_args_is_help=True, help="Galaxies · previsão de criativos com evidências.")
console = Console()


def output(value):
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, default=str))


@app.command()
def demo(seed: int = 42, iterations: int = 180):
    """Generate synthetic data, train, activate a DEMO model and compare three creatives."""
    settings = Settings().prepare()
    path = settings.workspace / "datasets" / "demo.jsonl"
    records = make_demo(seed)
    save_records(path, records)
    with console.status("Treinando demonstração sintética…"):
        report = train_model(
            records,
            settings.workspace / "models",
            TrainConfig(seed=seed, iterations=iterations),
            sha256_file(path),
        )
        Registry(settings.workspace / "models").promote(report["model_id"], demo=True)
    from galaxies.service import PredictionService

    comparison = PredictionService(settings).compare(demo_comparison())
    write_json(settings.workspace / "reports" / "demo-comparison.json", comparison)
    console.print(
        "[bold green]Demonstração pronta.[/bold green] Dados sintéticos; não é validação real."
    )
    console.print(f"Modelo: {report['model_id']}")
    console.print("Abra a interface com: [bold]uv run galaxies serve[/bold]")


@app.command()
def validate(dataset: Path):
    """Validate the dataset contract and print a data audit."""
    output(audit(load_records(dataset)))


@app.command("import-data")
def import_data(dataset: Path, name: str | None = None):
    """Validate and copy a local authorized dataset into the managed workspace."""
    settings = Settings().prepare()
    records = load_records(dataset)
    audit(records)
    filename = name or dataset.name
    if Path(filename).name != filename or Path(filename).suffix not in (
        ".jsonl",
        ".json",
        ".parquet",
    ):
        raise typer.BadParameter("Use um nome simples com extensão JSONL, JSON ou Parquet.")
    destination = settings.workspace / "datasets" / filename
    if destination.exists():
        raise typer.BadParameter(
            "Esse nome já existe; escolha outro para preservar a versão anterior."
        )
    save_records(destination, records)
    output({"dataset": filename, "sha256": sha256_file(destination), "audit": audit(records)})


@app.command()
def train(dataset: Path, iterations: int = 240, config: Path | None = None):
    """Train and evaluate an immutable model version; never activate automatically."""
    settings = Settings().prepare()
    options = (
        TrainConfig.model_validate(read_json(config))
        if config
        else TrainConfig(iterations=iterations)
    )
    output(
        train_model(
            load_records(dataset), settings.workspace / "models", options, sha256_file(dataset)
        )
    )


@app.command()
def experiments(dataset: Path, iterations: int = 120):
    """Run ablations, sector models, cold-start and ranking challengers, preserving final holdout."""
    from galaxies.experiments import run_suite

    settings = Settings().prepare()
    output(
        run_suite(
            load_records(dataset),
            settings.workspace / "models",
            settings.workspace / "reports",
            iterations,
        )
    )


@app.command()
def promote(model_id: str, demo: bool = False):
    """Explicitly activate a model that passed the evidence gate, or a labeled demo."""
    output(Registry(Settings().prepare().workspace / "models").promote(model_id, demo=demo))


@app.command()
def models():
    """List immutable versions and the current model."""
    registry = Registry(Settings().prepare().workspace / "models")
    output({"active": registry.active(), "models": registry.list()})


@app.command()
def predict(request: Path):
    """Predict a creative from a PredictRequest JSON file."""
    from galaxies.service import PredictionService

    output(
        PredictionService(Settings())
        .predict(PredictRequest.model_validate(read_json(request)))
        .model_dump(mode="json")
    )


@app.command()
def compare(request: Path):
    """Compare 2–10 creatives from a CompareRequest JSON file."""
    from galaxies.service import PredictionService

    output(PredictionService(Settings()).compare(CompareRequest.model_validate(read_json(request))))


@app.command()
def extract(
    media: Path, audio: bool = False, vision: bool = False, ocr: bool = False, vlm: bool = False
):
    """Extract cached attributes from a local authorized video. Optional models download on first use."""
    from galaxies.media import extract_media

    output(
        extract_media(
            media.resolve(), Settings().prepare(), audio=audio, vision=vision, ocr=ocr, vlm=vlm
        )
    )


@app.command()
def enrich(
    dataset: Path,
    media_root: Path,
    destination: Path,
    audio: bool = False,
    vision: bool = False,
    ocr: bool = False,
    vlm: bool = False,
):
    """Extract original authorized media referenced by media_id and write a new dataset version."""
    from galaxies.media import extract_media

    settings, root = Settings().prepare(), media_root.resolve()
    records = load_records(dataset)
    enriched = []
    for record in records:
        if not record.media_id:
            raise typer.BadParameter(f"{record.creative_id}: falta media_id.")
        path = (root / record.media_id).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise typer.BadParameter("Mídia fora da pasta autorizada ou inexistente.")
        attrs = extract_media(path, settings, audio=audio, vision=vision, ocr=ocr, vlm=vlm)
        value = record.model_dump()
        value["attributes_source"] = "media_extraction"
        value.update(
            {
                key: attrs[key]
                for key in (
                    "features",
                    "transcript",
                    "visual_embedding",
                    "evidence",
                    "content_hash",
                )
            }
        )
        enriched.append(type(record).model_validate(value))
    if destination.exists():
        raise typer.BadParameter("Destino já existe; preserve versões anteriores.")
    save_records(destination, enriched)
    output({"rows": len(enriched), "destination": str(destination)})


@app.command()
def snapshot(video_ids: str, destination: Path):
    """Collect current official YouTube metrics; never invent historical snapshots."""
    from galaxies.youtube import YouTubeCollector

    if destination.exists():
        raise typer.BadParameter("Snapshot já existe; use um novo arquivo.")
    values = YouTubeCollector(os.environ.get("YOUTUBE_API_KEY", "")).snapshot(video_ids.split(","))
    write_json(destination, values)
    output({"observations": len(values), "destination": str(destination)})


@app.command()
def assemble(
    manifest: Path,
    observations: Path,
    destination: Path,
    metric: str = "likes",
    horizon_days: int = 7,
):
    """Join a pre-publication manifest and observed snapshots at a fixed horizon."""
    if destination.exists():
        raise typer.BadParameter("Destino já existe.")
    records, pending = assemble_observations(
        read_json(manifest), read_json(observations), metric, horizon_days
    )
    save_records(destination, records)
    output({"rows": len(records), "pending_or_missed_horizon": pending})


@app.command("register-prospective")
def prospective_register(request: Path):
    """Freeze a real prediction before the planned publication time."""
    from galaxies.prospective import register
    from galaxies.service import PredictionService

    output(
        register(PredictionService(Settings()), PredictRequest.model_validate(read_json(request)))
    )


@app.command("prospective-report")
def prospective_report():
    """Report completed and pending prospective observations."""
    from galaxies.jobs import Store
    from galaxies.prospective import report

    output(report(Store(Settings().prepare())))


@app.command()
def doctor():
    """Check local dependencies, optional extractors and model availability."""
    from galaxies.media import capabilities

    settings = Settings().prepare()
    output(
        {
            "capabilities": capabilities(settings),
            "workspace": str(settings.workspace),
            "active_model": Registry(settings.workspace / "models").active(),
            "ffmpeg_path": shutil.which("ffmpeg"),
        }
    )


@app.command()
def monitor():
    """Summarize local prediction latency, missing attributes and prospective evidence."""
    from galaxies.quality import operational_report

    output(operational_report(Settings().prepare()))


@app.command("audit-extraction")
def audit_extraction(reference: Path, extracted: Path):
    """Compare human annotations with extracted numeric attributes on matched creative IDs."""
    from galaxies.quality import extraction_agreement

    output(extraction_agreement(read_json(reference), read_json(extracted)))


@app.command()
def serve(host: str | None = None, port: int | None = None):
    """Serve the visual workbench and API. Use a single worker for the local job queue."""
    import uvicorn

    from galaxies.api import create_app

    settings = Settings()
    if host:
        settings.host = host
    if port:
        settings.port = port
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="info")


@app.command()
def mcp():
    """Start the official MCP stdio server."""
    from galaxies.mcp_server import main

    main()


if __name__ == "__main__":
    app()
