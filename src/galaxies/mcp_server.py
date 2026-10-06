"""MCP stdio tool server using the official SDK. Stdout is reserved for protocol traffic."""

from typing import Any

from mcp.server.fastmcp import FastMCP

from galaxies.config import Settings
from galaxies.jobs import Jobs
from galaxies.schemas import CompareRequest, PredictRequest, TrainConfig
from galaxies.service import PredictionService


def create_mcp(settings: Settings | None = None):
    settings = (settings or Settings()).prepare()
    service = PredictionService(settings)
    jobs = Jobs(settings)
    mcp = FastMCP("Galaxies Creative Intelligence")

    @mcp.tool(structured_output=True)
    def predict_creative(request: PredictRequest) -> dict[str, Any]:
        """Estimate a creative before publication; always inspect evidence_level and confidence_reasons."""
        return service.predict(request).model_dump(mode="json")

    @mcp.tool(structured_output=True)
    def compare_creatives(request: CompareRequest) -> dict[str, Any]:
        """Compare 2–10 creatives in the SAME channel/context. Synthetic results cannot recommend campaigns."""
        return service.compare(request)

    @mcp.tool(structured_output=True)
    def train_model(dataset: str, config: TrainConfig | None = None) -> dict[str, Any]:
        """Start training on an existing workspace dataset. Returns a persistent job ID; does not promote."""
        return jobs.submit(dataset, config or TrainConfig())

    @mcp.tool(structured_output=True)
    def get_training_status(job_id: str) -> dict[str, Any]:
        """Read a training job's current state, report or failure."""
        return jobs.store.job(job_id)

    @mcp.tool(structured_output=True)
    def get_model_report(model_id: str | None = None) -> dict[str, Any]:
        """Read model provenance, temporal evaluation, uncertainty and limitations."""
        return service.bundle(model_id).report

    @mcp.tool(structured_output=True)
    def list_models() -> dict[str, Any]:
        """List available models and the currently active model."""
        return {"active_model_id": service.registry.active(), "items": service.registry.list()}

    return mcp, jobs


def main():
    mcp, jobs = create_mcp()
    try:
        mcp.run(transport="stdio")
    finally:
        jobs.close()


if __name__ == "__main__":
    main()
