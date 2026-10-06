"""Runtime configuration. No credentials or data live inside the source tree."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GALAXIES_", env_file=".env", extra="ignore")

    workspace: Path = Path(".runtime")
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    api_token: str | None = None
    max_upload_mb: int = Field(default=100, ge=1, le=1000)
    max_video_seconds: int = Field(default=180, ge=1, le=600)
    vlm_url: str = "http://127.0.0.1:11434"
    vlm_model: str | None = None

    def prepare(self) -> "Settings":
        self.workspace = self.workspace.expanduser().resolve()
        for name in ("datasets", "models", "media", "features", "runs", "reports"):
            (self.workspace / name).mkdir(parents=True, exist_ok=True)
        return self

    def contained(self, relative_path: str, area: str = "datasets") -> Path:
        """Resolve only existing regular files inside a managed storage area."""
        root = (self.workspace / area).resolve()
        candidate = (root / relative_path).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise ValueError(f"Arquivo inexistente ou fora de {area}.")
        return candidate
