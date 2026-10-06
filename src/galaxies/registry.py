"""Versioned local model registry. Promotion is explicit and reversible."""

import re
from pathlib import Path

from galaxies.io import read_json, utcnow, write_json
from galaxies.model import ModelBundle


class Registry:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def folder(self, model_id: str) -> Path:
        if not re.fullmatch(r"model-[0-9]{14}-[a-f0-9]{8}", model_id):
            raise ValueError("Identificador de modelo inválido.")
        path = self.root / model_id
        if not (path / "manifest.json").is_file():
            raise ValueError("Modelo não encontrado ou treinamento incompleto.")
        return path

    def list(self) -> list[dict]:
        return sorted(
            [
                read_json(path / "report.json")
                for path in self.root.glob("model-*")
                if (path / "manifest.json").is_file()
            ],
            key=lambda r: r["created_at"],
            reverse=True,
        )

    def active(self) -> str | None:
        path = self.root / "active.json"
        return read_json(path)["model_id"] if path.exists() else None

    def promote(self, model_id: str, *, demo: bool = False) -> dict:
        bundle = ModelBundle(self.folder(model_id))
        report = bundle.report
        if report["evidence_level"] == "synthetic_demo":
            if not demo:
                raise ValueError("Modelo sintético só pode ser ativado com --demo explícito.")
        elif not report["eligible_for_decision"]:
            raise ValueError("Modelo não cumpriu os critérios mínimos; consulte o relatório.")
        previous = self.active()
        event = {
            "model_id": model_id,
            "previous_model_id": previous,
            "promoted_at": utcnow().isoformat(),
            "demo": demo,
        }
        write_json(self.root / "active.json", event)
        write_json(self.root / f"promotion-{utcnow():%Y%m%d%H%M%S%f}.json", event)
        return event

    def load(self, model_id: str | None = None) -> ModelBundle:
        selected = model_id or self.active()
        if not selected:
            raise ValueError(
                "Nenhum modelo ativo. Execute galaxies demo ou treine e promova um modelo."
            )
        return ModelBundle(self.folder(selected))
