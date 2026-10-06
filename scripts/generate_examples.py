"""Regenerate public contracts and synthetic examples without external data."""

import json
from pathlib import Path

from galaxies.demo import demo_comparison, make_demo
from galaxies.schemas import CompareRequest, PostRecord, PredictRequest, TrainRequest

root = Path(__file__).resolve().parents[1]
examples, schemas = root / "examples", root / "docs" / "schemas"
examples.mkdir(exist_ok=True)
schemas.mkdir(parents=True, exist_ok=True)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


compare = demo_comparison()
save(examples / "compare.json", compare.model_dump(mode="json"))
save(examples / "predict.json", PredictRequest(creative=compare.creatives[0], context=compare.context).model_dump(mode="json"))
save(examples / "post-record.json", make_demo(channels=1, posts=1)[0].model_dump(mode="json"))
save(examples / "train-config.json", {"iterations":240,"depth":5,"coverage":.9,"seed":42})
save(examples / "mcp.json", {"mcpServers":{"galaxies":{"command":"uv","args":["--directory","/CAMINHO/ABSOLUTO/case-galaxies","run","galaxies","mcp"]}}})
for name, contract in (("post-record",PostRecord),("predict-request",PredictRequest),("compare-request",CompareRequest),("train-request",TrainRequest)):
    save(schemas / f"{name}.json", contract.model_json_schema())
print("Examples and schemas regenerated.")
