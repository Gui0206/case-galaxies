import pytest

from galaxies.config import Settings
from galaxies.data import save_records
from galaxies.demo import make_demo
from galaxies.model import train
from galaxies.registry import Registry
from galaxies.schemas import TrainConfig


@pytest.fixture(scope="session")
def records():
    return make_demo(channels=9, posts=60)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, records):
    settings = Settings(workspace=tmp_path_factory.mktemp("workspace")).prepare()
    save_records(settings.workspace / "datasets" / "test.jsonl", records)
    report = train(records, settings.workspace / "models", TrainConfig(iterations=45))
    Registry(settings.workspace / "models").promote(report["model_id"], demo=True)
    return settings, report
