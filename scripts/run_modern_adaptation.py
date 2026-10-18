import argparse
import json

from galaxies import modern_adaptation

parser = argparse.ArgumentParser(
    description="Train on recent data; test on untouched countries and later dates."
)
parser.add_argument("stage", choices=["develop", "final"])
args = parser.parse_args()
print(json.dumps(getattr(modern_adaptation, args.stage)(), indent=2, default=str))
