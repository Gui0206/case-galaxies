"""Run the transfer experiment without reusing the closed historical test."""

import argparse
import json

from galaxies import transfer_study

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("stage", choices=["prepare", "encode", "develop", "final"])
args = parser.parse_args()
if args.stage == "encode":
    from galaxies.reach_study import encode_titles

    result = encode_titles(transfer_study.ROOT)
else:
    result = getattr(transfer_study, args.stage)()
print(json.dumps(result, indent=2, default=str))
