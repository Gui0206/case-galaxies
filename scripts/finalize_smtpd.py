"""Freeze first, publish the reviewable selection, then open the reserved test once."""

import argparse
import json

from galaxies.smtpd_final import final, freeze

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("stage", choices=["freeze", "final"])
args = parser.parse_args()
print(json.dumps({"freeze": freeze, "final": final}[args.stage](), indent=2, default=str))
