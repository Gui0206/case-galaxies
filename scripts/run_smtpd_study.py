"""SMTPD study stages; selection never reads reserved 2024 outcomes."""

import argparse
import json
from pathlib import Path

from galaxies.smtpd_study import RUN, develop

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("stage", choices=["develop"])
parser.add_argument("--run", type=Path, default=RUN)
args = parser.parse_args()
print(json.dumps(develop(args.run), indent=2, default=str))
