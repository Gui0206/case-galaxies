"""Run prerequisite-gated research stages while acquisition continues.

This command never opens or measures the 2024 holdout. It only prepares auxiliary
title vectors and runs the previously declared development experiments.
"""

import argparse
import json
import time

from galaxies.io import read_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("stage", choices=["auxiliary", "media"])
args = parser.parse_args()
if args.stage == "auxiliary":
    from galaxies.reach_study import encode_titles
    from galaxies.smtpd_auxiliary import AUX, develop

    while not (AUX / "audit.json").exists():
        time.sleep(10)
    encode_titles(AUX)
    print(json.dumps(develop(), indent=2), flush=True)
else:
    from galaxies.smtpd_media_study import ROOT, develop, development_media_ready

    while True:
        ready, counts = development_media_ready()
        if ready:
            break
        acquisition = read_json(ROOT / "media-acquisition.json")
        if acquisition.get("blocked") or acquisition.get("stopped"):
            raise SystemExit("Media acquisition stopped; inspect recorded cause")
        time.sleep(10)
    print(json.dumps(develop(), indent=2), flush=True)
