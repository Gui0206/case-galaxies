"""Run each stage explicitly; final is a frozen, single-use evaluation."""

import argparse
import json

from galaxies.reach_study import develop, encode_titles, final, prepare, provision_language

parser = argparse.ArgumentParser()
parser.add_argument("stage", choices=["weights", "prepare", "encode", "develop", "final"])
args = parser.parse_args()
result = {
    "weights": provision_language,
    "prepare": prepare,
    "encode": encode_titles,
    "develop": develop,
    "final": final,
}[args.stage]()
if args.stage == "final":
    result = {
        k: result[k]
        for k in (
            "train_videos",
            "calibration_videos",
            "confirmed_error_gain",
            "confirmed_ranking_gain",
            "sections",
        )
    }
print(json.dumps(result, indent=2, default=str))
