"""Select a small context/content ensemble on development only, including ranking.

The ensemble is motivated by complementary validation errors. The whole grid,
including the no-content option, is retained, and 2024 stays inaccessible here.
"""

from pathlib import Path

import pandas as pd

from galaxies.io import read_json, sha256_file, utcnow, write_json
from galaxies.reach_study import macro_error
from galaxies.smtpd_study import CONTEXT_RULES, HISTORY
from galaxies.transfer_study import ranking_scores


def select(run: Path):
    if (run / "ensemble-selection.json").exists():
        raise ValueError("Keep the previous selection; do not overwrite")
    selection = read_json(run / "base-selection.json")
    predictions = pd.read_csv(run / "base-development.csv")
    names = list(selection["errors"])
    context = (
        HISTORY
        + CONTEXT_RULES
        + [n for n in names if n.startswith(("context_", "within_context_"))]
    )
    content = [n for n in names if n not in context]
    context_name, content_name = selection["context"], selection["content"]
    combinations = {}
    for weight in [0, 0.25, 0.5, 0.75, 1]:
        name = f"ensemble_{weight:g}"
        predictions[name] = (1 - weight) * predictions[context_name] + weight * predictions[
            content_name
        ]
        combinations[name] = {
            "context": context_name,
            "content": content_name,
            "content_weight": weight,
        }
    errors = {
        n: macro_error(predictions, predictions[n].to_numpy()) for n in content + list(combinations)
    }
    error_choice = min(errors, key=lambda n: (errors[n], n))
    ranks = ranking_scores(
        predictions, {n: predictions[n].to_numpy() for n in names + list(combinations)}
    )
    result = {
        "created_at": utcnow(),
        "base_plan_sha256": sha256_file(run / "base-plan.json"),
        "base_selection_sha256": sha256_file(run / "base-selection.json"),
        "base_predictions_sha256": sha256_file(run / "base-development.csv"),
        "history": selection["history"],
        "context": context_name,
        "content": error_choice,
        "ranking_history": max(HISTORY, key=lambda n: (ranks[n], n)),
        "ranking_context": max(context, key=lambda n: (ranks[n], n)),
        "ranking_content": max(content + list(combinations), key=lambda n: (ranks[n], n)),
        "combinations": combinations,
        "content_errors": errors,
        "ranking_scores": ranks,
    }
    predictions.to_csv(run / "ensemble-development.csv", index=False)
    result["predictions_sha256"] = sha256_file(run / "ensemble-development.csv")
    write_json(run / "ensemble-selection.json", result)
    return {
        k: result[k]
        for k in [
            "history",
            "context",
            "content",
            "ranking_history",
            "ranking_context",
            "ranking_content",
        ]
    }


if __name__ == "__main__":
    print(select(Path(".runtime/reports/smtpd-v2")))
