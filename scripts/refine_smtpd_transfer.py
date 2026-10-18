"""One bounded transfer refinement: preserve domain PCA and weight similar source content.

Go/no-go was fixed before this experiment: improve both the no-transfer semantic
ensemble and the strongest context by at least 2% on development, otherwise do
not add this extra family to the reserved confirmation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from galaxies.io import read_json, sha256_file, utcnow, write_json
from galaxies.reach_study import macro_error
from galaxies.smtpd_auxiliary import AUX, auxiliary_data, sufficient_statistics
from galaxies.smtpd_study import FOLDS, data, matrices, partitions

RUN = Path(".runtime/reports/smtpd-transfer-refinement")
BASE = Path(".runtime/reports/smtpd-v2")


def weighted_source_stats(x, frame, rows, importance):
    chosen = frame.iloc[rows]
    groups = chosen.channel_id.to_numpy()
    frequency = pd.Series(groups).value_counts()
    keep = pd.Series(groups).map(frequency).to_numpy() >= 2
    selected = rows[keep]
    groups = groups[keep]
    values = pd.DataFrame(x[selected])
    centered = values.to_numpy() - values.groupby(groups).transform("mean").to_numpy()
    target = pd.Series(frame.target_log.to_numpy()[selected])
    cy = target.to_numpy() - target.groupby(groups).transform("mean").to_numpy()
    # Topic importance is aggregated per creator, preserving the within-creator contrast.
    group_weight = pd.Series(importance[keep]).groupby(groups).transform("mean").to_numpy()
    counts = pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy()
    weight = group_weight / counts
    weight /= weight.sum()
    return (
        centered.T @ (weight[:, None] * centered),
        centered.T @ (weight * cy),
        float(1 / np.sum(weight**2)),
    )


def main():
    if RUN.exists():
        raise ValueError("Preserve the existing refinement result")
    frame, neighbors, semantic = data(development=True)
    aux, aux_pure = auxiliary_data()
    base = pd.read_csv(BASE / "ensemble-development.csv").set_index("creative_id")
    selection = read_json(BASE / "ensemble-selection.json")
    write_json(
        RUN / "plan.json",
        {
            "created_at": utcnow(),
            "code_sha256": sha256_file(Path(__file__)),
            "auxiliary_sha256": sha256_file(AUX / "cohort.parquet"),
            "fractions": [0, 0.1, 0.3, 0.5],
            "ridge_penalties": [10, 100, 1000],
            "source_weights": ["uniform", "clipped creator-level semantic density ratio"],
            "context_blends": [0.25, 0.5, 0.75, 1.0],
            "go_no_go": "at least 2% lower development error than BOTH existing semantic ensemble and strongest context; no final targets consulted",
        },
    )
    parts = []
    for start, stop in FOLDS:
        train, _, valid = partitions(frame, start, stop)
        encoder, x = matrices(frame, neighbors, semantic, train)
        primary = x["within_semantic"]
        auxiliary = np.column_stack(
            [aux_pure[:, :15], encoder.semantic_pca.transform(aux_pure[:, 15:])]
        )
        aux_train = np.flatnonzero(
            aux.available_at.to_numpy() <= frame.available_at.to_numpy()[train].max()
        )
        scale = StandardScaler().fit(primary[train])
        px, ax = scale.transform(primary), scale.transform(auxiliary)
        pg, ph, pn = sufficient_statistics(px, frame, train, monthly=False)
        classifier = LogisticRegression(
            C=1, max_iter=500, class_weight="balanced", random_state=42
        ).fit(
            np.vstack([px[train], ax[aux_train]]),
            np.r_[np.ones(len(train)), np.zeros(len(aux_train))],
        )
        probability = classifier.predict_proba(ax[aux_train])[:, 1]
        ratio = np.clip(probability / np.maximum(1 - probability, 1e-8), 0.05, 20)
        part = frame.iloc[valid][
            ["creative_id", "channel_id", "category", "published_at", "target_log"]
        ].copy()
        part["baseline"] = base.loc[part.creative_id, selection["content"]].to_numpy()
        context = base.loc[part.creative_id, selection["context"]].to_numpy()
        for domain_weight, importance in [("uniform", np.ones(len(aux_train))), ("similar", ratio)]:
            ag, ah, ess = weighted_source_stats(ax, aux, aux_train, importance)
            for fraction in [0, 0.1, 0.3, 0.5]:
                if fraction == 0 and domain_weight == "similar":
                    continue
                gram = ((1 - fraction) * pg + fraction * ag) * pn
                rhs = ((1 - fraction) * ph + fraction * ah) * pn
                for alpha in [10, 100, 1000]:
                    beta = np.linalg.solve(gram + alpha * np.eye(len(rhs)), rhs)
                    scores = px @ beta
                    prediction = np.array(
                        [
                            frame.target_log.iloc[neighbors[i][-20:]].mean()
                            + scores[i]
                            - scores[neighbors[i][-20:]].mean()
                            for i in valid
                        ]
                    )
                    prediction = np.maximum(0, prediction)
                    for blend in [0.25, 0.5, 0.75, 1.0]:
                        part[f"{domain_weight}:{fraction:g}:{alpha}:{blend:g}"] = (
                            1 - blend
                        ) * context + blend * prediction
            print(
                {"fold": start, "weight": domain_weight, "auxiliary_effective_rows": round(ess)},
                flush=True,
            )
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    names = [c for c in result if ":" in c] + ["baseline"]
    errors = {n: macro_error(result, result[n].to_numpy()) for n in names}
    best = min(errors, key=lambda n: (errors[n], n))
    context_selection = read_json(Path(".runtime/reports/smtpd-auxiliary/selection.json"))
    strongest_context = context_selection["errors"][context_selection["context"]]
    eligible = errors[best] <= 0.98 * min(errors["baseline"], strongest_context)
    result.to_csv(RUN / "development.csv", index=False)
    report = {
        "created_at": utcnow(),
        "selected": best,
        "error": errors[best],
        "old_content_error": errors["baseline"],
        "strongest_context_error": strongest_context,
        "adoption_eligible": bool(eligible),
        "errors": errors,
        "plan_sha256": sha256_file(RUN / "plan.json"),
        "predictions_sha256": sha256_file(RUN / "development.csv"),
    }
    write_json(RUN / "selection.json", report)
    print({k: v for k, v in report.items() if k not in ["errors", "created_at"]}, flush=True)


if __name__ == "__main__":
    main()
