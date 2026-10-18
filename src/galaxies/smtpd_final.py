"""Freeze every choice before opening SMTPD's single-use 2024 confirmation.

This module does not tune models. Freeze reads development predictions only;
final fits the frozen choices, exports forecasts, and evaluates the reserved IDs.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from galaxies.io import read_json, sha256_file, utcnow, write_json
from galaxies.reach_study import (
    conformal_radius,
    macro_error,
    paired_gain,
    prediction_metrics,
    ranking,
)
from galaxies.smtpd_auxiliary import AUX, fit_grid
from galaxies.smtpd_media_features import PROTOTYPES
from galaxies.smtpd_media_study import fit_media, load_media, predict_base
from galaxies.smtpd_study import (
    CALIBRATION,
    CONTEXT_RULES,
    HISTORY,
    ROOT,
    TEST_START,
    data,
    matrices,
    partitions,
)
from galaxies.smtpd_study import fingerprint as base_fingerprint
from galaxies.smtpd_study import inputs as base_inputs
from galaxies.transfer_study import ranking_scores

BASE = Path(".runtime/reports/smtpd-v2")
MEDIA = Path(".runtime/reports/smtpd-media")
AUX_RUN = Path(".runtime/reports/smtpd-auxiliary")
RUN = Path(".runtime/reports/smtpd-final")


def code_identity():
    files = [
        "smtpd.py",
        "smtpd_study.py",
        "smtpd_selection.py",
        "smtpd_auxiliary.py",
        "smtpd_media_study.py",
        "smtpd_media_features.py",
        "smtpd_final.py",
        "reach_study.py",
        "transfer_study.py",
        "retention_benchmark.py",
    ]
    return {
        "files": {n: sha256_file(Path(__file__).with_name(n)) for n in files},
        "packages": {
            n: importlib.metadata.version(n)
            for n in ["numpy", "pandas", "scikit-learn", "catboost", "scipy"]
        },
    }


def inputs():
    paths = [
        ROOT / "cohort.parquet",
        ROOT / "semantic.json",
        ROOT / "semantic.npy",
        ROOT / "short-media-plan.json",
        ROOT / "media-acquisition.json",
        ROOT / "features/index.json",
        AUX / "cohort.parquet",
        AUX / "semantic.json",
        AUX / "semantic.npy",
        PROTOTYPES,
    ]
    for run, names in [
        (
            BASE,
            [
                "base-plan.json",
                "base-selection.json",
                "base-development.csv",
                "ensemble-selection.json",
                "ensemble-development.csv",
            ],
        ),
        (MEDIA, ["plan.json", "selection.json", "development.csv"]),
        (AUX_RUN, ["plan.json", "selection.json", "development.csv"]),
    ]:
        paths += [run / n for n in names]
    return {str(p): sha256_file(p) for p in paths}


def freeze(run=RUN):
    if (run / "selection.json").exists():
        raise ValueError("Final selection already frozen")
    frame, _, _ = data(development=True)
    base_plan = read_json(BASE / "base-plan.json")
    if base_plan["code_sha256"] != base_fingerprint() or base_plan["inputs"] != base_inputs():
        raise ValueError("Base development no longer matches its code/data")
    for folder, module in [(AUX_RUN, "smtpd_auxiliary.py"), (MEDIA, "smtpd_media_study.py")]:
        plan = read_json(folder / "plan.json")
        if plan["code_sha256"] != sha256_file(Path(__file__).with_name(module)):
            raise ValueError("Development family code changed after selection")
        if folder == MEDIA and (
            plan["derived_features_code_sha256"]
            != sha256_file(Path(__file__).with_name("smtpd_media_features.py"))
            or plan["visual_prototypes_sha256"] != sha256_file(PROTOTYPES)
        ):
            raise ValueError("Derived media features changed after development")
    _, provenance = load_media(frame)
    base = pd.read_csv(BASE / "ensemble-development.csv").set_index("creative_id")
    auxiliary = pd.read_csv(AUX_RUN / "development.csv").set_index("creative_id")
    media = pd.read_csv(MEDIA / "development.csv").set_index("creative_id")
    validate_development(base, frame)
    if set(base.index) != set(auxiliary.index) or set(base.index) != set(media.index):
        raise ValueError("Candidate families evaluated on different validation videos")
    for part in [auxiliary, media]:
        np.testing.assert_allclose(base.target_log, part.loc[base.index].target_log, atol=1e-12)
    base_selection = read_json(BASE / "ensemble-selection.json")
    result = base[["channel_id", "category", "published_at", "target_log"]].copy()
    specs = {}
    for name in list(read_json(BASE / "base-selection.json")["errors"]) + list(
        base_selection["combinations"]
    ):
        label = f"base/{name}"
        specs[label] = {"family": "base", "name": name}
        result[label] = base[name]
    for name in auxiliary:
        if name.startswith("aux_"):
            label = f"auxiliary/{name}"
            specs[label] = {"family": "auxiliary", "name": name}
            result[label] = auxiliary.loc[result.index, name]
    for name in media:
        if name.startswith(("image:", "speech:", "editing:", "full:")):
            label = f"media/{name}"
            specs[label] = {"family": "media", "name": name}
            result[label] = media.loc[result.index, name]
    history = [f"base/{n}" for n in HISTORY]
    context = [
        f"base/{n}"
        for n in list(read_json(BASE / "base-selection.json")["errors"])
        if n in HISTORY + CONTEXT_RULES or n.startswith(("context_", "within_context_"))
    ] + [n for n in specs if n.startswith("auxiliary/aux_context")]
    content = [n for n in specs if n not in context]
    errors = {n: macro_error(result, result[n].to_numpy()) for n in specs}
    result = result.reset_index()
    ranks = ranking_scores(result, {n: result[n].to_numpy() for n in specs})
    selection = {
        role: min(group, key=lambda n: (errors[n], n))
        for role, group in [("history", history), ("context", context), ("content", content)]
    }
    selection.update(
        {
            "ranking_" + role: max(group, key=lambda n: (ranks[n], n))
            for role, group in [("history", history), ("context", context), ("content", content)]
        }
    )
    run.mkdir(parents=True, exist_ok=True)
    result.to_csv(run / "development.csv", index=False)
    selection.update(
        {
            "created_at": utcnow(),
            "code": code_identity(),
            "inputs": inputs(),
            "specs": specs,
            "errors": errors,
            "ranking_scores": ranks,
            "media_provenance": provenance,
            "primary_test_start": TEST_START,
            "primary_expected_videos": 515,
            "primary_expected_channels": 165,
            "minimum_error_gain_pct": 5,
            "minimum_videos": 500,
            "minimum_channels": 50,
            "minimum_ranking_pairs": 200,
            "minimum_ranking_channels": 30,
            "bootstrap_repetitions": 5000,
            "predictions_sha256": sha256_file(run / "development.csv"),
            "test_opened": False,
        }
    )
    write_json(run / "selection.json", selection)
    return {
        k: selection[k]
        for k in [
            "history",
            "context",
            "content",
            "ranking_history",
            "ranking_context",
            "ranking_content",
        ]
    }


def validate_development(predictions, frame):
    from galaxies.smtpd_study import FOLDS

    eligible = np.concatenate([partitions(frame, start, stop)[2] for start, stop in FOLDS])
    expected = frame.iloc[eligible].set_index("creative_id")
    if not predictions.index.is_unique or set(predictions.index) != set(expected.index):
        raise ValueError("Development predictions include unexpected or duplicate videos")
    np.testing.assert_allclose(
        predictions.target_log, expected.loc[predictions.index].target_log, atol=1e-12
    )
    np.testing.assert_allclose(
        predictions.published_at, expected.loc[predictions.index].published_at, atol=1e-8
    )


def verify(run=RUN):
    selection = read_json(run / "selection.json")
    if (
        selection["code"] != code_identity()
        or selection["inputs"] != inputs()
        or selection["predictions_sha256"] != sha256_file(run / "development.csv")
    ):
        raise ValueError("Frozen experiment inputs/code/selection changed")
    return selection


def final(run=RUN):
    selection = verify(run)
    with (run / "evaluation-opened.json").open("x") as stream:
        json.dump(
            {
                "opened_at": str(utcnow()),
                "selection_sha256": sha256_file(run / "selection.json"),
                "code_identity_sha256": hashlib.sha256(
                    json.dumps(code_identity(), sort_keys=True).encode()
                ).hexdigest(),
            },
            stream,
            indent=2,
        )
    frame, neighbors, semantic = data(development=False)
    all_train, warm, test = partitions(frame, TEST_START, final_train=True)
    _, _, cal = partitions(frame, *CALIBRATION)
    assert len(test) == selection["primary_expected_videos"]
    assert frame.iloc[test].channel_id.nunique() == selection["primary_expected_channels"]
    assert not set(frame.iloc[all_train].creative_id) & set(frame.iloc[cal].creative_id)
    encoder, x = matrices(frame, neighbors, semantic, all_train)
    folder = run / "models"
    encoder.save(folder / "base_encoder")
    positions = np.r_[cal, test]
    base_selection = read_json(BASE / "ensemble-selection.json")
    records, provenance = load_media(frame)
    available = frame.creative_id.isin(records).to_numpy()
    cached = {}
    aux_predictions = None

    def predict(family, name):
        nonlocal aux_predictions
        key = f"{family}/{name}"
        if key in cached:
            return cached[key]
        if family == "base":
            values = predict_base(
                name,
                base_selection,
                x,
                frame,
                all_train,
                warm,
                neighbors,
                positions,
                folder / "base",
            )
        elif family == "auxiliary":
            if aux_predictions is None:
                aux_predictions, _ = fit_grid(
                    frame,
                    all_train,
                    neighbors,
                    positions,
                    x["within_semantic_full"],
                    folder / "auxiliary",
                )
            raw, blend = name.split("@")
            context = predict("base", base_selection["context"])
            values = (1 - float(blend)) * context + float(blend) * aux_predictions[raw]
        elif family == "media":
            raw, blend = name.split("@")
            bundle, model = raw.split(":")
            values = predict("base", base_selection["content"]).copy()
            train = warm[available[warm]]
            ready = positions[available[positions]]
            media_values = fit_media(
                bundle,
                model,
                x["semantic_retrieval"],
                frame,
                train,
                ready,
                records,
                folder / "media" / raw.replace(":", "-"),
            )
            values[available[positions]] = (1 - float(blend)) * values[
                available[positions]
            ] + float(blend) * media_values
        else:
            raise ValueError("Unknown frozen family")
        cached[key] = values
        return values

    roles = {}
    for role in [
        "history",
        "context",
        "content",
        "ranking_history",
        "ranking_context",
        "ranking_content",
    ]:
        spec = selection["specs"][selection[role]]
        roles[role] = predict(spec["family"], spec["name"])
    # Ablations were independently selected in development, never by test outcome.
    media_selection = read_json(MEDIA / "selection.json")
    for bundle, name in media_selection["nonfallback"].items():
        roles[f"ablation_{bundle}"] = predict("media", name)
    roles["text_reference"] = predict("base", base_selection["content"])
    primary = frame.iloc[test].copy()
    export = primary[
        [
            "creative_id",
            "channel_id",
            "category",
            "published_at",
            "target",
            "target_log",
            "history_count",
            "duration",
            "title",
        ]
    ].copy()
    export["media_available"] = available[test]
    metrics, values, radii = {}, {}, {}
    for role, combined in roles.items():
        radius = conformal_radius(np.abs(frame.target_log.to_numpy()[cal] - combined[: len(cal)]))
        values[role] = combined[len(cal) :]
        radii[role] = radius
        export[role] = values[role]
        export[f"{role}_radius"] = radius
        metrics[role] = prediction_metrics(primary, values[role], radius)
    gains = {r: paired_gain(primary, values["content"], values[r]) for r in ["history", "context"]}
    ordering = ranking(
        primary, {r: values[f"ranking_{r}"] for r in ["history", "context", "content"]}
    )
    error_confirmed = (
        all(
            g["relative_reduction_pct"] >= selection["minimum_error_gain_pct"] and g["ci95"][0] > 0
            for g in gains.values()
        )
        and len(test) >= selection["minimum_videos"]
        and primary.channel_id.nunique() >= selection["minimum_channels"]
    )
    ranking_confirmed = (
        ordering["pairs"] >= selection["minimum_ranking_pairs"]
        and ordering["channels"] >= selection["minimum_ranking_channels"]
        and all(ordering[f"content_vs_{r}_ci95"][0] > 0 for r in ["history", "context"])
    )
    short = available[test]
    media_diagnostic = {
        "videos": int(short.sum()),
        "channels": int(primary.loc[short].channel_id.nunique()),
        "exploratory": True,
        "metrics": {
            r: prediction_metrics(primary.loc[short], v[short], radii[r])
            for r, v in values.items()
            if r.startswith("ablation_") or r in ["text_reference", "history", "context"]
        },
        "full_vs_text": paired_gain(
            primary.loc[short], values["ablation_full"][short], values["text_reference"][short]
        ),
    }
    export.to_csv(run / "test_predictions.csv", index=False)
    write_json(
        run / "calibration.json",
        {
            "videos": len(cal),
            "ids": frame.iloc[cal].creative_id.tolist(),
            "radii": radii,
            "coverage_nominal": 0.9,
        },
    )
    report = {
        "created_at": utcnow(),
        "selection_sha256": sha256_file(run / "selection.json"),
        "train_videos": len(warm),
        "all_training_videos": len(all_train),
        "calibration_videos": len(cal),
        "primary_videos": len(test),
        "primary_channels": int(primary.channel_id.nunique()),
        "primary_metrics": metrics,
        "content_gains": gains,
        "ranking": ordering,
        "media_ablation": media_diagnostic,
        "media_provenance": provenance,
        "confirmed_error_gain": bool(error_confirmed),
        "confirmed_ranking_gain": bool(ranking_confirmed),
        "case_solved": False,
        "production_eligible": False,
        "predictions_sha256": sha256_file(run / "test_predictions.csv"),
        "limitations": [
            "Selected surviving public videos; source excludes missing days and extreme popularity",
            "Incomplete same-channel histories; Howto & Style broader than fashion/beauty",
            "Day-seven timing nominal; eight-day conservative maturity, no original observation timestamps",
            "Recovered media may have changed since original publication",
            "Full-media ablation only on accessible <=180-second videos; remaining videos retain text predictions",
            "No fixed-age likes or prospective Galaxies validation; rankings are observational, not causal A/B effects",
        ],
    }
    write_json(run / "final.json", report)
    return {
        k: report[k]
        for k in [
            "primary_videos",
            "primary_channels",
            "content_gains",
            "confirmed_error_gain",
            "confirmed_ranking_gain",
        ]
    }
