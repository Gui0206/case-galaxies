"""Model competition, falsification diagnostics and one-use final evaluation."""

import shutil
import uuid
from pathlib import Path

import numpy as np

from galaxies.data import split_at, temporal_split
from galaxies.io import read_json, sha256_file, utcnow, write_json
from galaxies.metrics import paired_channel_bootstrap
from galaxies.model import train
from galaxies.protocol import (
    candidate_catalog,
    dataset_fingerprint,
    implementation_fingerprint,
    make_plan,
    select_candidate,
)
from galaxies.schemas import PostRecord, TrainConfig


def _metrics(report):
    return {
        "model_id": report["model_id"],
        "metrics": report["metrics"],
        "by_sector": report["by_sector"],
        "by_generalization": report["by_generalization"],
        "pair_probability_test": report["pair_probability_test"],
    }


def _predictions(root: Path, model_id: str, records: list[PostRecord]) -> np.ndarray:
    rows = read_json(root / model_id / "test_predictions.json")
    lookup = {r["creative_id"]: r["prediction"] for r in rows}
    if set(lookup) != {r.creative_id for r in records}:
        raise ValueError("Comparação exige exatamente os mesmos IDs de avaliação.")
    return np.log1p([lookup[r.creative_id] for r in records])


def _shuffle_content(rows: list[PostRecord], seed: int) -> list[PostRecord]:
    """Negative control: permute training content within channel, preserve time/context/labels."""
    rng = np.random.default_rng(seed)
    result = list(rows)
    for channel in sorted({r.channel_id for r in rows}):
        ids = [i for i, r in enumerate(rows) if r.channel_id == channel]
        donors = rng.permutation(ids)
        for i, donor in zip(ids, donors, strict=True):
            fields = {
                name: getattr(rows[donor], name)
                for name in ("features", "caption", "transcript", "visual_embedding", "evidence")
            }
            result[i] = rows[i].model_copy(update=fields)
    return result


def benchmark(
    records: list[PostRecord],
    model_root: Path,
    report_root: Path,
    iterations=180,
    folds=3,
    seed=42,
    progress=None,
) -> dict:
    plan, development, splits = make_plan(records, folds)
    configs = candidate_catalog(iterations, seed)
    run_id = f"research-{utcnow():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:8]}"
    folder = report_root / run_id
    folder.mkdir(parents=True, exist_ok=False)
    plan.update(
        {
            "run_id": run_id,
            "created_at": utcnow().isoformat(),
            "evidence_level": "synthetic_demo" if records[0].synthetic else "development_only",
            "candidates": {name: c.model_dump() for name, c in configs.items()},
        }
    )
    write_json(folder / "plan.json", plan)
    rows = []
    for fold, split in enumerate(splits):
        for name, config in configs.items():
            if progress:
                progress(f"Janela {fold + 1}/{folds} · {name}")
            result = train(
                development,
                model_root,
                config,
                plan["dataset_sha256"],
                split_override=split,
                evaluation_role="development",
            )
            _predictions(model_root, result["model_id"], split["parts"]["test"])
            rows.append({"candidate": name, "fold": fold, **_metrics(result)})
            write_json(folder / "progress.json", rows)
    selection = select_candidate(rows, configs)
    selected = configs[selection["selected"]]
    diagnostics = {
        "note": "Diagnostics on development data only; no final-window labels. They do not establish causal or real-world validity.",
        "ablations": [],
        "shuffled_content": [],
        "seed_stability": [],
        "cold_start": None,
    }
    last = splits[-1]
    for name, blocks in (
        ("context", ["context"]),
        ("context_text", ["context", "text"]),
        ("context_text_visual", ["context", "text", "visual"]),
        ("full", ["context", "text", "visual", "rubric"]),
    ):
        if progress:
            progress(f"Ablação do candidato selecionado · {name}")
        config = TrainConfig.model_validate({**selected.model_dump(), "feature_blocks": blocks})
        result = train(
            development, model_root, config, split_override=last, evaluation_role="ablation"
        )
        diagnostics["ablations"].append({"blocks": blocks, **_metrics(result)})
    # These are stability checks, not a 3-permutation significance test.
    for seed_offset in (1, 2, 3):
        if progress:
            progress(f"Controle negativo e estabilidade · seed {seed + seed_offset}")
        config = selected.model_copy(update={"seed": seed + seed_offset})
        result = train(
            development, model_root, config, split_override=last, evaluation_role="seed_diagnostic"
        )
        diagnostics["seed_stability"].append({"seed": config.seed, **_metrics(result)})
        shuffled = {
            **last,
            "parts": {
                **last["parts"],
                "train": _shuffle_content(last["parts"]["train"], config.seed),
            },
        }
        result = train(
            development,
            model_root,
            config,
            split_override=shuffled,
            evaluation_role="negative_control",
        )
        diagnostics["shuffled_content"].append({"seed": config.seed, **_metrics(result)})
    # Stratify withheld channels by sector; IDs only, never targets.
    channels = []
    for sector in sorted({r.sector for r in development}):
        values = sorted({r.channel_id for r in development if r.sector == sector})
        channels.extend(values[::5])
    from datetime import datetime

    heldout = split_at(development, [datetime.fromisoformat(c) for c in last["cutoffs"]], channels)
    config = selected.model_copy(update={"held_out_channels": channels})
    sizes = {name: len(part) for name, part in heldout["parts"].items()}
    if sizes["train"] < config.min_train_rows or any(
        sizes[name] < 20 for name in ("validation", "calibration", "test")
    ):
        diagnostics["cold_start"] = {
            "status": "insufficient_data",
            "withheld_channels": channels,
            "splits": sizes,
        }
    else:
        result = train(
            development,
            model_root,
            config,
            split_override=heldout,
            evaluation_role="cold_start_diagnostic",
        )
        diagnostics["cold_start"] = {
            "status": "evaluated",
            "withheld_channels": channels,
            **_metrics(result),
        }
    output = {
        "run_id": run_id,
        "plan_sha256": sha256_file(folder / "plan.json"),
        "dataset_sha256": plan["dataset_sha256"],
        "evidence_level": plan["evidence_level"],
        "selection": selection,
        "fold_results": rows,
        "diagnostics": diagnostics,
        "final_test_used": False,
        "completed_at": utcnow().isoformat(),
    }
    write_json(folder / "development.json", output)
    # Freeze the exact candidate choice before a final-evaluation request can read labels.
    write_json(
        folder / "selection.json",
        {
            "plan_sha256": output["plan_sha256"],
            "development_sha256": sha256_file(folder / "development.json"),
            "selected": selection["selected"],
            "strongest_context": selection["strongest_context"],
            "selected_config": selected.model_dump(),
            "context_config": configs[selection["strongest_context"]].model_dump(),
        },
    )
    return output


def final_evaluation(
    records: list[PostRecord], research_folder: Path, model_root: Path, progress=None
) -> dict:
    plan = read_json(research_folder / "plan.json")
    selection = read_json(research_folder / "selection.json")
    development_result = read_json(research_folder / "development.json")
    if plan.get("implementation_sha256") != implementation_fingerprint():
        raise ValueError(
            "Código/dependências de modelagem mudaram após o plano; refaça o desenvolvimento antes de abrir o teste final."
        )
    if (
        sha256_file(research_folder / "plan.json") != selection["plan_sha256"]
        or sha256_file(research_folder / "development.json") != selection["development_sha256"]
    ):
        raise ValueError("Plano ou resultados de seleção foram alterados.")
    if (
        selection["selected"] != development_result["selection"]["selected"]
        or selection["strongest_context"] != development_result["selection"]["strongest_context"]
    ):
        raise ValueError("Escolha diverge do resultado de desenvolvimento congelado.")
    fingerprint = dataset_fingerprint(records)
    if fingerprint != plan["dataset_sha256"]:
        raise ValueError("Dataset diferente do utilizado na seleção; não use o teste final.")
    if (
        selection["selected_config"] != plan["candidates"][selection["selected"]]
        or selection["context_config"] != plan["candidates"][selection["strongest_context"]]
    ):
        raise ValueError("Configuração diverge do catálogo congelado.")
    # This ledger is scoped to this research workspace. It is an audit guard, not security
    # against a user deliberately copying data/reports to another workspace.
    ledger = research_folder.parent / f"final-evaluated-{fingerprint}.json"
    try:
        with ledger.open("x") as handle:
            import json

            json.dump(
                {"run_id": plan["run_id"], "started_at": utcnow().isoformat(), "status": "started"},
                handle,
            )
    except FileExistsError as exc:
        raise ValueError(
            "Teste final desse dataset já foi aberto neste workspace. Consulte o registro existente; não retune usando seus resultados."
        ) from exc
    selected_config = TrainConfig.model_validate(selection["selected_config"])
    context_config = TrainConfig.model_validate(selection["context_config"])
    split = temporal_split(records)
    if split["cutoffs"][-1] != plan["final_start"]:
        raise ValueError("O corte final diverge do plano congelado.")
    if progress:
        progress("Teste final · candidato previamente selecionado")
    candidate = train(
        records,
        model_root,
        selected_config,
        fingerprint,
        split_override=split,
        evaluation_role="final_holdout",
    )
    if progress:
        progress("Teste final · referência forte de contexto")
    context = train(
        records,
        model_root,
        context_config,
        fingerprint,
        split_override=split,
        evaluation_role="final_reference",
    )
    test = split["parts"]["test"]
    content_gain = paired_channel_bootstrap(
        test,
        _predictions(model_root, candidate["model_id"], test),
        _predictions(model_root, context["model_id"], test),
        selected_config.seed,
    )
    sector_ok = all(
        m["rows"] >= 30 and m["interval_coverage"] >= selected_config.coverage - 0.10
        for m in candidate["by_sector"].values()
    )
    checks = {
        "real_data": not records[0].synthetic,
        "baseline_and_global_coverage": candidate["research_gate_passed"],
        "content_in_selected_model": any(b != "context" for b in selected_config.feature_blocks),
        "incremental_pair_gain_vs_context": content_gain["ci95_low"] is not None
        and content_gain["ci95_low"] > 0,
        "macro_log_mae_vs_context": candidate["metrics"]["macro_channel_mae_log"]
        < context["metrics"]["macro_channel_mae_log"],
        "sector_coverage_and_sample": sector_ok,
    }
    # Publish a fresh certified version; never modify already-published bundles in place.
    certified_id = f"model-{utcnow():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:8]}"
    folder = model_root / certified_id
    shutil.copytree(
        model_root / candidate["model_id"],
        folder,
        ignore=shutil.ignore_patterns("manifest.json", "report.json"),
    )
    report = {
        **candidate,
        "model_id": certified_id,
        "created_at": utcnow().isoformat(),
        "parent_model_id": candidate["model_id"],
        "selection_verified": True,
        "eligible_for_decision": all(checks.values()),
        "research_run_id": plan["run_id"],
        "strong_context_model_id": context["model_id"],
        "incremental_content_gain": content_gain,
        "decision_checks": checks,
    }
    write_json(folder / "report.json", report)
    write_json(
        folder / "manifest.json", {p.name: sha256_file(p) for p in folder.iterdir() if p.is_file()}
    )
    outcome = {
        "run_id": plan["run_id"],
        "evidence_level": report["evidence_level"],
        "selected": selection["selected"],
        "model_id": certified_id,
        "context_model_id": context["model_id"],
        "metrics": report["metrics"],
        "context_metrics": context["metrics"],
        "content_gain": content_gain,
        "checks": checks,
        "eligible_for_decision": report["eligible_for_decision"],
        "completed_at": utcnow().isoformat(),
        "note": "Synthetic data validates machinery only. On real data, this is a retrospective research gate, not proof of causal campaign uplift.",
    }
    write_json(research_folder / "final.json", outcome)
    write_json(
        ledger,
        {
            "run_id": plan["run_id"],
            "status": "completed",
            "model_id": certified_id,
            "completed_at": outcome["completed_at"],
        },
    )
    return outcome
