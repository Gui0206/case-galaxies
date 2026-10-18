"""Hash existing inference artifacts. Does not train, collect, or open evaluation."""

from pathlib import Path

from galaxies.io import sha256_file, write_json


def prepare(workspace=Path(".runtime")):
    models = workspace / "reports/smtpd-final/models"
    files = (
        list((models / "base_encoder").glob("*"))
        + [
            models / name
            for name in [
                "base/within_semantic_100.npz",
                "base/context_d5_logmean_20.cbm",
                "base/retrieval_d3_median_20.cbm",
                "auxiliary/auxiliary_models.npz",
                "media/editing-tree3/model.cbm",
                "media/editing-tree3/media_encoder.npz",
                "media/editing-tree3/media_encoder.json",
            ]
        ]
        + [
            workspace / name
            for name in [
                "public/smtpd/cohort.parquet",
                "public/smtpd/semantic.npy",
                "public/smtpd/semantic.json",
                "public/smtpd/features/index.json",
                "public/smtpd/features/extractor.json",
                "reports/smtpd-final/selection.json",
                "reports/smtpd-final/calibration.json",
                "reports/smtpd-final/final.json",
            ]
        ]
    )
    manifest = {
        "model_id": "smtpd-frozen-2024",
        "purpose": "inference only; no new evaluation",
        "files": {str(p.relative_to(workspace)): sha256_file(p) for p in sorted(files)},
    }
    write_json(workspace / "real-service-manifest.json", manifest)
    write_json(Path("docs/results/smtpd/inference-manifest.json"), manifest)
    print(f"{len(files)} existing artifacts registered; no models fitted.")


if __name__ == "__main__":
    prepare()
