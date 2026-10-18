"""Fast, read-only checkpoint. No network, model loading, fitting or test evaluation."""

import json
from datetime import datetime
from pathlib import Path
from statistics import median


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    root = Path(".runtime/public/smtpd")
    reports = Path(".runtime/reports")
    plan = read(root / "short-media-plan.json")
    acquisition = read(root / "media-acquisition.json")
    features = read(root / "features/index.json")
    final = reports / "smtpd-final"
    count = len(acquisition.get("rows", {}))
    failed = len(acquisition.get("failures", {}))
    remaining = max(0, plan.get("count", 0) - count - failed)
    times = sorted(
        datetime.fromisoformat(r["acquired_at"]) for r in acquisition.get("rows", {}).values()
    )[-21:]
    intervals = [
        (b - a).total_seconds()
        for a, b in zip(times, times[1:], strict=False)
        if (b - a).total_seconds() < 120
    ]
    blocked = acquisition.get("blocked") or acquisition.get("stopped")
    ready = acquisition.get("complete", False) and features.get("extraction_complete", False)
    development = (reports / "smtpd-media/selection.json").exists()
    opened = (final / "evaluation-opened.json").exists()
    complete = (final / "final.json").exists()
    verified = (final / "verification.json").exists()
    next_step = (
        "Review and publish the verified report; do not rerun final evaluation."
        if verified
        else "Recompute metrics with verify_smtpd_results.py."
        if complete
        else "Inspect the existing evaluation log; never delete its opening record."
        if opened
        else "Review recorded access failure before resuming; never bypass provider restrictions."
        if blocked
        else "Freeze selection, commit/push it, then evaluate exactly once."
        if ready and development
        else "Finish the existing acquisition/extraction and development jobs; do not start duplicates."
    )
    print(
        json.dumps(
            {
                "planned": plan.get("count"),
                "downloaded": count,
                "acquisition_failures": failed,
                "remaining_downloads": remaining,
                "extracted": len(features.get("rows", {})),
                "extraction_failures": len(features.get("failures", {})),
                "blocked": blocked,
                "estimated_download_minutes_at_recent_rate": round(
                    remaining * median(intervals) / 60
                )
                if intervals and not blocked
                else None,
                "media_development_complete": development,
                "final_selection_frozen": (final / "selection.json").exists(),
                "test_opened": opened,
                "test_complete": complete,
                "independently_verified": verified,
                "next_step": next_step,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
