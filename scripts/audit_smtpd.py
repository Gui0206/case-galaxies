"""Recover missing publication metadata from ordinary public YouTube pages.

No authentication, cookies, proxies, or engagement counts are used. A blocked
response is recorded and never bypassed. Targets remain outside this audit.
"""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import httpx
import pandas as pd

from galaxies.io import sha256_file, utcnow, write_json

ROOT = Path(".runtime/public/smtpd")


def parse_metadata(html, video_id):
    match = re.search(r"(?:var )?ytInitialPlayerResponse\s*=\s*", html)
    payload = json.JSONDecoder().raw_decode(html[match.end() :])[0] if match else {}
    details = payload.get("videoDetails", {})
    micro = payload.get("microformat", {}).get("playerMicroformatRenderer", {})
    play = payload.get("playabilityStatus", {})
    identity = details.get("videoId")
    if identity and identity != video_id:
        raise ValueError("Returned video identity differs from requested identity")
    return {
        "video_id": video_id,
        "playability": play.get("status"),
        "reason": play.get("reason"),
        "published_at": micro.get("publishDate"),
        "upload_date": micro.get("uploadDate"),
        "channel_id": details.get("channelId"),
        "duration": details.get("lengthSeconds"),
        "is_live": details.get("isLiveContent"),
    }


def main(delay=2.0, initial_wait=0):
    source = ROOT / "basic_view_pn.csv"
    f = pd.read_csv(source, usecols=["video_id", "user_url", "category"])
    f = f[f.category.isin(["Howto  Style", "Autos  Vehicles"])]
    # Row duplicates are excluded, not resolved by whichever label looks useful.
    f = f[~f.video_id.duplicated(keep=False)]
    valid_url = f.user_url.astype(str).str.startswith(
        ("http://www.youtube.com/", "https://www.youtube.com/")
    )
    f = f[valid_url]
    counts = f.groupby("user_url").video_id.transform("size")
    f = f[counts >= 4].sort_values("video_id").reset_index(drop=True)
    cache = ROOT / "metadata"
    cache.mkdir(exist_ok=True)
    write_json(
        ROOT / "metadata-plan.json",
        {
            "created_at": utcnow(),
            "source_sha256": sha256_file(source),
            "population": "Autos & Vehicles / Howto & Style; valid original channel URL with >=4 source videos; exclude duplicate IDs",
            "videos": len(f),
            "channels_by_original_url": int(f.user_url.nunique()),
            "selection_uses_outcomes": False,
            "warning": "Surviving public uploads only; dates/identity/duration recovered in 2026, historical content may have changed",
        },
    )

    def fetch(video_id):
        file = cache / f"{video_id}.json"
        if file.exists():
            return json.loads(file.read_text())
        time.sleep(delay)
        try:
            response = httpx.get(
                "https://www.youtube.com/watch",
                params={"v": video_id},
                timeout=25,
                follow_redirects=True,
            )
            if response.status_code in (403, 429):
                write_json(
                    ROOT / "access-stop.json",
                    {
                        "at": utcnow(),
                        "status": response.status_code,
                        "retry_after": response.headers.get("Retry-After"),
                        "video_id": video_id,
                    },
                )
                raise RuntimeError(
                    f"Access restriction HTTP {response.status_code}; acquisition stopped"
                )
            response.raise_for_status()
            result = parse_metadata(response.text, video_id)
            result.update(
                {
                    "status": response.status_code,
                    "retrieved_at": utcnow(),
                    "page_sha256": hashlib.sha256(response.content).hexdigest(),
                }
            )
        except (httpx.HTTPError, ValueError, KeyError) as error:
            result = {"video_id": video_id, "error": type(error).__name__, "retrieved_at": utcnow()}
        write_json(file, result)
        return result

    start = time.monotonic()
    results = []
    time.sleep(initial_wait)
    stopped = False
    try:
        for video_id in f.video_id:
            result = fetch(video_id)
            results.append(result)
            if len(results) % 100 == 0:
                print(
                    f"{len(results)}/{len(f)} metadata records; {time.monotonic() - start:.0f}s",
                    flush=True,
                )
    except RuntimeError as error:
        print(str(error), flush=True)
        stopped = True
    if not results:
        return
    output = f.merge(pd.DataFrame(results), on="video_id", validate="one_to_one")
    output.to_parquet(ROOT / "recovered-metadata.parquet", index=False)
    valid = output.published_at.notna() & output.channel_id.notna()
    audit = {
        "stopped_on_access_restriction": stopped,
        "intended_videos": len(f),
        "videos": len(output),
        "recovered": int(valid.sum()),
        "status_counts": output.playability.fillna("missing").value_counts().to_dict(),
        "metadata_sha256": sha256_file(ROOT / "recovered-metadata.parquet"),
    }
    write_json(ROOT / "metadata-audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--initial-wait", type=float, default=0)
    args = parser.parse_args()
    if args.delay < 1 or args.initial_wait < 0:
        parser.error("Delay must be >=1 second and initial wait non-negative")
    main(args.delay, args.initial_wait)
