"""Official metadata snapshots only. API access does not imply training/media rights."""

import time

import httpx

from galaxies.io import utcnow


class YouTubeCollector:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Defina YOUTUBE_API_KEY.")
        self.api_key = api_key

    def _get(self, resource: str, params: dict) -> dict:
        for attempt in range(4):
            response = httpx.get(
                f"https://www.googleapis.com/youtube/v3/{resource}",
                params={**params, "key": self.api_key},
                timeout=30,
            )
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(min(2**attempt, 8))
                continue
            if response.status_code != 200:
                # Never echo response/request URLs containing API keys.
                raise ValueError(f"YouTube API respondeu HTTP {response.status_code}.")
            return response.json()
        raise ValueError("YouTube API indisponível.")

    def snapshot(self, video_ids: list[str]) -> list[dict]:
        observations = []
        for offset in range(0, len(video_ids), 50):
            result = self._get(
                "videos",
                {
                    "part": "snippet,statistics,contentDetails",
                    "id": ",".join(video_ids[offset : offset + 50]),
                },
            )
            observed = utcnow().isoformat()
            for video in result.get("items", []):
                stats = video.get("statistics", {})
                observations.append(
                    {
                        "creative_id": video["id"],
                        "observed_at": observed,
                        "likes": int(stats["likeCount"]) if "likeCount" in stats else None,
                        "views": int(stats["viewCount"]) if "viewCount" in stats else None,
                        "channel_id": video["snippet"]["channelId"],
                        "published_at": video["snippet"]["publishedAt"],
                        "caption_at_collection": video["snippet"]["title"],
                        "duration": video.get("contentDetails", {}).get("duration"),
                        "source": "youtube-data-api-v3",
                    }
                )
        return observations

    def channel_video_ids(self, channel_id: str, limit: int = 200) -> list[str]:
        result = self._get("channels", {"part": "contentDetails", "id": channel_id})
        if not result.get("items"):
            raise ValueError("Canal não encontrado.")
        playlist = result["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
        ids, token = [], None
        while len(ids) < limit:
            params = {"part": "contentDetails", "playlistId": playlist, "maxResults": 50}
            if token:
                params["pageToken"] = token
            page = self._get("playlistItems", params)
            ids.extend(item["contentDetails"]["videoId"] for item in page.get("items", []))
            token = page.get("nextPageToken")
            if not token:
                break
        return ids[:limit]
