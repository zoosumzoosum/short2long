"""YouTube Data API v3 호출. 여기서 쓰는 요청은 모두 1회 1점 (하루 한도 10,000점)."""
import re

import httpx

from app.config import settings

API = "https://www.googleapis.com/youtube/v3"


class YouTubeError(Exception):
    pass


def _get(resource, **params):
    params["key"] = settings.youtube_api_key
    r = httpx.get(f"{API}/{resource}", params=params, timeout=10)
    if r.status_code != 200:
        # httpx 기본 에러(raise_for_status)는 URL을 통째로 담아 API 키가 로그·jobs.error에 남음 → 직접 만든 에러로
        try:
            reason = r.json()["error"]["errors"][0]["reason"]  # 예: quotaExceeded, rateLimitExceeded
        except Exception:
            reason = ""
        raise YouTubeError(f"{resource} {r.status_code} {reason}")  # worker가 재시도·실패 기록
    return r.json()


def uploads_playlist(channel_url):
    """채널 URL → 그 채널의 '업로드 재생목록' ID (①)"""
    if "/channel/" in channel_url:
        params = {"id": channel_url.rsplit("/", 1)[-1]}
    else:
        params = {"forHandle": channel_url.rsplit("/@", 1)[-1]}
    items = _get("channels", part="contentDetails", **params).get("items", [])
    if not items:
        raise ValueError(f"채널을 찾지 못함: {channel_url}")
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def split_playlist(uploads_id, kind):
    """업로드 재생목록(UU...)을 종류별로: 'long' → UULF..., 'short' → UUSH...
    공식 문서에는 없는 규칙이라 바뀔 수 있음. 2026-10-08 확인: CUTIE STREET UU 3,983 = LF 513 + SH 3,455 + LV 15"""
    return {"long": "UULF", "short": "UUSH"}[kind] + uploads_id[2:]


def playlist_videos(playlist_id):
    """재생목록의 (영상 ID, 제목)을 최신순으로 하나씩 내준다. 50개를 다 쓰면 다음 장을 요청 (②)"""
    params = {"part": "snippet", "playlistId": playlist_id, "maxResults": 50}
    while True:
        data = _get("playlistItems", **params)
        for item in data["items"]:
            s = item["snippet"]
            yield s["resourceId"]["videoId"], s["title"]
        if "nextPageToken" not in data:
            return
        params["pageToken"] = data["nextPageToken"]


def video_details(video_id):
    """영상 1편의 메타데이터 (snippet: 제목·설명란·채널·업로드 시각, contentDetails: 길이)"""
    items = _get("videos", part="snippet,contentDetails", id=video_id).get("items", [])
    if not items:
        raise ValueError(f"영상을 찾지 못함 (비공개·삭제?): {video_id}")
    return items[0]


def duration_seconds(iso):
    """API의 길이 표기 'PT9M46S' → 586"""
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso)
    d, h, mi, s = (int(x or 0) for x in m.groups())
    return ((d * 24 + h) * 60 + mi) * 60 + s
