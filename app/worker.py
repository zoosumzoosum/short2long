"""수집 worker: jobs 표에서 주문서를 하나씩 꺼내 처리한다 (설계 결정 ⑧)."""
import random
import subprocess
import time
from pathlib import Path

import psycopg

from app import youtube
from app.channels import load_rules, sample_shorts, wanted
from app.config import settings
from app.jobs import add_job

MAX_ATTEMPTS = 3  # 이만큼 실패하면 failed로 두고 더 시도하지 않음
IDLE_SECONDS = 5  # 할 일이 없을 때 쉬는 시간

# 찜 + 상태 변경을 한 문장으로: 다른 worker가 찜한 주문서는 건너뜀 (7강)
CLAIM_SQL = """
UPDATE jobs SET status = 'running', attempts = attempts + 1, updated_at = now()
WHERE id = (
    SELECT id FROM jobs
    WHERE status = 'pending'
    ORDER BY id
    LIMIT 1
    FOR UPDATE SKIP LOCKED
)
RETURNING id, kind, target, attempts
"""


def latest(playlist_id, rule, limit):
    """재생목록에서 규칙을 통과한 영상 ID를 최신순으로 limit개"""
    picked = []
    for video_id, title in youtube.playlist_videos(playlist_id):
        if len(picked) >= limit:
            break  # 필요한 만큼 모이면 다음 장은 요청하지 않음 (할당량 절약)
        if wanted(title, rule):
            picked.append(video_id)
    return picked


def list_channel(conn, channel_url):
    """채널에서 색인할 영상을 골라(결정 ⑪) 선정 이유를 남기고, 영상마다 fetch_video 주문서를 넣는다."""
    rule = load_rules()[channel_url]
    uploads = youtube.uploads_playlist(channel_url)
    picked = []  # (영상 ID, 형식, 층)
    if rule["official"]:
        long_ids = latest(youtube.split_playlist(uploads, "long"), rule, rule["long_limit"])
        picked += [(v, "long", None) for v in long_ids]
        if rule["shorts"]:
            # 층을 나누려면 쇼츠 전체 목록이 필요 (CUTIE STREET 약 70장 = 70점)
            shorts = [
                (v, t)
                for v, t in youtube.playlist_videos(youtube.split_playlist(uploads, "short"))
                if wanted(t, rule)
            ]
            picked += [(v, "short", s) for v, s in sample_shorts(shorts, rule["shorts"])]
    else:
        picked += [(v, "fan", None) for v in latest(uploads, rule, rule["limit"])]

    # 목록을 다 받은 뒤 한꺼번에 넣음: 중간에 실패하면 하나도 안 들어가서 재시도해도 중복이 없음
    with conn.transaction():
        for video_id, fmt, stratum in picked:
            conn.execute(
                """
                INSERT INTO selected_videos (video_id, channel_url, format, stratum)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (video_id) DO UPDATE SET format = EXCLUDED.format, stratum = EXCLUDED.stratum
                """,
                (video_id, channel_url, fmt, stratum),
            )
        added = sum(add_job(conn, "fetch_video", video_id) for video_id, _, _ in picked)
    print(f"[list_channel] {channel_url} → 영상 {len(picked)}편 중 {added}편 주문서")


UPSERT_VIDEO = """
INSERT INTO videos (video_id, channel_id, title, description, published_at, duration_sec)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (video_id) DO UPDATE SET
    title = EXCLUDED.title, description = EXCLUDED.description,
    duration_sec = EXCLUDED.duration_sec, collected_at = now()
"""


def fetch_video(conn, video_id):
    """영상 1편의 메타데이터를 videos 표에 넣는다. 이미 있으면 최신 값으로 갱신. (자막은 다음 단계)"""
    v = youtube.video_details(video_id)
    s = v["snippet"]
    conn.execute(
        UPSERT_VIDEO,
        (
            video_id,
            s["channelId"],
            s["title"],
            s.get("description"),
            s["publishedAt"],
            youtube.duration_seconds(v["contentDetails"]["duration"]),
        ),
    )
    add_job(conn, "download_media", video_id)  # 메타데이터가 들어가면 다음 단계 주문서
    print(f"[fetch_video] {video_id} {s['title'][:30]}")


MEDIA_DIR = Path("data/media")  # 호스트 ~/short2long/data/media (.gitignore). 특징을 뽑은 뒤 지운다 (결정 ⑫)


def download_media(conn, video_id):
    """360p 영상+소리를 mp4 하나로 받는다 (yt-dlp). 유튜브 차단을 피하려고 받은 뒤 쉬었다 감"""
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    out = MEDIA_DIR / f"{video_id}.mp4"
    if not out.exists():
        r = subprocess.run(
            [
                "yt-dlp", "--no-progress", "--no-playlist",
                "-f", "bv*[height<=360]+ba/b[height<=360]/b",  # 360p 이하 (화면 특징엔 충분)
                "--merge-output-format", "mp4",
                "-o", str(out),
                f"https://www.youtube.com/watch?v={video_id}",
            ],
            capture_output=True, text=True, timeout=1800,
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip().splitlines()[-1][:300])  # yt-dlp의 마지막 에러 줄
        time.sleep(random.uniform(5, 10))
    conn.execute(
        "UPDATE videos SET media_path = %s, media_bytes = %s WHERE video_id = %s",
        (str(out), out.stat().st_size, video_id),
    )
    print(f"[download_media] {video_id} {out.stat().st_size / 1e6:.1f}MB")


HANDLERS = {"list_channel": list_channel, "fetch_video": fetch_video, "download_media": download_media}


def run_one(conn):
    """주문서 하나를 처리한다. 할 일이 없으면 False."""
    job = conn.execute(CLAIM_SQL).fetchone()
    if job is None:
        return False
    job_id, kind, target, attempts = job
    try:
        HANDLERS[kind](conn, target)
    except Exception as e:
        # 시도 횟수가 남았으면 다시 대기열로, 아니면 실패로 확정
        status = "pending" if attempts < MAX_ATTEMPTS else "failed"
        error = f"{type(e).__name__}: {e}"
        print(f"[job {job_id}] 실패 {attempts}회 → {status} ({error})")
        conn.execute(
            "UPDATE jobs SET status = %s, error = %s, updated_at = now() WHERE id = %s",
            (status, error, job_id),
        )
    else:
        conn.execute(
            "UPDATE jobs SET status = 'done', error = NULL, updated_at = now() WHERE id = %s",
            (job_id,),
        )
    return True


def main():
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        print("worker 시작")
        while True:
            if not run_one(conn):
                time.sleep(IDLE_SECONDS)


if __name__ == "__main__":
    main()
