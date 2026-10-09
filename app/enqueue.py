"""주문서 넣기
  python -m app.enqueue          channels.yaml의 채널마다 list_channel
  python -m app.enqueue media    아직 안 받은 영상마다 download_media
"""
import sys

import psycopg

from app.channels import load_rules
from app.config import settings
from app.jobs import add_job


def main():
    with psycopg.connect(settings.database_url) as conn:
        if sys.argv[1:] == ["media"]:
            ids = [r[0] for r in conn.execute("SELECT video_id FROM videos WHERE media_path IS NULL ORDER BY video_id")]
            added = sum(add_job(conn, "download_media", v) for v in ids)
            print(f"download_media 주문서 {added}건 (대상 {len(ids)}편)")
            return
        for url in load_rules():
            added = add_job(conn, "list_channel", url)
            print(f"{'주문서 추가' if added else '이미 대기 중'}: list_channel {url}")


if __name__ == "__main__":
    main()
