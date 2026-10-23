"""핑거프린트 색인 만들기 / 평가셋 질의 돌리기.

색인 (data/media의 *.mp4 전부. 평가 manifest는 보지 않는다):
    python -m eval.run_fp index --media data/media --workers 4 [--reset]   # 영상마다 커밋, 이미 넣은 영상은 건너뜀
    python -m eval.run_fp cluster   # 색인이 끝난 뒤 한 번
질의 (결과 jsonl은 eval.score가 읽는 형식):
    python -m eval.run_fp query --eval data/eval/synth_v1 --out out.jsonl [--speeds grid]
DB 주소는 환경변수 DATABASE_URL.
"""
import argparse
import dataclasses
import json
import os
import resource
import statistics
import time
from multiprocessing import Pool
from pathlib import Path

import psycopg

from app import audio_fp

# 배속 격자: 간격 1.5%, 범위 ±12%. 범위와 간격은 synth_v1(배속 0.90~0.93·1.07~1.10)을 보며 골랐다.
# 보류셋에서 격자 밖인 1.25배속 질의는 R@1 0/20이다 (격자를 넓히면 질의 지연이 그만큼 늘어난다).
# 간격 1.5%. 주파수 칸 폭(15.6Hz)이 1kHz 근처에서 약 1.5%라 이보다 성기면 피크가 칸을 건너뛴다
SPEED_GRID_DEFAULT = "0.88,0.895,0.91,0.925,0.94,0.955,0.97,0.985,1.0,1.015,1.03,1.045,1.06,1.075,1.09,1.105,1.12"


def make_params(args):
    """명령줄 옵션으로 기본 파라미터 일부를 바꾼다 (None이면 기본값)."""
    over = {k: v for k, v in (("index_shifts", getattr(args, "index_shifts", None)),
                               ("query_shifts", getattr(args, "query_shifts", None))) if v is not None}
    return dataclasses.replace(audio_fp.DEFAULT, **over)


def _fp_one(job):
    """작업 프로세스: 영상 하나의 해시를 뽑는다 (DB는 부모만 만진다)."""
    path, p = job
    x = audio_fp.decode_audio(path)
    hashes, ts = audio_fp.fingerprint(x, p, p.index_shifts)
    return Path(path).stem, len(x) / p.sr, hashes, ts, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def cmd_index(args):
    p = make_params(args)
    paths = sorted(str(f) for f in Path(args.media).glob("*.mp4"))
    t0 = time.time()
    # autocommit: 영상마다 커밋 -> 색인 도중에도 다른 연결의 SELECT가 막히지 않고, 중단해도 넣은 영상은 남는다
    with audio_fp.connect(os.environ["DATABASE_URL"]) as conn:
        if args.reset:
            conn.execute("TRUNCATE audio_fp, audio_fp_src RESTART IDENTITY")  # autocommit이라 바로 커밋되어 잠금이 풀린다
        done = {r[0] for r in conn.execute("SELECT video_id FROM audio_fp_src")}
        todo = [f for f in paths if Path(f).stem not in done]  # 이미 넣은 영상은 건너뜀 (중단 후 이어 하기)
        print(f"전체 {len(paths)}편 중 이미 색인 {len(paths) - len(todo)}편 건너뜀")
        with Pool(args.workers) as pool:
            _index_loop(conn, pool, todo, p, t0, args)


def _index_loop(conn, pool, todo, p, t0, args):
    n_hash, n_vid, skipped, rss = 0, 0, [], 0
    for vid, dur, hashes, ts, r in pool.imap_unordered(_fp_one, [(f, p) for f in todo]):
        rss = max(rss, r)
        if len(hashes) == 0:
            skipped.append(vid)  # 소리 없음 등
        if audio_fp.insert_hashes(conn, vid, dur, hashes, ts):  # 영상마다 커밋
            n_hash += len(hashes)
            n_vid += 1
    print(f"색인 영상 {n_vid}편, 해시 {n_hash:,}개, 소리 없음/해시 0: {skipped}")
    print(f"걸린 시간 {time.time() - t0:.0f}s (워커 {args.workers}개, 추출+삽입), 워커 최대 메모리 {rss/1024:.0f}MB")
    print("색인이 끝나면 `python -m eval.run_fp cluster`로 표를 해시 순서로 정리할 것 (질의 속도)")


def cmd_cluster(args):
    """해시 순서대로 표를 다시 쓴다: 같은 해시의 행이 디스크에서 붙어 있어 질의 때 읽는 페이지가 크게 줄어든다.
    표 전체를 잠그는 무거운 작업이라 색인과 분리했다 (색인이 모두 끝난 뒤 한 번)."""
    with audio_fp.connect(os.environ["DATABASE_URL"]) as conn:
        conn.execute("CLUSTER audio_fp USING audio_fp_hash_idx")
        conn.execute("ANALYZE audio_fp")
        size = conn.execute(
            "SELECT pg_size_pretty(pg_total_relation_size('audio_fp')), "
            "pg_size_pretty(pg_relation_size('audio_fp')), pg_size_pretty(pg_indexes_size('audio_fp'))").fetchone()
    print(f"audio_fp 용량: 전체 {size[0]} (표 {size[1]} + 인덱스 {size[2]})")


def cmd_query(args):
    evald = Path(args.eval)
    manifest = [json.loads(l) for l in open(evald / "manifest.jsonl", encoding="utf-8") if l.strip()]
    spec = SPEED_GRID_DEFAULT if args.speeds == "grid" else args.speeds
    speeds = tuple(float(s) for s in spec.split(",")) if spec else (1.0,)
    speeds = tuple(sorted(speeds, key=lambda v: abs(v - 1.0)))  # 1.0에 가까운 것부터 (확실하면 일찍 끝냄)
    p = make_params(args)
    lat, rows = [], []
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        for m in manifest:
            qid = m["query_id"]  # 정답 정보(source_video_id 등)는 쓰지 않는다
            t0 = time.perf_counter()
            x = audio_fp.decode_audio(evald / f"{qid}.wav")
            cands = audio_fp.query_audio(conn, x, p, speeds=speeds)
            lat.append(time.perf_counter() - t0)
            rows.append({"query_id": qid, "candidates": cands, "latency_ms": round(lat[-1] * 1000)})
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"질의 {len(rows)}개, 배속 {len(speeds)}단계")
    print(f"질의 1개 지연(디코드+추출+DB 투표): 중앙 {statistics.median(lat)*1000:.0f}ms, 평균 {statistics.mean(lat)*1000:.0f}ms, "
          f"최악 {max(lat)*1000:.0f}ms; 프로세스 최대 메모리 {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f}MB")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("index")
    a.add_argument("--media", required=True)
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--index-shifts", type=int, default=None, help="색인 격자 수 (기본: 파라미터 기본값)")
    a.add_argument("--reset", action="store_true", help="색인 전에 audio_fp 표를 비운다")
    a.set_defaults(fn=cmd_index)
    c = sub.add_parser("cluster")
    c.set_defaults(fn=cmd_cluster)
    b = sub.add_parser("query")
    b.add_argument("--eval", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--query-shifts", type=int, default=None, help="질의 격자 수")
    b.add_argument("--speeds", default="", help="쉼표로 구분한 배속 목록, 또는 grid(0.88~1.12, 17단계). 비우면 1.0만")
    b.set_defaults(fn=cmd_query)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
