"""화면 임베딩 색인 만들기 / 평가셋 질의 돌리기 (호스트 WSL에서 uv로 실행; GPU가 필요해서 컨테이너가 아님).

    uv run --group vis python -m eval.run_vis index --media data/media --models dinov2s,clip --decoders 3
    uv run --group vis python -m eval.run_vis query --model dinov2s --eval data/eval/synth_vis_v1 --out out.jsonl

index: 영상 프레임 디코드·자르기는 CPU 프로세스 여럿, 임베딩은 이 프로세스(GPU)가 한다.
       영상마다 커밋하고 이미 넣은 영상은 건너뛴다 (중단 후 이어 하기). 디코드는 한 번만 하고 모델 여러 개에 같이 쓴다.
query: 결과 jsonl은 eval.score가 읽는 형식 + 지연(latency_ms)과 단계별 시간. 정답 정보는 보지 않는다.
DB 주소는 환경변수 DATABASE_URL (명령줄에 쓰지 않는다: ps에 보임).
서비스 이미지(Docker)에는 없다: 호스트 GPU에서 `uv run --group vis`로 실행한다. 질의 p50: GPU(RTX 4050) 535ms / CPU만 6.7초 (조건은 app/visual_emb.py 맨 위).
"""
import argparse
import dataclasses
import json
import multiprocessing as mp
import os
import queue
import statistics
import time
from pathlib import Path

import numpy as np

from app import visual_emb as ve

CHUNK = 48  # 디코더가 한 번에 넘기는 프레임 수 (메모리: 48 x view 4 x 150KB = 30MB)


def _decoder(task_q, out_q, fps, kinds):
    """CPU 프로세스: 영상을 받아 프레임 -> view 이미지로 바꿔 조각(chunk)으로 넘긴다."""
    while True:
        path = task_q.get()
        if path is None:
            return
        vid = Path(path).stem
        try:
            times, vids, imgs, last = [], [], [], 0.0
            for t, img in ve.iter_frames(path, fps):
                last = t
                for k, v in ve.make_views(img, kinds):
                    times.append(t)
                    vids.append(ve.VIEW_IDS[k])
                    imgs.append(v)
                if len(times) >= CHUNK * len(kinds):
                    out_q.put((vid, "chunk", times, vids, np.stack(imgs)))
                    times, vids, imgs = [], [], []
            if times:
                out_q.put((vid, "chunk", times, vids, np.stack(imgs)))
            out_q.put((vid, "done", last, None, None))
        except Exception as e:  # 한 영상이 깨져도 나머지는 계속
            out_q.put((vid, "error", str(e)[:200], None, None))


def cmd_index(args):
    models = args.models.split(",")
    paths = sorted(str(f) for f in Path(args.media).glob("*.mp4"))
    conn = ve.connect(os.environ["DATABASE_URL"])
    for m in models:
        ve.tables(m)  # 모르는 모델 이름이면 모델을 올리기 전에 ValueError
    # 모델(CUDA)을 먼저 올리고 디코더 프로세스는 그 뒤에 띄운다. 그래서 fork가 아니라 spawn을 쓴다:
    # fork하면 자식이 부모의 CUDA 상태를 물려받아 깨진다.
    ctx = mp.get_context("spawn")
    embedders = {m: ve.Embedder(m) for m in models}
    for m, e in embedders.items():
        conn.execute(ve.ddl(m, e.dim))
        if args.reset:
            s, f = ve.tables(m)
            conn.execute(f"TRUNCATE {s}, {f} RESTART IDENTITY")
        # 이미 들어 있는 색인이 다른 설정(모델 revision·fps·views)이면 섞어 넣지 않는다
        ve.check_params(conn, m, args.fps, keys=tuple(ve.index_params(m)), views=args.views)
    need = {}  # 영상 -> 아직 색인 안 된 모델들
    done = {m: {r[0] for r in conn.execute(f"SELECT video_id FROM {ve.tables(m)[0]}")} for m in models}
    for p in paths:
        vid = Path(p).stem
        miss = [m for m in models if vid not in done[m]]
        if miss:
            need[vid] = miss
    todo = [p for p in paths if Path(p).stem in need]
    if args.limit:
        todo = todo[:args.limit]
        need = {Path(p).stem: need[Path(p).stem] for p in todo}
    print(f"전체 {len(paths)}편 중 색인할 것 {len(todo)}편, 모델 {models}", flush=True)

    task_q, out_q = ctx.Queue(), ctx.Queue(maxsize=8)
    for p in todo:
        task_q.put(p)
    for _ in range(args.decoders):
        task_q.put(None)
    procs = [ctx.Process(target=_decoder, args=(task_q, out_q, args.fps, args.views), daemon=True)
             for _ in range(args.decoders)]
    for pr in procs:
        pr.start()

    t0 = time.time()
    part = {}  # 영상 -> {"times": [], "views": [], "embs": {모델: [배열]}}
    n_done, n_rows, errors = 0, 0, []
    t_embed = 0.0
    while n_done + len(errors) < len(todo):
        try:
            vid, kind, a, b, imgs = out_q.get(timeout=30)
        except queue.Empty:
            # 30초 동안 소식이 없다: 디코더가 죽었는지 본다 (정상 종료는 exitcode 0, 죽으면 0이 아님)
            dead = [pr.exitcode for pr in procs if pr.exitcode not in (None, 0)]
            if dead:
                lost = sorted(set(part))  # 조각만 오고 끝 신호가 없는 영상 (죽은 프로세스가 맡았을 가능성이 큼)
                left = len(todo) - n_done - len(errors)
                print(f"디코더 사망 exitcode {dead}. 색인 중단. 끝내지 못한 영상 {left}편 (그중 진행 중이던 것: {lost}). "
                      f"같은 명령을 다시 실행하면 이미 넣은 영상은 건너뛰고 이어 한다.", flush=True)
                raise SystemExit(1)
            continue
        if kind == "error":
            errors.append((vid, a))
            part.pop(vid, None)
            print(f"실패 {vid}: {a}", flush=True)
            continue
        st = part.setdefault(vid, {"times": [], "views": [], "embs": {m: [] for m in need[vid]}})
        if kind == "chunk":
            te = time.time()
            for m in need[vid]:
                st["embs"][m].append(embedders[m].embed(imgs))
            t_embed += time.time() - te
            st["times"] += a
            st["views"] += b
            continue
        # done: a = 마지막 프레임 시각
        st = part.pop(vid)
        for m in need[vid]:
            embs = np.concatenate(st["embs"][m]) if st["embs"][m] else np.zeros((0, embedders[m].dim))
            ve.index_rows(conn, m, vid, a, st["times"], st["views"], embs, ve.index_params(m, args.fps, args.views))
        n_done += 1
        n_rows += len(st["times"])
        if n_done % 10 == 0 or n_done == len(todo):
            el = time.time() - t0
            print(f"[{n_done}/{len(todo)}] 행 {n_rows:,} 경과 {el:.0f}s (임베딩 {t_embed:.0f}s)", flush=True)
    for pr in procs:
        pr.join(timeout=10)
    el = time.time() - t0
    print(f"색인 영상 {n_done}편, 행(프레임x view) {n_rows:,}, 실패 {len(errors)}: {errors}")
    print(f"걸린 시간 {el:.0f}s (디코더 {args.decoders}개, 임베딩 GPU 합계 {t_embed:.0f}s, 모델 {len(models)}개 동시)")
    for m, e in embedders.items():
        s, f = ve.tables(m)
        size = conn.execute(f"SELECT pg_size_pretty(pg_total_relation_size('{f}')), pg_size_pretty(pg_relation_size('{f}')), "
                            f"pg_size_pretty(pg_indexes_size('{f}')), (SELECT count(*) FROM {f})").fetchone()
        print(f"{m}: dim {e.dim}, {f} 전체 {size[0]} (표 {size[1]} + 인덱스 {size[2]}), 행 {size[3]:,}")
    print(f"GPU 최대 할당 {max(e.peak_vram_mb() for e in embedders.values()):.0f}MB (프로세스 전체, 모델 {len(models)}개 로드 상태)")


def load_manifest(evald):
    return [json.loads(l) for l in open(Path(evald) / "manifest.jsonl", encoding="utf-8") if l.strip()]


def cmd_query(args):
    evald = Path(args.eval)
    manifest = load_manifest(evald)
    if args.limit:
        manifest = manifest[:args.limit]
    ve.tables(args.model)  # 모르는 모델 이름이면 바로 ValueError
    emb = ve.Embedder(args.model)
    qp = ve.QueryParams(topk=args.topk, ef_search=args.ef, sim_floor=args.sim_floor, win=args.win,
                        max_per_frame=args.max_per_frame, query_views=args.query_views, trim=args.trim, power=args.power,
                        flip=args.flip, index_views=args.index_views)
    conn = ve.connect(os.environ["DATABASE_URL"])
    # 첫 질의는 CUDA·HNSW 캐시가 차가워 느리다 -> 지연 통계에서 빼기 위해 한 번 데운다 (결과는 버림)
    ve.query_clip(conn, args.model, emb, evald / f"{manifest[0]['query_id']}.mp4", qp, args.fps)
    import torch
    if emb.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rows, lat, parts = [], [], {"decode": [], "embed": [], "knn": [], "vote": []}
    for m in manifest:
        qid = m["query_id"]  # 정답 정보(source_video_id 등)는 쓰지 않는다
        t0 = time.perf_counter()
        cands, tm = ve.query_clip(conn, args.model, emb, evald / f"{qid}.mp4", qp, args.fps)
        lat.append(time.perf_counter() - t0)
        for k in parts:
            parts[k].append(tm.get(k, 0.0))
        rows.append({"query_id": qid, "candidates": cands, "latency_ms": round(lat[-1] * 1000),
                     "stage_ms": {k: round(tm.get(k, 0) * 1000) for k in parts}})
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    srt = sorted(lat)
    p = lambda q: srt[min(len(srt) - 1, int(np.ceil(q / 100 * len(srt))) - 1)] * 1000
    print(f"모델 {args.model}, 질의 {len(rows)}개, 설정 {dataclasses.asdict(qp)}")
    print(f"질의 1개 지연(디코드+임베딩+DB+투표): p50 {p(50):.0f}ms, p95 {p(95):.0f}ms, 최악 {srt[-1]*1000:.0f}ms; "
          + ", ".join(f"{k} 평균 {statistics.mean(v)*1000:.0f}ms" for k, v in parts.items()))
    if emb.device == "cuda":
        print(f"GPU 최대 할당(질의 구간) {emb.peak_vram_mb():.0f}MB")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("index")
    a.add_argument("--media", required=True)
    a.add_argument("--models", default="dinov2s")
    a.add_argument("--decoders", type=int, default=3)
    a.add_argument("--fps", type=float, default=1.0)
    a.add_argument("--views", default="cazf", help="색인할 view 글자들 (c a z f)")
    a.add_argument("--limit", type=int, default=0, help="앞에서 N편만 (시험용)")
    a.add_argument("--reset", action="store_true", help="색인 전에 이 모델의 표를 비운다")
    a.set_defaults(fn=cmd_index)
    b = sub.add_parser("query")
    b.add_argument("--model", required=True)
    b.add_argument("--eval", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--fps", type=float, default=1.0)
    b.add_argument("--topk", type=int, default=20)
    b.add_argument("--ef", type=int, default=100)
    b.add_argument("--sim-floor", type=float, default=0.3)
    b.add_argument("--win", type=int, default=2)
    b.add_argument("--max-per-frame", type=int, default=5)
    b.add_argument("--query-views", default="cz")
    b.add_argument("--index-views", default="cazf")
    b.add_argument("--power", type=float, default=1.0)
    b.add_argument("--trim", type=float, default=0.0, help="단색 가장자리 띠를 자르는 표준편차 기준 (0 = 안 함)")
    b.add_argument("--flip", action="store_true", help="좌우반전 질의도 같이 찾기")
    b.add_argument("--limit", type=int, default=0)
    b.set_defaults(fn=cmd_query)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
