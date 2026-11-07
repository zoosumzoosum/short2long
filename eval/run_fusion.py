"""소리+화면 결합 실행 (호스트 WSL, GPU 필요). 두 단계로 나눈다:

  cache  질의마다 소리·화면의 '원점수'를 창별로 한 번만 뽑아 json으로 저장 (DB·GPU를 쓰는 비싼 단계)
  fuse   캐시 위에서 결합(RRF 등)만 다시 돌려 결과 jsonl을 만든다 (DB·GPU 없이 수 초). 손잡이 조정은 이 위에서 한다

    uv run --group vis python -m eval.run_fusion cache --eval data/eval/synth_mm_v1 --out cache.json --windows 6:3,4:2
    uv run python -m eval.run_fusion fuse --cache cache.json --window 6:3 --out out.jsonl --k 1

DB 주소는 환경변수 DATABASE_URL (명령줄에 쓰지 않는다). 정답 정보(manifest의 source_video_id 등)는 보지 않는다.
창 후보를 뽑는 방법: 소리는 질의 전체의 해시를 한 번 만들어 창 시간대로 잘라 쓰고(창마다 다시 뽑지 않는다),
화면은 질의 전체의 이웃 검색을 한 번 하고 창 안의 프레임만 투표시킨다.
"""
import argparse
import dataclasses
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

from app import audio_fp, fusion
from app import visual_emb as ve

SPEED_GRID = tuple(float(s) for s in
                   "0.88,0.895,0.91,0.925,0.94,0.955,0.97,0.985,1.0,1.015,1.03,1.045,1.06,1.075,1.09,1.105,1.12".split(","))
SPEED_GRID = tuple(sorted(SPEED_GRID, key=lambda v: abs(v - 1.0)))  # eval/run_fp.py와 같은 순서
MODEL = "dinov2b"


def parse_windows(spec):
    """'6:3,4:2' -> [(6.0, 3.0), (4.0, 2.0)] (창 길이:간격)"""
    return [tuple(float(x) for x in s.split(":")) for s in spec.split(",")]


def wkey(win, hop):
    return f"{win:g}:{hop:g}"


def load_cache(path):
    return json.load(open(path, encoding="utf-8"))


def cmd_cache(args):
    evald = Path(args.eval)
    manifest = [json.loads(l) for l in open(evald / "manifest.jsonl", encoding="utf-8") if l.strip()]
    if args.limit:
        manifest = manifest[:args.limit]
    configs = parse_windows(args.windows)
    p = audio_fp.DEFAULT
    qp = ve.QueryParams()
    emb = ve.Embedder(MODEL)
    conn = ve.connect(os.environ["DATABASE_URL"])
    ve.check_params(conn, MODEL)
    a_names = dict(conn.execute("SELECT src_id, video_id FROM audio_fp_src").fetchall())
    v_src, _ = ve.tables(MODEL)
    v_names = dict(conn.execute(f"SELECT src_id, video_id FROM {v_src}").fetchall())

    cache = {wkey(*c): {} for c in configs}
    for n, m in enumerate(manifest):
        qid = m["query_id"]
        t0 = time.perf_counter()
        # ---- 공통(창과 무관): 디코드, 소리 해시, 화면 임베딩·이웃 검색
        x = audio_fp.decode_audio(evald / f"{qid}.wav")
        frames = list(ve.iter_frames(evald / f"{qid}.mp4", 1.0))
        dur = max(len(x) / p.sr, (frames[-1][0] + 1.0) if frames else 0.0)
        speeds = [1.0]
        if len(x):  # 소리가 있으면 질의 전체로 배속을 한 번만 고른다 (창마다 17배속을 다 돌리면 느리다)
            top = audio_fp.query_audio(conn, x, p, speeds=SPEED_GRID, top_k=1)
            if top and top[0]["speed"] != 1.0:
                speeds.append(top[0]["speed"])
        fps_ = {sp: audio_fp.fingerprint(audio_fp.resample_speed(x, sp), p, p.query_shifts) for sp in speeds} if len(x) else {}
        imgs, q_frame = [], []
        for fi, (_, img) in enumerate(frames):
            for _, v in ve.make_views(img, qp.query_views):
                imgs.append(v)
                q_frame.append(fi)
        nn = ve.knn_db(conn, MODEL, emb.embed(np.stack(imgs)), qp) if imgs else np.zeros((0, 5))
        q_time = [t for t, _ in frames]
        shared = time.perf_counter() - t0
        # ---- 창 설정마다: 창별 투표
        for win, hop in configs:
            t1 = time.perf_counter()
            raw = []
            for a, b in fusion.make_windows(dur, win, hop):
                aud = {}
                for sp, (h, ts) in fps_.items():
                    for c in fusion.audio_window_cands(conn, h, ts, a * sp, b * sp, a_names, p):
                        if c["video_id"] not in aud or c["score"] > aud[c["video_id"]]["score"]:
                            aud[c["video_id"]] = c
                aud = sorted(aud.values(), key=lambda c: -c["score"])[:10]
                vis = fusion.vis_window_cands(nn, q_frame, q_time, a, b, ve.vote, qp, v_names)
                raw.append({"a": round(a, 3), "b": round(b, 3), "audio": aud, "vis": vis})
            cache[wkey(win, hop)][qid] = {"dur": dur, "windows": raw, "speeds": speeds,
                                          "ms": round((shared + time.perf_counter() - t1) * 1000)}
        if (n + 1) % 10 == 0:
            print(f"[{n + 1}/{len(manifest)}] 공통 {shared * 1000:.0f}ms", flush=True)
    json.dump(cache, open(args.out, "w", encoding="utf-8"))
    ms = {k: statistics.median(v["ms"] for v in d.values()) for k, d in cache.items()}
    print(f"캐시 저장 {args.out}: 질의 {len(manifest)}개, 창 설정별 추출 지연 중앙(ms, GPU 공유 중) {ms}")
    if emb.device == "cuda":
        print(f"GPU 최대 할당 {emb.peak_vram_mb():.0f}MB")


def params_from_args(args):
    return fusion.FusionParams(
        win=args.win, hop=args.hop, k=args.k, w_audio=args.w_audio, w_vis=args.w_vis, group_tol=args.group_tol,
        merge_tol=args.merge_tol, switch=args.switch, guard_tol=args.guard_tol, bridge=not args.no_bridge, min_windows=args.min_windows)


def fuse_cache(cache_w, p):
    """캐시(창 설정 하나분) -> 결과 행 목록. 캐시에 든 추출 지연(ms)을 latency_ms로 쓴다 (결합 자체는 몇 ms)."""
    rows = []
    for qid, d in cache_w.items():
        t0 = time.perf_counter()
        res = fusion.build_result(d["windows"], p)
        rows.append({"query_id": qid, **res,
                     "latency_ms": round(d["ms"] + (time.perf_counter() - t0) * 1000)})
    return rows


def cmd_fuse(args):
    cache = load_cache(args.cache)
    p = params_from_args(args)
    if args.params_json:  # tune_fusion이 저장한 설정에서 시작하고, --set key=값 으로 일부만 바꾼다
        base = fusion.FusionParams(**json.load(open(args.params_json)))
        over = {}
        for kv in args.set or []:
            k, _, v = kv.partition("=")
            over[k] = json.loads(v)
        p = dataclasses.replace(base, **over)
    rows = fuse_cache(cache[wkey(p.win, p.hop)], p)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(dataclasses.asdict(p), open(args.out + ".params.json", "w"))  # W&B config용
    print(f"결과 {len(rows)}개 -> {args.out}  설정 {dataclasses.asdict(p)}")


def add_param_args(ap):
    d = fusion.FusionParams()
    ap.add_argument("--win", type=float, default=d.win)
    ap.add_argument("--hop", type=float, default=d.hop)
    ap.add_argument("--k", type=float, default=d.k)
    ap.add_argument("--w-audio", type=float, default=d.w_audio)
    ap.add_argument("--w-vis", type=float, default=d.w_vis)
    ap.add_argument("--group-tol", type=float, default=d.group_tol)
    ap.add_argument("--switch", type=float, default=d.switch)
    ap.add_argument("--guard-tol", type=float, default=d.guard_tol)
    ap.add_argument("--merge-tol", type=float, default=d.merge_tol)
    ap.add_argument("--no-bridge", action="store_true")
    ap.add_argument("--min-windows", type=int, default=d.min_windows)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--eval", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--windows", default="6:3", help="창 길이:간격(초)을 쉼표로, 예 6:3,4:2")
    c.add_argument("--limit", type=int, default=0)
    c.set_defaults(fn=cmd_cache)
    f = sub.add_parser("fuse")
    f.add_argument("--cache", required=True)
    f.add_argument("--out", required=True)
    add_param_args(f)
    f.add_argument("--params-json", help="tune_fusion --out 이 만든 설정 (있으면 위 옵션 대신 이것을 쓴다)")
    f.add_argument("--set", nargs="*", help="--params-json 위에 덮어쓸 key=JSON값, 예 switch=0 bridge=false")
    f.set_defaults(fn=cmd_fuse)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
