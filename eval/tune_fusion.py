"""결합 손잡이 조정 (캐시 위에서, DB·GPU 없음). dev 평가셋 하나만 쓴다.

과적합을 줄이려고 손잡이를 한 번에 하나씩만 돌리고(좌표 하강), 5겹 교차검증으로
'조정에 안 쓴 질의'에서의 점수를 함께 본다 (질의를 종류별로 고르게 5묶음으로 나눔).

    uv run python -m eval.tune_fusion --cache C.json --eval data/eval/synth_mm_v1 [--out best.json]

목적함수 = (SegR + 비-compilation R@1 + (1 - swap 속은 비율)) / 3  -- 세 대표 지표를 똑같이 본다.
"""
import argparse
import dataclasses
import json
import random

from app import fusion
from eval import run_fusion
from eval.score import read_jsonl, score_queries, single_rows, summarize, summarize_seg

WINDOWS = [(3.0, 1.5), (4.0, 2.0), (6.0, 3.0), (8.0, 4.0)]
KNOBS = {
    "wh": WINDOWS,
    "k": [1.0, 3.0, 10.0, 30.0, 60.0],
    "w_vis": [0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0],
    "group_tol": [1.5, 3.0, 5.0, 1e6],
    "merge_tol": [1.5, 3.0, 5.0],
    "switch": [0.0, 0.5, 1.0, 2.0, 4.0, 8.0],
    "bridge": [True, False],
    "guard_tol": [1.5, 3.0, 5.0],
}


def with_knob(p, name, value):
    if name == "wh":
        return dataclasses.replace(p, win=value[0], hop=value[1])
    return dataclasses.replace(p, **{name: value})


def metrics(rows):
    """대표 지표 묶음 + 목적함수."""
    seg, sing = summarize_seg(rows), summarize(single_rows(rows))
    fooled = seg["fooled_rate"] if seg["fooled_rate"] is not None else 0.0
    return {"SegR": seg["SegR"], "R@1": sing["R@1"], "V@1": sing["V@1"], "fooled": fooled,
            "obj": (seg["SegR"] + sing["R@1"] + (1 - fooled)) / 3}


def evaluate(manifest, caches, p):
    """manifest(질의 목록)에 대해 p로 결합 -> (지표, 채점 행)."""
    cache_w = caches[run_fusion.wkey(p.win, p.hop)]
    ids = {m["query_id"] for m in manifest}
    res = run_fusion.fuse_cache({q: d for q, d in cache_w.items() if q in ids}, p)
    rows, _ = score_queries(manifest, res)
    return metrics(rows), rows


def descend(manifest, caches, base, knobs, rounds=2):
    """좌표 하강을 시작점 4곳(영상 단위 묶기 여부 x 화면 가중 1.0/1.5)에서 돌려 가장 좋은 결과를 쓴다.
    한 곳에서만 시작하면 '손잡이 하나만 바꿔서는 안 좋아지는' 조합(영상 단위 묶기 + 큰 화면 가중)을 못 찾는다."""
    starts = [dataclasses.replace(base, group_tol=g, w_vis=w) for g in (base.group_tol, 1e6) for w in (base.w_vis, 1.5)]
    results = [descend_one(manifest, caches, s, knobs, rounds) for s in starts]
    return max(results, key=lambda r: r[1])


def descend_one(manifest, caches, base, knobs, rounds=2):
    """좌표 하강: 손잡이를 하나씩 돌며 목적함수를 최대로 하는 값을 고른다 (동점이면 현재 값 유지 = 변화 최소)."""
    p, best = base, evaluate(manifest, caches, base)[0]["obj"]
    for _ in range(rounds):
        changed = False
        for name, values in knobs.items():
            for v in values:
                q = with_knob(p, name, v)
                o = evaluate(manifest, caches, q)[0]["obj"]
                if o > best + 1e-9:
                    p, best, changed = q, o, True
        if not changed:
            break
    return p, best


def folds(manifest, n=5, seed=0):
    """종류(kind)별로 섞어 n묶음으로 나눈다."""
    rng = random.Random(seed)
    out = [[] for _ in range(n)]
    by_kind = {}
    for m in manifest:
        by_kind.setdefault(m["kind"], []).append(m)
    i = 0
    for kind in sorted(by_kind):
        ms = by_kind[kind]
        rng.shuffle(ms)
        for m in ms:
            out[i % n].append(m)
            i += 1
    return out


def cross_validate(manifest, caches, base, knobs, n=5):
    """묶음 하나를 빼고 나머지로 조정 -> 뺀 묶음에서 점수. 묶음별 점수를 질의 수로 가중 평균."""
    fs = folds(manifest, n)
    tot = {"SegR": 0.0, "R@1": 0.0, "V@1": 0.0, "fooled": 0.0, "obj": 0.0}
    chosen = []
    for i, held in enumerate(fs):
        train = [m for j, f in enumerate(fs) if j != i for m in f]
        p, _ = descend(train, caches, base, knobs)
        mt = evaluate(held, caches, p)[0]
        w = len(held) / len(manifest)
        for k in tot:
            tot[k] += w * mt[k]
        chosen.append({"win": p.win, "hop": p.hop, "k": p.k, "w_vis": p.w_vis})
    return tot, chosen


def fmt(m):
    return "  ".join(f"{k}={v:.3f}" for k, v in m.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", help="고른 설정을 쓸 json")
    ap.add_argument("--no-cv", action="store_true")
    args = ap.parse_args()
    manifest = read_jsonl(f"{args.eval}/manifest.jsonl")
    caches = run_fusion.load_cache(args.cache)
    knobs = dict(KNOBS)
    base = fusion.FusionParams()
    print(f"조정 손잡이: {list(knobs)}  ")
    print("기본값 :", fmt(evaluate(manifest, caches, base)[0]))
    p, _ = descend(manifest, caches, base, knobs)
    m, rows = evaluate(manifest, caches, p)
    print("dev 전체로 조정 :", fmt(m))
    print("고른 설정 :", {k: v for k, v in dataclasses.asdict(p).items() if v != getattr(base, k) or k in ("win", "hop")})
    if not args.no_cv:
        cv, chosen = cross_validate(manifest, caches, base, knobs)
        print("5겹 교차검증(조정에 안 쓴 질의) :", fmt(cv))
        print("묶음별 고른 창·k·w_vis :", chosen)
    if args.out:
        json.dump(dataclasses.asdict(p), open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
