"""매칭 결과 채점: Recall@1, Recall@5, 구간 오차, 변형별·format별 표.

결과 파일(jsonl) 한 줄: {"query_id": "q001", "candidates": [{"video_id": ..., "start_sec": ..., "score": ...}]}
후보는 score 큰 순서로 본다 (같으면 파일 순서). 결과가 없는 질의는 실패로 센다.
정답 = 영상 ID 일치 + 시작 시각 오차 <= --tol 초.

실행: python -m eval.score --manifest M.jsonl --results R.jsonl [--tol 3]
"""
import argparse
import json
import statistics
import sys

KS = (1, 5)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rank_per_video(candidates):
    """score 큰 순으로 정렬하되 영상마다 최고 후보 하나만 남긴다.
    같은 영상의 구간 후보가 top5를 채우면 다른 영상이 밀려나 Recall@5가 왜곡되기 때문이다.
    sorted는 안정 정렬이라 동점은 파일 순서."""
    seen, ranked = set(), []
    for c in sorted(candidates, key=lambda c: -c["score"]):
        if c["video_id"] not in seen:
            seen.add(c["video_id"])
            ranked.append(c)
    return ranked


def judge(truth, candidates, tol):
    """(구간 정답 순위, 그 오차, 영상 정답 순위, 그 오차). 없으면 None."""
    rank = err = vrank = verr = None
    for r, c in enumerate(rank_per_video(candidates), start=1):
        if c["video_id"] != truth["source_video_id"]:
            continue
        vrank, verr = r, abs(c["start_sec"] - truth["start_sec"])  # 영상은 맞음 (시각 무시)
        if verr <= tol:
            rank, err = r, verr
        break  # 영상당 후보가 하나뿐이라 더 볼 필요 없음
    return rank, err, vrank, verr


def score_queries(manifest, results, tol=3.0):
    """질의마다 {transform, format, rank, err}를 돌려준다. 결과 없으면 rank=None."""
    by_id = {}
    for r in results:
        if r["query_id"] in by_id:
            raise ValueError(f"결과에 query_id가 두 번 나옴: {r['query_id']}")
        by_id[r["query_id"]] = r
    rows = []
    for m in manifest:
        res = by_id.get(m["query_id"])
        rank, err, vrank, verr = judge(m, res["candidates"], tol) if res else (None,) * 4
        rows.append({"query_id": m["query_id"], "transform": m["transform"],
                     "format": m["format"], "rank": rank, "err": err,
                     "vrank": vrank, "verr": verr, "missing": res is None})
    unknown = sorted(set(by_id) - {m["query_id"] for m in manifest})
    return rows, unknown


def summarize(rows):
    """행 묶음 하나의 요약. 오차는 1위가 정답인 질의만 대상으로 한다."""
    n = len(rows)
    out = {"n": n}
    for k in KS:
        hits = sum(1 for r in rows if r["rank"] is not None and r["rank"] <= k)
        out[f"R@{k}"] = hits / n if n else 0.0
    for k in KS:  # 영상 단위: 시각은 무시하고 영상 ID만 맞으면 정답
        hits = sum(1 for r in rows if r["vrank"] is not None and r["vrank"] <= k)
        out[f"V@{k}"] = hits / n if n else 0.0
    errs = [r["err"] for r in rows if r["rank"] == 1]
    out["err_mean"] = statistics.mean(errs) if errs else None
    out["err_median"] = statistics.median(errs) if errs else None
    # 1위 영상이 맞았을 때 시각 오차 (tol로 자르지 않음 → 영상은 찾았는데 시각이 얼마나 빗나갔나)
    verrs = [r["verr"] for r in rows if r["vrank"] == 1]
    out["verr_median"] = statistics.median(verrs) if verrs else None
    return out


def group_by(rows, key):
    groups = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    return {k: summarize(v) for k, v in sorted(groups.items())}


def fmt_table(title, groups):
    lines = [f"[{title}]  (R=영상+시각 정답, V=영상만 정답, err=1위가 R정답일 때, 영상맞음오차=1위 영상이 맞았을 때 tol 무시)",
             f"{'':10}{'n':>5}{'R@1':>8}{'R@5':>8}{'V@1':>8}{'V@5':>8}"
             f"{'err평균':>10}{'err중앙':>10}{'영상맞음오차중앙':>18}"]
    for name, s in groups.items():
        c = ["-" if s[k] is None else f"{s[k]:.2f}" for k in ("err_mean", "err_median", "verr_median")]
        lines.append(f"{name:10}{s['n']:>5}{s['R@1']:>8.3f}{s['R@5']:>8.3f}{s['V@1']:>8.3f}{s['V@5']:>8.3f}"
                     f"{c[0]:>10}{c[1]:>10}{c[2]:>18}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="매칭 결과 채점")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--tol", type=float, default=3.0, help="시작 시각 허용 오차(초)")
    args = ap.parse_args()

    rows, unknown = score_queries(read_jsonl(args.manifest), read_jsonl(args.results), args.tol)
    print(f"허용 오차 {args.tol}s, 질의 {len(rows)}개, 결과 없음 {sum(r['missing'] for r in rows)}개 (실패로 계산)")
    if unknown:
        print(f"경고: manifest에 없는 query_id가 결과에 있음 {len(unknown)}개 (무시): {unknown[:5]}", file=sys.stderr)
    print()
    print(fmt_table("전체", {"all": summarize(rows)}))
    print()
    print(fmt_table("변형별", group_by(rows, "transform")))
    print()
    print(fmt_table("format별", group_by(rows, "format")))


if __name__ == "__main__":
    main()
