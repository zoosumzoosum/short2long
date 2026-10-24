"""매칭 결과 채점: Recall@1, Recall@5, 구간 오차, 변형별·format별 표.

결과 파일(jsonl) 한 줄: {"query_id": "q001", "candidates": [{"video_id": ..., "start_sec": ..., "score": ...}]}
후보는 score 큰 순서로 본다 (같으면 파일 순서). 결과가 없는 질의는 실패로 센다.
정답 = 영상 ID 일치 + 시작 시각 오차 <= --tol 초.

구간 결과(멀티모달 평가셋 v2): 한 줄에 "segments": [{"q_start","q_end","video_id","start","score"}, ...]를 낼 수 있다.
q_start~q_end는 질의 클립 안의 시간(초), start는 그 구간이 원본 영상에서 시작하는 시각.
segments가 없는 결과(기존 방법)는 1위 후보를 '질의 전체를 덮는 구간 하나'로 보고 채점한다.

구간 채점(SegR) 규칙: 정답 구간마다, 질의 시간으로 그 정답 구간의 50% 이상 겹치는 예측 구간 중 score 1등이
영상이 같고 시각 오차(구간 시작을 정답 구간 시작에 맞춰 환산) <= tol이면 정답. SegR = 맞힌 정답 구간 / 전체 정답 구간.
seg_wrong = 1등이 있는데 틀림, seg_missing = 겹치는 예측이 없음, seg_extra = 정답 구간이 없는 곳(질의 시간 기준)에
절반 넘게 놓인 예측 구간 수. swap 질의(audio_decoy 있음)는 1위 영상이 디코이면 '속았다'(fooled).
compilation 질의는 정답이 여럿이라 R@k/V@k 표에서는 빼고 SegR 표에만 넣는다.
주의: 단일 질의(both/mute/swap)는 정답 구간이 질의 전체라 예측 구간이 질의 길이의 절반 이상이어야 채점된다
-> 결합 방법은 같은 영상에서 시간차가 일정하게 이어지는 창을 하나의 구간으로 합쳐 내야 한다.
시각 환산(start + (t.q_start - p.q_start))은 1배속을 가정한다 (speed 변형 질의는 구간이 길수록 조금 어긋남).

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


def truth_segments(m):
    """정답 구간 목록. segments가 없으면 질의 전체가 원본 한 구간."""
    if m.get("segments"):
        return m["segments"]
    dur = m.get("query_dur_sec") or (m["end_sec"] - m["start_sec"])
    return [{"q_start": 0.0, "q_end": dur, "video_id": m["source_video_id"], "start": m["start_sec"]}]


def candidates_of(res):
    """R@k/V@k용 후보 목록. segments만 낸 결과는 구간을 (영상, 시작 시각, 점수) 후보로 본다."""
    if res.get("candidates") is not None:
        return res["candidates"]
    return [{"video_id": s["video_id"], "start_sec": s["start"], "score": s["score"]}
            for s in res.get("segments", [])]


def pred_segments(m, res):
    """예측 구간 목록. 결과가 segments를 안 냈으면 1위 후보를 질의 전체 구간 하나로 본다."""
    if res is None:
        return []
    if res.get("segments") is not None:
        return res["segments"]
    ranked = rank_per_video(candidates_of(res))
    if not ranked:
        return []
    top = ranked[0]
    q_end = max(s["q_end"] for s in truth_segments(m))
    return [{"q_start": 0.0, "q_end": q_end, "video_id": top["video_id"],
             "start": top["start_sec"], "score": top["score"]}]


def overlap(a, b):
    """두 시간 구간(q_start, q_end)이 겹치는 길이(초)."""
    return max(0.0, min(a["q_end"], b["q_end"]) - max(a["q_start"], b["q_start"]))


def judge_segments(m, res, tol):
    """구간 채점 한 질의분: 정답 구간 수·맞힌 수·틀린 수·못 낸 수·엉뚱한 곳 예측 수, 속았는지."""
    truths, preds = truth_segments(m), pred_segments(m, res)
    out = {"seg_total": len(truths), "seg_correct": 0, "seg_wrong": 0, "seg_missing": 0, "seg_extra": 0}
    for t in truths:
        need = 0.5 * (t["q_end"] - t["q_start"])
        near = [p for p in preds if overlap(p, t) > 0 and overlap(p, t) >= need]
        if not near:
            out["seg_missing"] += 1
            continue
        # 승자 = 이 정답 구간과 더 많이 겹친 예측, 겹침이 (0.1초 단위로) 같으면 score 1등.
        # score만 보면 옆 구간까지 길게 덮은 점수 높은 예측이 가로챈다. 동점이면 먼저 나온 것
        w = max(near, key=lambda p: (round(overlap(p, t), 1), p["score"]))
        shifted = w["start"] + (t["q_start"] - w["q_start"])  # 예측 구간 시작이 달라도 정답 구간 시작 기준으로 환산
        if w["video_id"] == t["video_id"] and abs(shifted - t["start"]) <= tol:
            out["seg_correct"] += 1
        else:
            out["seg_wrong"] += 1
    for p in preds:
        covered = sum(overlap(p, t) for t in truths)
        if covered < 0.5 * (p["q_end"] - p["q_start"]):
            out["seg_extra"] += 1
    out["fooled"] = None
    if m.get("audio_decoy"):
        out["fooled"] = bool(preds) and max(preds, key=lambda p: p["score"])["video_id"] == m["audio_decoy"]["video_id"]
    return out


def score_queries(manifest, results, tol=3.0):
    """질의마다 {transform, format, rank, err, 구간 채점 항목}을 돌려준다. 결과 없으면 rank=None."""
    by_id = {}
    for r in results:
        if r["query_id"] in by_id:
            raise ValueError(f"결과에 query_id가 두 번 나옴: {r['query_id']}")
        by_id[r["query_id"]] = r
    rows = []
    for m in manifest:
        res = by_id.get(m["query_id"])
        rank, err, vrank, verr = judge(m, candidates_of(res), tol) if res else (None,) * 4
        rows.append({"query_id": m["query_id"], "transform": m["transform"],
                     "format": m["format"], "rank": rank, "err": err,
                     "vrank": vrank, "verr": verr, "missing": res is None,
                     "kind": m.get("kind"), **judge_segments(m, res, tol)})
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


def single_rows(rows):
    """R@k/V@k는 정답이 하나인 질의만 대상으로 한다 (compilation 제외)."""
    return [r for r in rows if r["kind"] != "compilation"]


def summarize_seg(rows):
    n_seg = sum(r["seg_total"] for r in rows)
    fooled = [r["fooled"] for r in rows if r["fooled"] is not None]
    return {"n": len(rows), "n_seg": n_seg,
            "SegR": sum(r["seg_correct"] for r in rows) / n_seg if n_seg else 0.0,
            "seg_wrong": sum(r["seg_wrong"] for r in rows),
            "seg_missing": sum(r["seg_missing"] for r in rows),
            "seg_extra": sum(r["seg_extra"] for r in rows),
            "n_swap": len(fooled), "fooled_rate": sum(fooled) / len(fooled) if fooled else None}


def group_by(rows, key, summarizer=summarize):
    groups = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    return {k: summarizer(v) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def fmt_seg_table(title, groups):
    lines = [f"[{title}]  (SegR=정답 구간 중 맞힌 비율, wrong=틀림, missing=예측 없음, extra=정답 없는 곳의 예측, 속은비율=swap에서 1위가 디코이)",
             f"{'':14}{'질의':>5}{'구간':>6}{'SegR':>8}{'wrong':>7}{'missing':>9}{'extra':>7}{'swap수':>7}{'속은비율':>9}"]
    for name, s in groups.items():
        fr = "-" if s["fooled_rate"] is None else f"{s['fooled_rate']:.3f}"
        lines.append(f"{str(name):14}{s['n']:>5}{s['n_seg']:>6}{s['SegR']:>8.3f}{s['seg_wrong']:>7}"
                     f"{s['seg_missing']:>9}{s['seg_extra']:>7}{s['n_swap']:>7}{fr:>9}")
    return "\n".join(lines)


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
    singles = single_rows(rows)
    if len(singles) != len(rows):
        print(f"R@k/V@k 표는 정답이 하나인 질의 {len(singles)}개만 (compilation {len(rows) - len(singles)}개는 SegR 표에서)")
    print()
    print(fmt_table("전체", {"all": summarize(singles)}))
    print()
    print(fmt_table("변형별", group_by(singles, "transform")))
    print()
    print(fmt_table("format별", group_by(singles, "format")))
    if any(r["kind"] for r in rows):  # 멀티모달 평가셋일 때만
        print()
        print(fmt_seg_table("구간 채점: 전체", {"all": summarize_seg(rows)}))
        print()
        print(fmt_seg_table("구간 채점: 종류별", group_by(rows, "kind", summarize_seg)))


if __name__ == "__main__":
    main()
