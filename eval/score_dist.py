"""정답 점수와 오답 점수가 갈리는지 본다 ("못 찾음" 문턱을 둘 수 있나).

  정답 점수 = 정답 영상 후보의 점수 (영상+시각이 맞은 질의만)
  오답 점수 = 정답이 아닌 영상 중 가장 높은 후보의 점수
              (색인에 원본이 없는 질의가 받을 최고 점수의 대용: 원본이 없으면 1위는 늘 '오답'이므로)
실행: python -m eval.score_dist --manifest M.jsonl --results R.jsonl [--tol 3]
출력: 두 분포의 분위수, 문턱 하나로 둘을 나눌 때 최선의 정확도, AUC(정답이 오답보다 높을 확률).
"""
import argparse

from eval.score import read_jsonl


def collect(manifest, results, tol):
    by_id = {r["query_id"]: r for r in results}
    pos, neg = [], []
    for m in manifest:
        res = by_id.get(m["query_id"])
        if not res or not res["candidates"]:
            continue
        wrong = [c["score"] for c in res["candidates"] if c["video_id"] != m["source_video_id"]]
        right = [c["score"] for c in res["candidates"]
                 if c["video_id"] == m["source_video_id"] and abs(c["start_sec"] - m["start_sec"]) <= tol]
        if right:
            pos.append(max(right))
        if wrong:
            neg.append(max(wrong))
    return pos, neg


def quantiles(v, qs=(0, 10, 50, 90, 100)):
    s = sorted(v)
    return {q: s[min(len(s) - 1, int(round(q / 100 * (len(s) - 1))))] for q in qs}


def auc(pos, neg):
    """정답 점수가 무작위 오답 점수보다 클 확률 (동점은 반)."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def best_threshold(pos, neg):
    """문턱 th: 점수 >= th면 '찾음'. 정답을 찾음으로, 오답을 못찾음으로 맞히는 비율의 평균이 최대인 th."""
    best = (0.0, None)
    for th in sorted(set(pos + neg)):
        acc = (sum(p >= th for p in pos) / len(pos) + sum(n < th for n in neg) / len(neg)) / 2
        if acc > best[0]:
            best = (acc, th)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--tol", type=float, default=3.0)
    args = ap.parse_args()
    pos, neg = collect(read_jsonl(args.manifest), read_jsonl(args.results), args.tol)
    print(f"정답 점수 n={len(pos)}  분위수(0/10/50/90/100%): {quantiles(pos)}")
    print(f"오답 점수 n={len(neg)}  분위수(0/10/50/90/100%): {quantiles(neg)}")
    a = auc(pos, neg)
    acc, th = best_threshold(pos, neg)
    print(f"AUC {a:.3f}; 최선의 문턱 {th} 에서 균형 정확도 {acc:.3f}")


if __name__ == "__main__":
    main()
