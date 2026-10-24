"""평가 결과 하나를 W&B run 하나로 올린다. 지표는 eval/score.py와 같은 함수로 계산한다.

실행(호스트 WSL, 컨테이너 불필요):
  uv run --group eval python -m eval.log_wandb --manifest M.jsonl --results R.jsonl (줄 이어서)
      --name A-fp-grid1 --group audio-match-wk3 --config method=fingerprint index_shifts=1

summary 키는 평평하게: R@1, R@1/noise(변형별), R@1/format=short(format별), latency_p50 등.
프로젝트는 W&B 기본(비공개)으로 만들어지며, 이 코드는 공개 설정을 건드리지 않는다.
"""
import argparse
import hashlib
import json
import math

from eval.score import group_by, read_jsonl, score_queries, single_rows, summarize, summarize_seg

ENTITY = "zoosumzoosum-kookmin-university"  # 본인 혼자인 기본 팀 (W&B는 개인 entity 없이 기본 팀을 줌, 2026-10-12 확인)
PROJECT = "short2long"
METRICS = ("R@1", "R@5", "V@1", "V@5", "err_mean", "err_median", "verr_median")
SEG_METRICS = ("SegR", "seg_wrong", "seg_missing", "seg_extra", "fooled_rate")  # 구간 채점(멀티모달 평가셋 v2)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_config(pairs, json_str):
    """key=value 목록과 JSON 문자열을 합친다. 값은 JSON으로 읽어 보고 안 되면 문자열로 둔다."""
    cfg = json.loads(json_str) if json_str else {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        if not _:
            raise SystemExit(f"--config는 key=value 형식이어야 함: {p}")
        try:
            cfg[k] = json.loads(v)
        except json.JSONDecodeError:
            cfg[k] = v
    return cfg


def percentile(values, p):
    """가장 가까운 순위 방식 백분위(소규모 표본용, numpy 불필요)."""
    s = sorted(values)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


def build_summary(rows, results):
    """wandb.summary에 넣을 평평한 dict. None(오차 없음)은 뺀다."""
    out = {"n": len(rows), "missing": sum(r["missing"] for r in rows)}

    def add(suffix, s):
        for k in METRICS:
            if s[k] is not None:
                out[k + suffix] = s[k]

    singles = single_rows(rows)  # R@k/V@k는 정답이 하나인 질의만 (compilation 제외)
    add("", summarize(singles))
    for t, s in group_by(singles, "transform").items():
        add(f"/{t}", s)
    for f, s in group_by(singles, "format").items():
        add(f"/format={f}", s)
    if any(r["kind"] for r in rows):  # 멀티모달 평가셋일 때만 SegR·속은 비율을 올린다
        def add_seg(suffix, s):
            for k in SEG_METRICS:
                if s[k] is not None:
                    out[k + suffix] = s[k]
        add_seg("", summarize_seg(rows))
        for kind, s in group_by(rows, "kind", summarize_seg).items():
            add_seg(f"/kind={kind}", s)
    lat = [r["latency_ms"] for r in results if r.get("latency_ms") is not None]
    if lat:
        out["latency_p50_ms"] = percentile(lat, 50)
        out["latency_p90_ms"] = percentile(lat, 90)
        out["latency_max_ms"] = max(lat)
    return out


def query_table_rows(manifest, results, rows):
    by_id = {r["query_id"]: r for r in results}
    out = []
    for m, row in zip(manifest, rows):
        res = by_id.get(m["query_id"])
        # segments만 낸 결과도 있으므로 후보가 없으면 점수 최고 구간을 1위로 본다
        cands = (res.get("candidates") or [dict(c, start_sec=c["start"]) for c in res.get("segments") or []]) if res else []
        top = max(cands, key=lambda c: c["score"]) if cands else None
        out.append([m["query_id"], m["transform"], m["format"], m["source_video_id"], m["start_sec"],
                    top["video_id"] if top else None, top["start_sec"] if top else None,
                    top["score"] if top else None, row["rank"] == 1,
                    res.get("latency_ms") if res else None])
    return out


def main():
    ap = argparse.ArgumentParser(description="평가 결과를 W&B에 올리기")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--group")
    ap.add_argument("--config", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--config-json", help="JSON 문자열 또는 .json 파일 경로")
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--tol", type=float, default=3.0)
    ap.add_argument("--entity", default=ENTITY)
    ap.add_argument("--project", default=PROJECT)
    args = ap.parse_args()

    import wandb  # eval 그룹에만 설치돼 있어 여기서 import

    cj = args.config_json
    if cj and not cj.lstrip().startswith("{"):
        with open(cj, encoding="utf-8") as f:
            cj = f.read()
    config = parse_config(args.config, cj)
    config.update(manifest=args.manifest, manifest_sha256=sha256_file(args.manifest), tol=args.tol)

    manifest, results = read_jsonl(args.manifest), read_jsonl(args.results)
    rows, unknown = score_queries(manifest, results, args.tol)
    summary = build_summary(rows, results)

    run = wandb.init(entity=args.entity, project=args.project, name=args.name, group=args.group,
                     tags=args.tags, config=config)
    run.summary.update(summary)
    run.log({
        "per_query": wandb.Table(
            columns=["query_id", "transform", "format", "truth_video", "truth_start", "top1_video",
                     "top1_start", "top1_score", "correct", "latency_ms"],
            data=query_table_rows(manifest, results, rows)),
        "summary_table": wandb.Table(
            columns=["group", "name", "n"] + list(METRICS),
            data=[["all", "all", s["n"]] + [s[k] for k in METRICS] for s in [summarize(rows)]]
            + [[kind, name, s["n"]] + [s[k] for k in METRICS]
               for kind, key in (("transform", "transform"), ("format", "format"))
               for name, s in group_by(rows, key).items()]),
    })
    print(f"R@1={summary['R@1']:.3f} R@5={summary['R@5']:.3f} 결과없음={summary['missing']} "
          f"manifest에 없는 id={len(unknown)}")
    print("run:", run.url)
    run.finish()


if __name__ == "__main__":
    main()
