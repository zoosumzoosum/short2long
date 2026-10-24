"""멀티모달 평가셋(소리+화면 결합용) 만들기. make_synth.py의 변형 함수를 가져다 쓰고, 그 파일은 건드리지 않는다
(건드리면 synth_v1·synth_vis 재현이 깨질 위험이 있어서).

질의 종류 (manifest의 kind):
  both        소리 변형(bgm/noise/speed 중 1) + 화면 변형(vcrop/zoom/overlay/mirror/frame 중 1)을 함께
  mute        화면 변형 + 소리 없음 (무음 AAC 트랙. 이유: 노트 참고)
  swap        화면은 원본 A(+화면 변형), 소리는 '다른 영상 B의 소리'로 통째 교체. 정답은 A, B는 audio_decoy
  compilation 서로 다른 원본 2~4개에서 3~8초씩 잘라 이어 붙인 15~30초 (segments에 구간별 정답)

보류셋: 영상 단위로 dev/holdout을 먼저 나눈 뒤 각자의 영상 풀에서만 뽑는다
(구간 원본·swap 디코이·배경음 영상 모두 같은 풀 안에서만 고른다 → 두 세트가 쓰는 영상 ID가 겹치지 않는다).

실행 (make_synth.py 독스트링과 같은 docker 방식, 질의 mp4 sha256은 manifest 각 줄과 sha256.txt):
    python -m eval.make_synth_mm --out data/eval/synth_mm_v1 --part dev --seed 1 \\
        --avoid data/eval/synth_v1 data/eval/synth_holdout_s2 data/eval/synth_vis_v1 ...
    python -m eval.make_synth_mm --out data/eval/synth_mm_holdout --part holdout --seed 2 --avoid (같은 목록)
dev/holdout 겹침 확인: python -m eval.make_synth_mm --compare data/eval/synth_mm_v1 data/eval/synth_mm_holdout
"""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

from eval import make_synth as ms

AUDIO_TFS = ["bgm", "noise", "speed"]
VIS_TFS = ["vcrop", "zoom", "overlay", "mirror", "frame"]   # combo는 자체 배경음이 있어 제외
SEG_VIS_TFS = ["vcrop", "frame"]   # 이어 붙이려면 모두 360x640이 되어야 한다: 가로 영상은 잘라 채우거나(vcrop) 띠를 두르거나(frame)
DEFAULT_COUNTS = {"both": 30, "mute": 20, "swap": 25, "compilation": 25}
# 단일 원본 클립의 format 비율 10칸 주기: long 5 / fan 3 / short 2 (기존 평가셋 50/35/15와 비슷, 긴 영상이 위치 찾기가 가장 어렵다)
FORMAT_CYCLE = ["long"] * 5 + ["fan"] * 3 + ["short"] * 2
SEG_LEN = (3.0, 8.0)
TOTAL_LEN = (15.0, 30.0)
N_SEG_WEIGHTS = {2: 2, 3: 5, 4: 3}   # 2개짜리는 길이 합 15초 조건 때문에 8초 근처만 가능해 드물게


# ---------- 1. 영상 단위 분할 ----------
def split_catalog(catalog, avoid_ids, split_seed=0):
    """영상을 dev/holdout으로 나눈다 (겹침 0). format별로, '기존 평가셋에 안 쓴 영상'을 먼저 번갈아 나눠
    양쪽에 새 영상이 비슷하게 들어가게 한다. split_seed가 고정이라 --seed를 바꿔도 분할은 같다."""
    rng = random.Random(split_seed)
    dev, hold = [], []
    for fmt in sorted({c["format"] for c in catalog}):
        pool = sorted((c for c in catalog if c["format"] == fmt), key=lambda c: c["video_id"])
        fresh = [c for c in pool if c["video_id"] not in avoid_ids]
        used = [c for c in pool if c["video_id"] in avoid_ids]
        rng.shuffle(fresh)
        rng.shuffle(used)
        for i, c in enumerate(fresh + used):
            (dev if i % 2 == 0 else hold).append(c)
    return dev, hold


# ---------- 2. 계획 (순수 함수: ffmpeg 없음) ----------
class Picker:
    """영상 풀에서 차례로 뽑는다. 기존 평가셋에 안 쓴 영상을 먼저, 다 쓰면 처음부터 다시."""

    def __init__(self, pool, avoid_ids, rng):
        def order(items):
            fresh = [c for c in items if c["video_id"] not in avoid_ids]
            used = [c for c in items if c["video_id"] in avoid_ids]
            rng.shuffle(fresh)
            rng.shuffle(used)
            return fresh + used
        self.orders = {fmt: order([c for c in pool if c["format"] == fmt])
                       for fmt in sorted({c["format"] for c in pool})}
        self.orders["*"] = order(pool)
        self.cursor = {k: 0 for k in self.orders}

    def take(self, fmt="*", min_dur=0.0, exclude=()):
        order = self.orders[fmt]
        for step in range(len(order)):
            i = (self.cursor[fmt] + step) % len(order)
            c = order[i]
            if c["duration"] >= min_dur and c["video_id"] not in exclude:
                self.cursor[fmt] = i + 1
                return c
        raise RuntimeError(f"조건에 맞는 영상이 없음: format={fmt} min_dur={min_dur}")


def plan_compilation(picker, rng):
    n = rng.choices(list(N_SEG_WEIGHTS), weights=list(N_SEG_WEIGHTS.values()))[0]
    for _ in range(1000):  # 길이 합이 15~30초가 될 때까지 다시 뽑는다
        lens = [round(rng.uniform(*SEG_LEN), 2) for _ in range(n)]
        if TOTAL_LEN[0] <= sum(lens) <= TOTAL_LEN[1]:
            break
    else:
        raise RuntimeError("compilation 길이를 못 정함")
    segs, used = [], set()
    for ln in lens:
        c = picker.take("*", min_dur=ln + 1.5, exclude=used)
        used.add(c["video_id"])
        segs.append({"video_id": c["video_id"], "format": c["format"], "len": ln,
                     "start": round(rng.uniform(0, c["duration"] - ln - 0.5), 2),
                     "vis_tf": rng.choice(SEG_VIS_TFS)})
    return segs


def plan_clips(pool, avoid_ids, seed, counts=None, min_len=10, max_len=30):
    """클립 계획 목록. 종류·format·변형은 주기로 돌려 균등하게 배정하고 마지막에 섞는다(번호로 종류가 안 보이게)."""
    counts = counts or DEFAULT_COUNTS
    rng = random.Random(seed)
    picker = Picker(pool, avoid_ids, rng)
    items = []
    for kind in ("both", "mute", "swap", "compilation"):
        for i in range(counts.get(kind, 0)):
            items.append({"kind": kind, "i": i})
    rng.shuffle(items)
    singles = 0
    for item in items:
        kind, i = item["kind"], item["i"]
        if kind == "compilation":
            item["segments"] = plan_compilation(picker, rng)
            continue
        fmt = FORMAT_CYCLE[singles % len(FORMAT_CYCLE)]  # 주기로 배정
        singles += 1
        c = picker.take(fmt, min_dur=min_len + 1.5)
        length = round(rng.uniform(min_len, min(max_len, c["duration"] - 1)), 2)
        item.update(video_id=c["video_id"], format=c["format"], stratum=c.get("stratum"),
                    length=length, start=round(rng.uniform(0, c["duration"] - length - 0.5), 2),
                    vis_tf=VIS_TFS[i % len(VIS_TFS)])
        if kind == "both":
            item["audio_tf"] = AUDIO_TFS[i % len(AUDIO_TFS)]
    for n, item in enumerate(items, start=1):
        item["query_id"] = f"q{n:03d}"
    return items


# ---------- 3. 렌더 ----------
def clip_rng(seed, qid):
    """클립마다 독립 난수: 하나를 다시 만들어도 다른 클립에 영향이 없다."""
    return random.Random(f"{seed}:{qid}")


def mux_replace_audio(video, audio_args, dur, out):
    """video의 영상은 그대로 두고 소리를 audio_args(입력 옵션)가 가리키는 것으로 바꾼다. -shortest로 길이를 영상에 맞춘다."""
    ms.run(["ffmpeg", "-y", "-hide_banner", "-i", str(video)] + audio_args
           + ["-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
              "-t", f"{dur:.3f}", "-shortest", "-movflags", "+faststart", str(out)])


def pick_decoy(src_id, length, rng, pool, media_dir):
    """swap에 쓸 B의 소리 구간: A와 다른 영상, 거의 무음이 아닌 구간 (같은 풀에서만)."""
    others = [c for c in pool if c["video_id"] != src_id and c["duration"] >= length + 1]
    for _ in range(20):
        b = rng.choice(others)
        b_start = round(rng.uniform(0, b["duration"] - length - 0.5), 2)
        if ms.segment_volume_db(media_dir / f"{b['video_id']}.mp4", b_start, length) > -50:
            return b, b_start
    raise RuntimeError("디코이로 쓸 만한 소리 구간을 못 찾음")


def render_single(item, out, rng, pool, media_dir, tmpdir):
    """both/mute/swap 하나. 화면 변형 → (소리 처리) 순서. 반환: (params, 추가 manifest 필드)."""
    src = media_dir / f"{item['video_id']}.mp4"
    length, kind = item["length"], item["kind"]
    # 중간 파일 이름을 원본 영상 ID로 해야 pick_bgm이 '자기 자신의 소리'를 배경음으로 고르지 않는다
    mid = tmpdir / f"{item['video_id']}.mp4"
    vis_params = ms.render(item["vis_tf"], src, item["start"], length, mid, rng, pool, media_dir, tmpdir)
    params = {"visual": vis_params}
    extra = {"vis_tf": item["vis_tf"]}
    if kind == "both":
        dur = ms.probe_duration(mid)
        params["audio"] = ms.render(item["audio_tf"], mid, 0.0, dur, out, rng, pool, media_dir, tmpdir)
        extra["audio_tf"] = item["audio_tf"]
    elif kind == "mute":
        dur = ms.probe_duration(mid)
        mux_replace_audio(mid, ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=44100:cl=mono"], dur, out)
        extra["audio_tf"] = "mute"
    elif kind == "swap":
        dur = ms.probe_duration(mid)
        b, b_start = pick_decoy(item["video_id"], dur, rng, pool, media_dir)
        mux_replace_audio(mid, ["-ss", f"{b_start:.3f}", "-t", f"{dur:.3f}", "-i",
                                str(media_dir / f"{b['video_id']}.mp4")], dur, out)
        extra["audio_tf"] = "swap"
        extra["audio_decoy"] = {"video_id": b["video_id"], "start_sec": b_start,
                                "end_sec": round(b_start + dur, 2)}
    mid.unlink()
    return params, extra


def render_compilation(item, out, rng, pool, media_dir, tmpdir):
    """구간마다 360x640으로 잘라 만든 뒤 concat 필터로 이어 붙인다 (소리는 구간 원래 소리, 모노 44.1kHz로 맞춤)."""
    parts, segments, t = [], [], 0.0
    for k, s in enumerate(item["segments"]):
        p = tmpdir / f"{item['query_id']}_seg{k}.mp4"
        sp = ms.render(s["vis_tf"], media_dir / f"{s['video_id']}.mp4", s["start"], s["len"], p, rng, pool, media_dir, tmpdir)
        parts.append(p)
        dur = ms.probe_duration(p)  # 계획한 길이가 아니라 만들어진 구간 파일의 실제 길이로 경계를 잡는다 (누적 오차 방지)
        segments.append({"q_start": round(t, 2), "q_end": round(t + dur, 2), "video_id": s["video_id"],
                         "start": s["start"], "vis_tf": s["vis_tf"], "params": sp})
        t += dur
    cmd = ["ffmpeg", "-y", "-hide_banner"]
    fc = ""
    for i, p in enumerate(parts):
        cmd += ["-i", str(p)]
        fc += f"[{i}:v]fps=30,setsar=1[v{i}];[{i}:a]{ms.mono_filter(p)},aresample=44100[a{i}];"
    fc += "".join(f"[v{i}][a{i}]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=1:a=1[v][a]"
    ms.run(cmd + ["-filter_complex", fc, "-map", "[v]", "-map", "[a]"] + ms.ENC + [str(out)])
    for p in parts:
        p.unlink()
    return {}, {"segments": segments, "audio_tf": "original", "vis_tf": "segment"}


# ---------- 4. 겹침 확인 ----------
def used_video_ids(manifest):
    """manifest 한 벌이 쓴 모든 영상 ID: 정답 원본, compilation 구간 원본, swap 디코이, 배경음 영상."""
    ids = set()

    def walk(x):  # params 안 어디에 있든 bgm_video_id를 모은다
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "bgm_video_id":
                    ids.add(v)
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    for m in manifest:
        ids.add(m["source_video_id"])
        ids.update(s["video_id"] for s in m.get("segments", []))
        if m.get("audio_decoy"):
            ids.add(m["audio_decoy"]["video_id"])
        walk(m.get("params", {}))
    return ids


def read_manifest(d):
    with open(Path(d) / "manifest.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser(description="멀티모달 평가셋 생성")
    ap.add_argument("--out")
    ap.add_argument("--media-dir", default="data/media")
    ap.add_argument("--part", choices=["dev", "holdout"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--split-seed", type=int, default=0, help="dev/holdout 분할 시드 (두 세트에 같은 값)")
    ap.add_argument("--avoid", nargs="*", default=[], help="가능하면 피할 기존 평가셋 폴더들 (겹치는 영상 수는 config.json에 기록)")
    ap.add_argument("--counts", nargs="*", default=[], metavar="KIND=N", help="종류별 개수 덮어쓰기(테스트용)")
    ap.add_argument("--no-wav", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar="DIR", help="두 평가셋이 쓴 영상 ID가 겹치는지 확인하고 끝낸다")
    args = ap.parse_args()

    if args.compare:
        a, b = (used_video_ids(read_manifest(d)) for d in args.compare)
        print(f"A {len(a)}편, B {len(b)}편, 겹침 {len(a & b)}편 {sorted(a & b)[:5]}")
        raise SystemExit(0 if not a & b else 1)
    if not args.out or not args.part:
        ap.error("--out과 --part가 필요")

    media_dir, out = Path(args.media_dir), Path(args.out)
    counts = dict(DEFAULT_COUNTS)
    for kv in args.counts:
        k, _, v = kv.partition("=")
        counts[k] = int(v)
    avoid = set()
    for d in args.avoid:
        avoid |= {m["source_video_id"] for m in read_manifest(d)}

    catalog = [c for c in ms.load_catalog(os.environ["DATABASE_URL"], media_dir, 10)
               if ms.has_audio(media_dir / f"{c['video_id']}.mp4")]   # 소리 있는 영상만 (소리 질의의 정답 후보)
    dev, hold = split_catalog(catalog, avoid, args.split_seed)
    pool = dev if args.part == "dev" else hold
    assert not {c["video_id"] for c in dev} & {c["video_id"] for c in hold}
    plan = plan_clips(pool, avoid, args.seed, counts)

    out.mkdir(parents=True, exist_ok=True)
    tmpdir = out / "_tmp"
    tmpdir.mkdir(exist_ok=True)
    manifest = []
    for n, item in enumerate(plan, start=1):
        qid, mp4, rng = item["query_id"], out / f"{item['query_id']}.mp4", clip_rng(args.seed, item["query_id"])
        if item["kind"] == "compilation":
            params, extra = render_compilation(item, mp4, rng, pool, media_dir, tmpdir)
            first = extra["segments"][0]
            row = {"source_video_id": first["video_id"], "start_sec": first["start"],   # 단일 답 채점기용 대표값(첫 구간)
                   "end_sec": round(first["start"] + item["segments"][0]["len"], 2), "format": "mixed", "stratum": None}
        else:
            params, extra = render_single(item, mp4, rng, pool, media_dir, tmpdir)
            row = {"source_video_id": item["video_id"], "start_sec": item["start"],
                   "end_sec": round(item["start"] + item["length"], 2),
                   "format": item["format"], "stratum": item["stratum"]}
        if not args.no_wav:
            ms.to_wav(mp4, out / f"{qid}.wav")
        manifest.append({"query_id": qid, "kind": item["kind"], **row,
                         "query_dur_sec": round(ms.probe_duration(mp4), 3),
                         "transform": item["kind"], "params": params, **extra,
                         "sha256": hashlib.sha256(mp4.read_bytes()).hexdigest(),
                         "seed": args.seed, "part": args.part})
        print(f"[{n}/{len(plan)}] {qid} {item['kind']}", flush=True)

    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(out / "sha256.txt", "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(f"{row['sha256']}  {row['query_id']}.mp4\n")
    used = used_video_ids(manifest)
    cfg = {"seed": args.seed, "split_seed": args.split_seed, "part": args.part, "counts": counts,
           "avoid": args.avoid, "pool_videos": len(pool), "used_videos": len(used),
           "used_overlap_with_avoid": len(used & avoid),
           "source_overlap_with_avoid": len({m["source_video_id"] for m in manifest} & avoid),
           "other_part_videos_used": len(used & {c["video_id"] for c in (hold if args.part == "dev" else dev)})}
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    for p in tmpdir.iterdir():
        p.unlink()
    tmpdir.rmdir()
    print("완료:", cfg)


if __name__ == "__main__":
    main()
