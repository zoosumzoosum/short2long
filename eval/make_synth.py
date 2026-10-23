"""합성 클립 평가셋 만들기.

색인 대상 영상에서 10~30초 구간을 무작위로 잘라 변형(소리 위주)한 '질의 클립'을 만든다.
정답은 (원본 영상 ID, 자른 시작 시각)이라 사람이 라벨링할 필요가 없다.
같은 시드면 같은 평가셋이 나온다.

실행 (worker 이미지 안: ffmpeg와 psycopg가 있다. 저장소 루트에서, DB 주소는 <db>에 맞춘다):
    docker run --rm --user 1000:1000 -e PYTHONDONTWRITEBYTECODE=1 \\
      --network short2long_default --env-file .env \\
      -v $PWD/eval:/code/eval -v $PWD/data/media:/code/data/media:ro \\
      -v $PWD/data/eval:/code/data/eval short2long-worker \\
      sh -c 'DATABASE_URL=postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@<db>:5432/$POSTGRES_DB \\
             python -m eval.make_synth --out data/eval/synth_v1 --seed 1'
(--user를 주지 않으면 결과 파일이 root 소유가 된다.)
"""
import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import psycopg

# 소리 위주(이번 주) 변형 + 화면 변형(4주 차용). 기본은 소리 변형 4개 + crop.
TRANSFORMS = ["none", "bgm", "noise", "speed", "crop"]
BGM_REL_DB = (10, 15)  # 배경음이 질의 원본 구간보다 이만큼(dB) 작게


def run(cmd, stream="stderr"):
    """ffmpeg/ffprobe 실행. 실패하면 마지막 줄을 담아 예외를 낸다.
    ffmpeg 로그는 stderr, ffprobe 결과는 stdout으로 나온다."""
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} 실패: {p.stderr.strip().splitlines()[-1:]}")
    return p.stdout if stream == "stdout" else p.stderr


def has_audio(path):
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=index", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    return p.returncode == 0 and p.stdout.strip() != ""


def mono_filter(path):
    """소리를 모노로 내리는 필터 조각. 스테레오는 (왼+오)/2로 직접 내린다."""
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=channels", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    if p.stdout.strip() == "1":
        return "anull"
    return "pan=mono|c0=0.5*c0+0.5*c1"


def probe_duration(path):
    """파일 길이(초)를 ffprobe로 잰다. DB의 duration_sec은 쓰지 않는다(0으로 적힌 영상이 있었다)."""
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "csv=p=0", str(path)], stream="stdout")
    return float(out.strip())


def load_catalog(database_url, media_dir, min_len):
    """DB에서 (영상 ID, format, stratum)을 읽고(SELECT만), 길이는 파일에서 잰다."""
    with psycopg.connect(database_url) as conn:
        rows = conn.execute(
            "SELECT v.video_id, s.format, s.stratum "
            "FROM selected_videos s JOIN videos v USING (video_id) "
            "ORDER BY v.video_id").fetchall()  # 순서를 고정해야 같은 시드가 같은 결과를 낸다
    catalog = []
    for vid, fmt, stratum in rows:
        path = media_dir / f"{vid}.mp4"
        if not path.exists():  # 파일이 남아 있는 것만
            continue
        dur = probe_duration(path)
        if dur >= min_len + 1.5:  # 최소 길이 + 앞뒤 여유가 안 되면 후보에서 뺀다
            catalog.append({"video_id": vid, "duration": dur,
                            "format": fmt, "stratum": stratum})
    return catalog


def segment_volume_db(path, start, length):
    """원본 구간(모노로 내린 소리)의 평균 세기(dB). SNR·배경음 크기를 맞추려고 잰다.
    못 읽으면 조용히 기본값을 쓰지 않고 에러를 낸다."""
    log = run(["ffmpeg", "-hide_banner"] + cut_args(path, start, length)
              + ["-vn", "-af", f"{mono_filter(path)},volumedetect", "-f", "null", "-"])
    m = re.search(r"mean_volume:\s*(-?[\d.]+) dB", log)
    if not m:
        raise RuntimeError(f"mean_volume을 못 읽음: {path} {start}s+{length}s")
    return float(m.group(1))


def cut_args(src, start, length):
    # -ss를 -i 앞에 두면 빠르고, 다시 인코딩하므로 시각도 정확하다
    return ["-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(src)]


ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
       "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart"]


def render(transform, src, start, length, out, rng, all_sources, media_dir, tmpdir):
    """변형 하나를 적용해 out(mp4)을 쓰고, 정해진 변형 인자(params)를 돌려준다."""
    base = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "info"]

    if transform == "none":
        run(base + cut_args(src, start, length) + ENC + [str(out)])
        return {}

    if transform == "bgm":
        # (섞기 전에 모노로 직접 내린다: ffmpeg 자동 다운믹스는 리샘플과 겹치면 +3dB 커져 SNR·배경음 크기가 틀어졌다)
        # 다른 영상의 소리를 깔되, 세기는 '질의 원본 구간보다 10~15dB 작게' 맞춘다:
        # 두 구간의 평균 세기를 재서 gain = (원본 − 배경) − 목표차 로 계산
        others = [c for c in all_sources
                  if c["video_id"] != src.stem and c["duration"] >= length + 1]
        src_db = segment_volume_db(src, start, length)
        for _ in range(20):  # 거의 무음인 배경 구간이면 gain이 터무니없이 커지므로 다시 뽑는다
            other = rng.choice(others)
            bgm_src = media_dir / f"{other['video_id']}.mp4"
            o_start = round(rng.uniform(0, other["duration"] - length - 0.5), 3)
            bgm_db = segment_volume_db(bgm_src, o_start, length)
            if bgm_db > -50:
                break
        else:
            raise RuntimeError("배경으로 쓸 만한 구간을 못 찾음")
        target = round(rng.uniform(*BGM_REL_DB), 1)
        gain = round(src_db - bgm_db - target, 1)
        rel_db = round(src_db - (bgm_db + gain), 2)  # 실측 세기 기준 실제 차이 (목표차와 반올림 오차만큼 다름)
        cmd = (base + cut_args(src, start, length)
               + cut_args(bgm_src, o_start, length)
               + ["-filter_complex",
                  f"[0:a]{mono_filter(src)}[s];[1:a]{mono_filter(bgm_src)},volume={gain}dB[b];"
                  "[s][b]amix=inputs=2:duration=first:normalize=0[a]",
                  "-map", "0:v", "-map", "[a]"] + ENC + [str(out)])
        run(cmd)
        return {"bgm_video_id": other["video_id"], "bgm_start_sec": o_start,
                "bgm_gain_db": gain, "target_rel_db": target, "rel_db": rel_db,
                "src_mean_db": src_db, "bgm_mean_db": bgm_db}

    if transform == "noise":
        snr = 10
        sig_db = segment_volume_db(src, start, length)
        # 균등분포 잡음의 RMS = 진폭/√3 이므로 목표 RMS에 √3을 곱해 진폭을 정한다
        amp = min(1.0, 10 ** ((sig_db - snr) / 20) * math.sqrt(3))
        cmd = (base + cut_args(src, start, length)
               + ["-f", "lavfi", "-t", f"{length:.3f}", "-i",
                  f"anoisesrc=color=white:amplitude={amp:.5f}:sample_rate=44100:seed={rng.randrange(10**6)}",
                  "-filter_complex",
                  f"[0:a]{mono_filter(src)},aresample=44100[s];[s][1:a]amix=inputs=2:duration=first:normalize=0[a]",
                  "-map", "0:v", "-map", "[a]"] + ENC + [str(out)])
        run(cmd)
        return {"snr_db": snr, "signal_mean_db": sig_db, "noise_amplitude": round(amp, 5)}

    if transform == "speed":
        # asetrate: 재생 속도를 바꾸면 음정도 같이 바뀐다 (atempo는 음정 유지라 쇼츠와 다름)
        factor = round(rng.choice([-1, 1]) * rng.uniform(0.07, 0.10) + 1, 3)
        sr = 44100
        cmd = (base + cut_args(src, start, length)
               + ["-filter_complex",
                  f"[0:v]setpts=PTS/{factor}[v];"
                  f"[0:a]aresample={sr},asetrate={sr * factor:.1f},aresample={sr}[a]",
                  "-map", "[v]", "-map", "[a]"] + ENC + [str(out)])
        run(cmd)
        return {"factor": factor}

    if transform == "crop":
        # 가로 영상의 가운데 부근을 9:16으로 자르고 확대(zoom), 360x640으로 맞춘다. 소리는 그대로.
        zoom = round(rng.uniform(1.0, 1.3), 2)
        fx = round(rng.uniform(0.2, 0.8), 2)  # 가로 위치 (0=왼쪽 끝, 1=오른쪽 끝)
        fy = round(rng.uniform(0.2, 0.8), 2)
        # 이미 세로인 영상이면 폭 전체를 쓰고, 가로면 높이*9/16 폭을 쓴다
        w = f"min(iw\\,ih*9/16)/{zoom}"
        h = f"min(iw\\,ih*9/16)/{zoom}*16/9"
        vf = f"crop=w={w}:h={h}:x=(iw-out_w)*{fx}:y=(ih-out_h)*{fy},scale=360:640,setsar=1"
        run(base + cut_args(src, start, length) + ["-vf", vf] + ENC + [str(out)])
        return {"zoom": zoom, "x_frac": fx, "y_frac": fy, "out_size": "360x640"}

    raise ValueError(transform)


def to_wav(mp4, wav):
    """소리만 16kHz 모노 wav로 (소리 매칭 후보가 바로 읽도록)."""
    run(["ffmpeg", "-y", "-hide_banner", "-i", str(mp4), "-vn", "-ac", "1",
         "-ar", "16000", str(wav)])


def pick_sources(catalog, counts, rng, media_dir):
    """format별 개수만큼 원본을 뽑는다. 소리가 없는 영상은 건너뛴다. 가능하면 겹치지 않게."""
    chosen = []
    for fmt, n in counts.items():
        pool = [c for c in catalog if c["format"] == fmt]
        rng.shuffle(pool)
        picked = []
        for c in pool:
            if len(picked) == n:
                break
            if has_audio(media_dir / f"{c['video_id']}.mp4"):
                picked.append(c)
        if len(picked) < n:
            print(f"경고: {fmt} 후보 {len(picked)}개뿐 (요청 {n}개). 모자란 만큼 중복 사용", file=sys.stderr)
            while picked and len(picked) < n:
                picked.append(rng.choice(picked))
        chosen += picked
    rng.shuffle(chosen)
    return chosen




def main():
    ap = argparse.ArgumentParser(description="합성 클립 평가셋 생성")
    ap.add_argument("--out", default="data/eval/synth_v1")
    ap.add_argument("--media-dir", default="data/media")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-long", type=int, default=50)
    ap.add_argument("--n-fan", type=int, default=35)
    ap.add_argument("--n-short", type=int, default=15)
    ap.add_argument("--transforms", nargs="+", default=TRANSFORMS, choices=TRANSFORMS,
                    help="클립마다 하나씩 돌아가며 배정 (기본: 전부)")
    ap.add_argument("--min-len", type=float, default=10)
    ap.add_argument("--max-len", type=float, default=30)
    ap.add_argument("--no-wav", action="store_true", help="16kHz 모노 wav를 만들지 않음")
    args = ap.parse_args()

    media_dir = Path(args.media_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tmpdir = out / "_tmp"
    tmpdir.mkdir(exist_ok=True)
    rng = random.Random(args.seed)

    catalog = load_catalog(os.environ["DATABASE_URL"], media_dir, args.min_len)
    counts = {"long": args.n_long, "fan": args.n_fan, "short": args.n_short}
    sources = pick_sources(catalog, counts, rng, media_dir)

    manifest = []
    seen = {}  # format별로 따로 번호를 세어 변형을 돌려 배정 → 변형마다 format이 같은 비율로 들어감
    for i, src_info in enumerate(sources):
        k = seen[src_info["format"]] = seen.get(src_info["format"], -1) + 1
        transform = args.transforms[k % len(args.transforms)]
        qid = f"q{i + 1:03d}"  # 이름에 변형·정답이 드러나지 않게 번호만
        length = round(rng.uniform(args.min_len, min(args.max_len, src_info["duration"] - 1)), 2)
        start = round(rng.uniform(0, src_info["duration"] - length - 0.5), 2)
        src = media_dir / f"{src_info['video_id']}.mp4"
        mp4 = out / f"{qid}.mp4"
        params = render(transform, src, start, length, mp4, rng, catalog, media_dir, tmpdir)
        if not args.no_wav:
            to_wav(mp4, out / f"{qid}.wav")
        manifest.append({
            "query_id": qid, "source_video_id": src_info["video_id"],
            "start_sec": start, "end_sec": round(start + length, 2),
            "query_dur_sec": round(probe_duration(mp4), 3),  # 질의 파일 실제 길이 (speed는 원본 구간과 다름)
            "transform": transform, "params": params,
            "format": src_info["format"], "stratum": src_info["stratum"],
        })
        print(f"[{i + 1}/{len(sources)}] {qid} {transform} {src_info['video_id']} {start}s+{length}s", flush=True)

    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    for p in tmpdir.iterdir():
        p.unlink()
    tmpdir.rmdir()
    print(f"완료: {len(manifest)}개 → {out}/manifest.jsonl")


if __name__ == "__main__":
    main()
