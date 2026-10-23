"""소리 핑거프린트 (랜드마크 해시, Shazam 방식). numpy/scipy만 쓴다.

흐름
  1) 소리를 8kHz 모노로 읽는다                       decode_audio
  2) 짧은 구간(64ms)마다 주파수별 세기를 구한다        spectrogram   (시간 x 주파수 표)
  3) 주변보다 센 점(피크)만 남긴다                    find_peaks    (사진의 별자리처럼)
  4) 피크 두 개를 짝지어 해시 하나로 만든다            make_hashes   (주파수1, 주파수2, 시간차)
  5) 색인: 해시를 DB에 (hash, 영상, 시각)으로 저장     index_video
  6) 질의: 질의 해시와 같은 해시를 DB에서 찾아 "영상 + (원본시각 - 질의시각)"에 투표.
     진짜 원본이면 투표가 한 오프셋에 몰린다.            query_audio

왜 이게 변형에 강한가: 피크는 '주변보다 센 점'이라 배경음이 깔려도 큰 소리의 피크는 남고,
해시는 피크 둘의 '상대 위치'라 질의가 원본 어디서 시작하든 같은 값이 나온다.
"""
import subprocess
from dataclasses import dataclass

import numpy as np
import psycopg
from scipy.ndimage import maximum_filter
from scipy.signal import resample_poly


@dataclass(frozen=True)
class FpParams:
    sr: int = 8000          # 샘플레이트. 사람 목소리·악기 대부분이 4kHz 아래라 8kHz면 충분 (용량 절약)
    n_fft: int = 512        # 창 길이 64ms. 주파수 칸 폭 = 8000/512 = 15.6Hz
    hop: int = 256          # 창을 32ms씩 민다 -> 시각 해상도 32ms (허용 오차 3초에 비해 충분)
    peak_t: int = 11        # 피크 = 시간 11프레임(±0.16초) x 주파수 `peak_f`칸 안에서 가장 센 점
    peak_f: int = 25        #   ... 주파수 ±195Hz
    peaks_per_sec: int = 20  # 초당 최대 피크 수 (센 순). 해시 수(=DB 용량)와 잡음 내성의 균형
    fan_out: int = 6        # 피크 하나(anchor)를 뒤따르는 피크 몇 개와 짝지을지
    dt_max: int = 63        # 짝 사이 최대 시간차 (프레임, 약 2초). 6비트에 들어가는 값
    floor_db: float = 60.0  # 전체 최고(99퍼센타일)보다 60dB 넘게 작은 점은 피크로 안 침 (무음 구간 잡음 방지)
    abs_floor_db: float = -90.0  # 절대 하한: 디지털 무음(-120dB 평평한 판)은 어디서나 '동점 최고점'이 되어 가짜 피크가 쏟아진다
    q: int = 4              # 프레임 하나를 q등분한 '미세 칸'. 시각(t)은 미세 칸 단위로 저장한다 (아래 '격자 어긋남' 참고)
    index_shifts: int = 4   # 색인할 때 소리를 hop/q씩 밀어 이만큼 다시 뽑는다 (해시 수가 이 배수로 늘어남)
    query_shifts: int = 1   # 질의도 마찬가지 (질의 지연이 이 배수로 늘어남)


DEFAULT = FpParams()


def decode_audio(path, sr=DEFAULT.sr):
    """ffmpeg로 소리를 모노 float32 배열로 읽는다. 소리가 없으면 빈 배열."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg 실패({path}): {p.stderr.decode(errors='replace').strip()[-200:]}")
    return np.frombuffer(p.stdout, dtype=np.float32)


def spectrogram(x, p=DEFAULT):
    """(프레임 수, 주파수 칸) 크기의 세기 표(dB). 칸 0~1(직류·초저음)은 버린다."""
    if len(x) < p.n_fft:
        return np.zeros((0, 0), dtype=np.float32)
    n_frames = 1 + (len(x) - p.n_fft) // p.hop
    # 복사 없이 겹치는 창을 쌓아 보는 numpy 기법 (as_strided와 같은 효과)
    frames = np.lib.stride_tricks.sliding_window_view(x, p.n_fft)[:: p.hop][:n_frames]
    window = np.hanning(p.n_fft).astype(np.float32)
    mag = np.abs(np.fft.rfft(frames * window, axis=1))
    db = 20.0 * np.log10(mag + 1e-6)
    return db[:, 2:2 + 255].astype(np.float32)  # 주파수 칸 255개 = 8비트에 들어감 (약 31Hz ~ 4kHz)


def find_peaks(S, p=DEFAULT):
    """(t, f) 피크 좌표 배열 두 개, 시간순. 초당 `peaks_per_sec`개까지만 센 순으로 남긴다."""
    if S.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    local_max = S == maximum_filter(S, size=(p.peak_t, p.peak_f), mode="nearest")
    local_max &= S > np.percentile(S[:: max(1, len(S) // 2000)], 99) - p.floor_db  # 표본으로 99퍼센타일 근사
    local_max &= S > p.abs_floor_db
    t, f = np.nonzero(local_max)
    val = S[t, f]
    # 1초(31프레임) 묶음마다 센 순으로 상위 N개만: 소리가 큰 구간이 해시를 독차지하지 못하게
    win = t // (p.sr // p.hop)
    order = np.lexsort((-val, win))
    win_sorted = win[order]
    start_of_group = np.r_[0, np.flatnonzero(np.diff(win_sorted)) + 1]
    group_idx = np.cumsum(np.r_[0, np.diff(win_sorted) != 0])
    rank = np.arange(len(order)) - start_of_group[group_idx]
    keep = order[rank < p.peaks_per_sec]
    t, f = t[keep], f[keep]
    order = np.lexsort((f, t))  # 시간순 (같은 시각이면 주파수순)
    return t[order], f[order]


def make_hashes(t, f, p=DEFAULT):
    """피크 짝 -> (hash, t) 두 배열. hash = f1<<14 | f2<<6 | dt  (22비트).
    anchor 뒤따르는 k번째 피크(k=1..fan_out)와 짝짓고, 시간차가 1~dt_max 프레임인 것만 쓴다."""
    hs, ts = [], []
    for k in range(1, p.fan_out + 1):
        if len(t) <= k:
            break
        dt = t[k:] - t[:-k]
        ok = (dt >= 1) & (dt <= p.dt_max)
        f1, f2 = f[:-k][ok], f[k:][ok]
        hs.append((f1 << 14) | (f2 << 6) | dt[ok])
        ts.append(t[:-k][ok])
    if not hs:
        return np.zeros(0, np.int32), np.zeros(0, np.int32)
    return np.concatenate(hs).astype(np.int32), np.concatenate(ts).astype(np.int32)


def fingerprint(x, p=DEFAULT, n_shifts=1):
    """소리 배열 -> (hash 배열, t 배열). t는 미세 칸 단위(프레임 x q + 밀린 칸 번호).

    격자 어긋남: 프레임은 32ms 격자에 놓이는데, 질의가 원본 중간에서 시작하면 질의의 격자와 원본의
    격자가 반 칸쯤 어긋난다. 그러면 피크가 이웃 프레임으로 넘어가 두 피크의 시간차(dt)가 1씩 달라지고
    해시가 안 맞는다. 그래서 소리를 hop/q 샘플씩 밀어 격자를 q가지로 다시 뽑아 모두 저장(질의)하면
    어느 한 격자는 거의 맞는다. 색인 쪽(index_shifts)에서 하면 질의 지연은 그대로이고 DB만 커진다."""
    all_h, all_t = [], []
    for i in range(n_shifts):
        t, f = find_peaks(spectrogram(x[i * p.hop // p.q:], p), p)
        h, tt = make_hashes(t, f, p)
        all_h.append(h)
        all_t.append(tt * p.q + i)
    return np.concatenate(all_h), np.concatenate(all_t).astype(np.int32)


# ---------------------------------------------------------------- 색인 (DB)

def connect(url):
    """색인용 연결. autocommit이라 insert_hashes의 `conn.transaction()`이 진짜 트랜잭션(영상마다 커밋)이 된다.
    autocommit이 아니면 그 블록이 savepoint로만 동작해 색인 전체가 한 트랜잭션이 되고,
    TRUNCATE가 잡은 잠금을 끝까지 쥐어 다른 연결의 SELECT까지 막는다 (실제로 겪음)."""
    return psycopg.connect(url, autocommit=True)


def index_video(conn, video_id, x, p=DEFAULT):
    """영상 하나의 해시를 DB에 넣는다. 이미 색인된 영상이면 아무것도 안 하고 False."""
    hashes, ts = fingerprint(x, p, p.index_shifts)
    return insert_hashes(conn, video_id, len(x) / p.sr, hashes, ts)


def insert_hashes(conn, video_id, dur_sec, hashes, ts):
    """COPY로 한꺼번에 넣는다 (한 줄씩 INSERT보다 훨씬 빠름). 영상 하나가 한 트랜잭션이어야 하므로
    conn은 autocommit 연결(connect)이어야 한다. 이미 있는 영상이면 False."""
    with conn.transaction():
        row = conn.execute(
            "INSERT INTO audio_fp_src (video_id, dur_sec, n_hashes) VALUES (%s, %s, %s) "
            "ON CONFLICT (video_id) DO NOTHING RETURNING src_id",
            (video_id, dur_sec, len(hashes))).fetchone()
        if row is None:
            return False
        src_id = row[0]
        with conn.cursor().copy("COPY audio_fp (hash, src_id, t) FROM STDIN") as cp:
            for h, t in zip(hashes.tolist(), ts.tolist()):
                cp.write_row((h, src_id, t))
    return True


# ---------------------------------------------------------------- 질의

def resample_speed(x, factor):
    """factor배 속도(음정도 같이 변함)로 바뀐 질의를 원래 속도로 되돌린다.
    factor=1.1이면 소리가 1/1.1 길이로 줄어 있으니 길이를 1.1배로 늘린다."""
    if abs(factor - 1.0) < 1e-9:
        return x
    return resample_poly(x, round(factor * 1000), 1000).astype(np.float32)


# 오프셋 투표를 합치는 폭(미세 칸): 피크가 프레임 경계에서 하나씩 어긋나기도 하므로 이웃을 합쳐 센다
# (q=4면 ±2칸 = 한 프레임 남짓)
def smooth_width(p):
    return p.q // 2

VOTE_SQL = """
SELECT f.src_id, f.t - q.t AS off, count(*) AS c
FROM unnest(%s::int[], %s::int[]) AS q(h, t)
JOIN audio_fp f ON f.hash = q.h
GROUP BY f.src_id, f.t - q.t
HAVING count(*) >= 2
ORDER BY c DESC
LIMIT %s
"""


def vote(conn, hashes, ts, p=DEFAULT, top_rows=1500):
    """질의 해시 하나 묶음을 투표시켜 {src_id: (점수, 시작 프레임)}를 돌려준다.
    점수 = 이웃 오프셋(off±smooth) 투표 합의 최대값. 오프셋은 미세 칸 단위."""
    if len(hashes) == 0:
        return {}
    rows = conn.execute(VOTE_SQL, (hashes.tolist(), ts.tolist(), top_rows)).fetchall()
    w = smooth_width(p)
    by_src = {}
    for src_id, off, c in rows:
        by_src.setdefault(src_id, {})[off] = c
    best = {}
    for src_id, offs in by_src.items():
        # (점수, 오프셋): 오프셋 off 주변 합을 점수로, 가장 높은 off를 고른다
        score, off = max(
            (sum(offs.get(o + d, 0) for d in range(-w, w + 1)), o) for o in offs)
        best[src_id] = (score, off)
    return best


# 이 점수 이상이면 "진짜 일치"로 보고 남은 배속은 시도하지 않는다.
# 주의: 이 문턱(100), 배속 격자(±12%), 색인 4격자, 절대 하한 -90dB은 synth_v1 결과를 보며 골랐다.
# synth_v1에서는 우연히 맞은 최대 점수가 21, 진짜 일치가 784 이상이라 100이 안전해 보였다.
# 그러나 보류셋(synth_holdout_s2, 처음 보는 100개)에서는 소리가 부분만 겹치는 다른 영상이
# 695·808점을 받아 "우연 최대 21"이 깨졌다 -> 문턱 100은 "진짜 일치" 판별이 아니라 "속도 탐색을 멈춰도 될 만큼 강함"일 뿐이다.
# 보류셋 결과: R@1 0.94 (같은 소리의 중복 영상을 정답으로 쳐 주면 1.00).
# 한계: 배속 격자 밖(1.25배속) 질의는 R@1 0/20, 색인에 없는 질의는 부분 중복 때문에 "못 찾음"을 점수로 가르기 어렵다.
STOP_SCORE = 100


def query_audio(conn, x, p=DEFAULT, speeds=(1.0,), top_k=10, stop_score=STOP_SCORE):
    """질의 소리 x -> 후보 목록 [{video_id, start_sec, score, speed}] (점수 큰 순, 영상당 1개).
    speeds의 각 배속(앞에서부터)으로 되돌려 질의해 영상별 최고 점수를 취한다.
    확실한 일치(stop_score 이상)가 나오면 나머지 배속은 건너뛴다 -> 대부분의 질의는 1.0 한 번으로 끝."""
    best = {}  # src_id -> (score, off, speed)
    for sp in speeds:
        if best and max(b[0] for b in best.values()) >= stop_score:
            break
        hashes, ts = fingerprint(resample_speed(x, sp), p, p.query_shifts)
        for src_id, (score, off) in vote(conn, hashes, ts, p).items():
            if src_id not in best or score > best[src_id][0]:
                best[src_id] = (score, off, sp)
    top = sorted(best.items(), key=lambda kv: -kv[1][0])[:top_k]
    if not top:
        return []
    names = dict(conn.execute(
        "SELECT src_id, video_id FROM audio_fp_src WHERE src_id = ANY(%s)", ([s for s, _ in top],)).fetchall())
    sec_per_unit = p.hop / p.q / p.sr
    return [{"video_id": names[s], "start_sec": round(max(off, 0) * sec_per_unit, 3),
             "score": float(score), "speed": sp} for s, (score, off, sp) in top]
