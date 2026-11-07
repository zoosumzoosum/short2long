"""소리+화면 결합 (4주 차, 후보 B: 순위 기반 결합 RRF + Viterbi 창 잇기 + 오프셋 가드).

흐름
  1) 질의를 겹치는 창으로 자른다                                   make_windows
  2) 창마다 소리·화면이 각자 후보 (영상, 오프셋, 점수) 목록을 낸다   audio_window_cands / vis_window_cands
     오프셋 = '질의 시각 0초가 원본의 몇 초인가' (창 위치와 상관없이 같은 뜻 -> 창끼리 비교 가능)
  3) 두 목록을 순위만 써서 합친다: 점수 = Σ 가중 / (k + 순위)     rrf_window
     같은 영상이고 오프셋이 비슷하면 같은 후보로 묶는다
  4) 창별 1등을 같은 영상·같은 오프셋끼리 이어 붙여 구간으로 만든다    merge_windows
  5) 구간 목록(+기존 candidates 형식)으로 내보낸다                   build_result

왜 점수가 아니라 순위인가: 소리 점수(투표 수)와 화면 점수(유사도 합)는 단위가 달라 더할 수 없다.
순위는 단위가 없다. 대신 '1등이지만 확신 없음'을 구분 못 한다.
('결과 없음' 규칙은 dev에서 효과가 없거나 해로워 껐고 코드에서 뺐다: DECISIONS 참고.)

이 파일의 결합 로직은 DB·GPU 없이 돈다 (테스트는 가짜 후보로). 창별 후보를 뽑는 함수만 DB를 쓴다.
"""
from collections import Counter
from dataclasses import dataclass

import numpy as np

from app import audio_fp


@dataclass(frozen=True)
class FusionParams:
    # 창
    win: float = 6.0          # 창 길이(초)
    hop: float = 3.0          # 창 시작 간격(초)
    # RRF
    k: float = 1.0            # 1/(k+순위): 클수록 1등과 2등의 차이가 작아진다
    w_audio: float = 1.0      # 방법별 가중. 같으면 소리 1등 vs 화면 1등이 동점 -> 아래 w로 결정
    w_vis: float = 2.0        # 소리 1등 영상 != 화면 1등 영상일 때 화면을 믿는 정도 (swap 방어). 순위만으로는 동점이라 사전 지식으로 정한다
    group_tol: float = 1e6    # 같은 영상이고 오프셋 차가 이 초 이내면 같은 후보. 1e6 = 영상 단위로 묶는다 (오프셋은 pick_offset이 고른다)
    # 구간 합치기
    merge_tol: float = 3.0    # 이웃 창 오프셋 차가 이 초 이내면 같은 구간
    switch: float = 1.0       # 창 사이 후보를 갈아탈 때의 벌점 (1/(k+1) 단위). 0이면 창마다 독립 1등, 크면 한 후보를 끈질기게 이어 간다
    bridge: bool = True       # 앞뒤 창이 같은 구간인데 가운데 한 창만 다르면(또는 없으면) 가운데를 흡수
    guard_tol: float = 3.0    # 소리 오프셋을 쓰는 조건: 그 창의 소리 1등이거나, 화면 오프셋과 이 초 이내 (소리 낮은 순위의 우연한 일치가 오프셋을 망치지 않게)
    min_windows: int = 1      # 이보다 적은 창으로 된 구간은 버린다


# ---------------------------------------------------------------- 창 만들기

def make_windows(dur, win, hop):
    """[(시작, 끝)] 초. 마지막 창은 끝이 dur에 닿게 당긴다. dur이 win보다 짧으면 창 하나."""
    if dur <= win:
        return [(0.0, float(dur))]
    out, s = [], 0.0
    while s + win < dur - 1e-9:
        out.append((s, s + win))
        s += hop
    out.append((dur - win, float(dur)))
    return out


def owned_ranges(windows):
    """창마다 '내가 책임지는 질의 시간'. 겹치는 창의 가운데 hop 만큼만 맡고, 처음 창은 시작까지, 마지막 창은 끝까지 맡는다.
    경계에 걸친 창(두 장면이 섞인)이 옆 구간까지 번지지 않게 한다."""
    n = len(windows)
    if n == 1:
        return list(windows)
    mids = [(windows[i][1] + windows[i + 1][0]) / 2 for i in range(n - 1)]  # 이웃 창의 겹침 한가운데
    out = []
    for i, (a, b) in enumerate(windows):
        lo = windows[0][0] if i == 0 else mids[i - 1]
        hi = windows[-1][1] if i == n - 1 else mids[i]
        out.append((lo, hi))
    return out


# ---------------------------------------------------------------- 창별 후보 뽑기 (DB 필요)

def sec_per_unit(p=audio_fp.DEFAULT):
    return p.hop / p.q / p.sr


def audio_window_cands(conn, hashes, ts, a, b, names, p=audio_fp.DEFAULT, top_k=10):
    """질의 전체의 해시(hashes, ts)에서 질의 시간 [a,b)초의 것만 골라 투표 -> [{video_id, off, score}] 점수 큰 순.
    ts는 미세 칸 단위이고 질의 시각 0초 기준이다. 오프셋도 '질의 0초가 원본 몇 초'로 나온다."""
    u = 1.0 / sec_per_unit(p)
    keep = (ts >= a * u) & (ts < b * u)
    best = audio_fp.vote(conn, hashes[keep], ts[keep], p)
    top = sorted(best.items(), key=lambda kv: -kv[1][0])[:top_k]
    return [{"video_id": names[s], "off": off * sec_per_unit(p), "score": float(sc)} for s, (sc, off) in top]


def vis_window_cands(nn, q_frame, q_time, a, b, vis_vote, qp, names, top_k=10):
    """화면: 질의 전체의 이웃 행 nn에서 시각 [a,b)초 프레임의 것만 골라 투표. vis_vote = visual_emb.vote."""
    q_frame, q_time = np.asarray(q_frame), np.asarray(q_time)
    in_win = (q_time[q_frame] >= a) & (q_time[q_frame] < b)
    if len(nn) == 0:
        return []
    sub = nn[in_win[nn[:, 0].astype(int)]]
    res = vis_vote(sub, q_frame, q_time, qp)
    top = sorted(res.items(), key=lambda kv: -kv[1][0])[:top_k]
    return [{"video_id": names[s], "off": float(st), "score": float(sc)} for s, (sc, st, _) in top]


# ---------------------------------------------------------------- RRF

def pick_offset(cl, p):
    """묶인 후보의 대표 오프셋. 소리가 더 정밀하지만, 소리 낮은 순위(2등 이하)가 우연히 같은 영상에 걸린 것은
    엉터리 오프셋일 수 있다 -> 소리 1등이거나 화면 오프셋과 guard_tol 이내일 때만 소리 것을 쓴다.
    (guard_tol 조건만 두면 화면 오프셋이 자주 틀리는 both에서 소리가 바로잡아 주던 것이 사라진다.)"""
    a, v = cl["offs"].get("audio"), cl["offs"].get("vis")
    if a is None:
        return v
    if v is None or cl["ranks"]["audio"] == 1 or abs(a - v) <= p.guard_tol:
        return a
    return v


def rrf_window(audio, vis, p=FusionParams()):
    """창 하나의 두 후보 목록 -> 합친 후보 [{video_id, off, score, audio_rank, vis_rank}] 점수 큰 순.
    같은 영상이고 오프셋 차 <= group_tol이면 한 후보. 대표 오프셋은 pick_offset.
    점수가 정확히 같으면(소리 1등 영상 != 화면 1등 영상, 가중 같음) 입력 순서대로 소리가 앞선다. 그래서 동점을 깨는 w_vis를 따로 조정한다."""
    items = []  # (기여 점수, 방법, 순위, 후보)
    for method, lst, w in (("audio", audio, p.w_audio), ("vis", vis, p.w_vis)):
        for r, c in enumerate(lst, start=1):
            items.append((w / (p.k + r), method, r, c))
    items.sort(key=lambda it: -it[0])  # 기여가 큰 것부터 클러스터를 만든다 (sort는 안정)
    clusters = []
    for contrib, method, rank, c in items:
        for cl in clusters:
            if cl["video_id"] == c["video_id"] and method not in cl["ranks"] and \
                    abs(next(iter(cl["offs"].values())) - c["off"]) <= p.group_tol:
                break
        else:
            cl = {"video_id": c["video_id"], "score": 0.0, "ranks": {}, "offs": {}}
            clusters.append(cl)
        cl["score"] += contrib
        cl["ranks"][method] = rank
        cl["offs"][method] = c["off"]
    for cl in clusters:
        cl["off"] = pick_offset(cl, p)
    clusters.sort(key=lambda cl: -cl["score"])
    return [{"video_id": cl["video_id"], "off": cl["off"], "score": cl["score"],
             "audio_rank": cl["ranks"].get("audio"), "vis_rank": cl["ranks"].get("vis")} for cl in clusters]


def fuse_window(raw_audio, raw_vis, p=FusionParams()):
    """한 창: RRF. (합친 목록, 소리가 자기 1등으로 본 영상 or None)"""
    return rrf_window(raw_audio, raw_vis, p), (raw_audio[0]["video_id"] if raw_audio else None)


# ---------------------------------------------------------------- 구간 합치기

def merge_windows(wins, p=FusionParams()):
    """wins: 시간순 [{own: (q_start, q_end), best: 합친 후보 1등 or None, audio_top: 영상 or None}].
    이웃한 창의 1등이 같은 영상이고 오프셋 차 <= merge_tol이면 한 구간. 구간 = {q_start, q_end, video_id, off, score, n}."""
    labels = [w["best"] for w in wins]
    if p.bridge:  # A X A 꼴에서 X(다른 후보나 없음) 한 창을 A로 본다: 잡음 한 창이 구간을 둘로 쪼개지 않게
        for i in range(1, len(labels) - 1):
            l, r = labels[i - 1], labels[i + 1]
            if l and r and same_seg(l, r, p) and not (labels[i] and same_seg(labels[i], l, p)):
                labels[i] = l
    segs = []
    for w, c in zip(wins, labels):
        if c is None:
            segs.append(None)  # 구간 끊김
            continue
        cur = segs[-1] if segs else None
        if cur and cur["video_id"] == c["video_id"] and abs(cur["off"] - c["off"]) <= p.merge_tol:
            cur["q_end"] = w["own"][1]
            cur["offs"].append(c["off"])
            cur["scores"].append(c["score"])
            cur["audio_tops"].append(w["audio_top"])
            cur["off"] = float(np.median(cur["offs"]))
        else:
            segs.append({"q_start": w["own"][0], "q_end": w["own"][1], "video_id": c["video_id"], "off": c["off"],
                         "offs": [c["off"]], "scores": [c["score"]], "audio_tops": [w["audio_top"]]})
    out = []
    for s in segs:
        if s and len(s["offs"]) >= p.min_windows:
            out.append(s)
    return out


def pick_path(cands, p):
    """창마다 후보 목록(합친 점수 순) -> 창마다 고른 후보(또는 None) 목록. 동적 계획법(Viterbi).
    경로 점수 = 고른 후보들의 합친 점수 합 - 벌점. 이웃 창에서 다른 영상·다른 오프셋으로 갈아타면 벌점 switch/(k+1),
    '없음'(None)을 끼우거나 빠져나오는 데 그 절반. 창 하나만 튀는 잡음은 벌점이 이득보다 커서 이웃을 따라가고,
    여러 창 이어서 바뀌는 진짜 장면 전환(짜깁기)은 이득이 벌점을 넘어 따라간다."""
    lam = p.switch / (p.k + 1)

    def cost(a, b):
        if a is None and b is None:
            return 0.0
        if a is None or b is None:
            return lam / 2
        return 0.0 if same_seg(a, b, p) else lam

    nodes = [[None] + list(c) for c in cands]
    best = [[0.0 if n is None else n["score"] for n in nodes[0]]]
    back = [[-1] * len(nodes[0])]
    for i in range(1, len(nodes)):
        row, brow = [], []
        for n in nodes[i]:
            gain = 0.0 if n is None else n["score"]
            opts = [best[i - 1][j] - cost(m, n) for j, m in enumerate(nodes[i - 1])]
            j = max(range(len(opts)), key=opts.__getitem__)
            row.append(opts[j] + gain)
            brow.append(j)
        best.append(row)
        back.append(brow)
    j = max(range(len(best[-1])), key=best[-1].__getitem__)
    path = []
    for i in range(len(nodes) - 1, -1, -1):
        path.append(nodes[i][j])
        j = back[i][j]
    return path[::-1]


def same_seg(c1, c2, p):
    return c1["video_id"] == c2["video_id"] and abs(c1["off"] - c2["off"]) <= p.merge_tol


def to_segment(s):
    """내부 구간 -> 채점기 형식 {q_start, q_end, video_id, start, score} (+ audio_source).
    start = 구간이 원본에서 시작하는 시각 = 오프셋 + q_start. 점수 = 창 점수 합(오래 이어질수록 근거가 많다).
    audio_source: 소리가 이 구간의 영상과 다른 영상을 가리켰으면(swap) 그 영상 ID."""
    tops = Counter(t for t in s["audio_tops"] if t)
    src, n_src = tops.most_common(1)[0] if tops else (None, 0)
    if n_src * 2 <= len(s["audio_tops"]):  # 창의 절반 넘게 같은 영상을 가리킬 때만 (경계에 걸친 한두 창의 이웃 구간 소리는 무시)
        src = None
    out = {"q_start": round(s["q_start"], 3), "q_end": round(s["q_end"], 3), "video_id": s["video_id"],
           "start": round(max(s["off"] + s["q_start"], 0.0), 3), "score": round(sum(s["scores"]), 4)}
    if src and src != s["video_id"]:
        out["audio_source"] = src
    return out


# ---------------------------------------------------------------- 후보 목록 (기존 R@k 형식)

def window_candidates(per_window_fused, p=FusionParams(), top_k=10):
    """모든 창의 합친 후보를 (영상, 오프셋)별로 모아 점수를 더한다 -> [{video_id, start_sec, score}] 점수 큰 순.
    오프셋은 merge_tol 이내면 같은 후보 (점수 가중 평균)."""
    pool = []  # 영상별 [{off(가중평균), w, score}]
    by_video = {}
    for fused in per_window_fused:
        for c in fused:
            lst = by_video.setdefault(c["video_id"], [])
            for e in lst:
                if abs(e["off"] - c["off"]) <= p.merge_tol:
                    tot = e["score"] + c["score"]
                    e["off"] = (e["off"] * e["score"] + c["off"] * c["score"]) / tot
                    e["score"] = tot
                    break
            else:
                lst.append({"off": c["off"], "score": c["score"]})
    for v, lst in by_video.items():
        for e in lst:
            pool.append({"video_id": v, "start_sec": round(max(e["off"], 0.0), 3), "score": round(e["score"], 4)})
    pool.sort(key=lambda c: -c["score"])
    return pool[:top_k]


def build_result(raw_windows, p=FusionParams()):
    """raw_windows: [{a, b, audio: [...], vis: [...]}] (창별 방법 후보) -> {"segments": [...], "candidates": [...]}.
    창 길이·간격은 raw_windows를 만들 때 이미 정해졌다."""
    ranges = owned_ranges([(w["a"], w["b"]) for w in raw_windows])
    wins, fused_all = [], []
    for w, own in zip(raw_windows, ranges):
        fused, atop = fuse_window(w["audio"], w["vis"], p)
        fused_all.append(fused)
        wins.append({"own": own, "best": fused[0] if fused else None, "audio_top": atop, "keep": fused})
    if p.switch > 0 and wins:
        for w, c in zip(wins, pick_path([w["keep"] for w in wins], p)):
            w["best"] = c
    segs = [to_segment(s) for s in merge_windows(wins, p)]
    return {"segments": segs, "candidates": window_candidates(fused_all, p)}
