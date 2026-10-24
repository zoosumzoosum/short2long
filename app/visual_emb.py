"""화면 매칭: 사전학습 이미지 임베딩 + pgvector (4주 차, 후보 B).

흐름
  1) 영상에서 1초에 1장씩 프레임을 뽑는다                          iter_frames
  2) 프레임 하나를 '보는 방식(view)' 여러 개로 잘라 정사각형으로 만든다  make_views
  3) 이미지 모델(DINOv2 등)이 정사각형 하나를 벡터 하나로 바꾼다        Embedder
  4) 색인: (영상, 시각, view, 벡터)를 DB(pgvector halfvec + HNSW)에 저장   index_rows
  5) 질의: 질의 프레임마다 가장 닮은 색인 프레임 top-k를 찾고,
     "영상 + (원본시각 - 질의시각)"에 투표한다. 진짜 원본이면 투표가 한 오프셋에 몰린다.   query_frames, vote

소리 핑거프린트(audio_fp.py)와 같은 '오프셋 투표'를 쓰지만, 해시가 정확히 같아야 하는 대신
'벡터가 비슷하면' 표를 준다. 그래서 자르기·확대처럼 화면이 조금 바뀐 질의도 이웃으로 잡힌다.

view (프레임을 자르는 방법). 질의가 가로 영상을 세로로 자른 것일 수 있어 한 가지로는 모자란다:
  c  가운데 정사각형       a  긴 변의 앞쪽 끝 정사각형(가로면 왼쪽, 세로면 위)
  z  긴 변의 뒤쪽 끝 정사각형   f  전체를 정사각형으로 눌러 담기(비율 왜곡, 자르지 않음)
색인에는 c,a,z,f를 모두 넣고, 질의 쪽에서 어떤 view·좌우반전을 쓸지 고른다 (반전은 질의 쪽에서만 한다:
색인을 두 배로 키우지 않고 질의 프레임만 두 배로 늘리면 같은 효과다).

실행 환경: 이 모듈의 모델 부분(Embedder)은 서비스 이미지(Docker)에 없다. 호스트 GPU에서
`uv run --group vis python -m eval.run_vis ...`로 돌린다 (Docker에 GPU 런타임이 없고 torch가 이미지를 5GB 넘게 키운다).
질의 1개 지연 (dinov2b, synth_vis_holdout_s3 100개, query_views cz, 다른 작업 없을 때, 2026-10-24):
RTX 4050 Laptop GPU p50 535ms / p95 1031ms. GPU 없이 CPU(WSL 12스레드)만이면 p50 6.7초(10-23 심사 측정).

torch·transformers·av·PIL은 eval용 의존성 그룹(`--group vis`)에만 있으므로 이 파일은 함수 안에서 늦게 import한다
(서비스 이미지에서 이 파일을 import해도 깨지지 않게, 테스트도 numpy만으로 돈다).
"""
import time
from dataclasses import dataclass

import numpy as np
import psycopg

SIZE = 224  # 모델 입력 한 변 (DINOv2·CLIP·SigLIP 공통)
VIEW_IDS = {"c": 0, "a": 1, "z": 2, "f": 3}
VIEW_NAMES = {v: k for k, v in VIEW_IDS.items()}


# ---------------------------------------------------------------- 모델

@dataclass(frozen=True)
class ModelSpec:
    hf: str        # Hugging Face 이름
    mean: tuple    # 입력 정규화 (모델마다 다름)
    std: tuple
    kind: str      # "dino" = pooler_output(CLS), "clip" = get_image_features
    revision: str  # HF 저장소의 고정 커밋 해시: 모델 파일이 나중에 바뀌어 벡터가 달라지는 일을 막는다


_IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
_CLIP = ((0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758))
_HALF = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

MODELS = {
    "dinov2s": ModelSpec("facebook/dinov2-small", *_IMAGENET, "dino",
                         "ed25f3a31f01632728cabb09d1542f84ab7b0056"),
    "dinov2b": ModelSpec("facebook/dinov2-base", *_IMAGENET, "dino",
                         "f9e44c814b77203eaa57a6bdbbd535f21ede1415"),
    "clip": ModelSpec("openai/clip-vit-base-patch16", *_CLIP, "clip",
                       "57c216476eefef5ab752ec549e440a49ae4ae5f3"),
    "siglip": ModelSpec("google/siglip-base-patch16-224", *_HALF, "clip",
                         "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"),
}


class Embedder:
    """이미지 모델 하나. uint8 정사각형 배열 -> 길이 1로 맞춘 벡터. GPU fp16, 작은 배치(6GB 공유)."""

    def __init__(self, name, device="cuda", batch=64):
        import torch
        from transformers import AutoModel
        tables(name)  # 모르는 모델 이름이면 여기서 ValueError
        self.name, self.spec, self.batch = name, MODELS[name], batch
        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(self.spec.hf, revision=self.spec.revision, dtype=dtype).to(self.device).eval()
        shape = (1, 3, 1, 1)
        self.mean = torch.tensor(self.spec.mean, device=self.device, dtype=dtype).view(shape)
        self.std = torch.tensor(self.spec.std, device=self.device, dtype=dtype).view(shape)
        self.dim = self.embed(np.zeros((1, SIZE, SIZE, 3), np.uint8)).shape[1]

    def embed(self, imgs):
        """imgs: (N, 224, 224, 3) uint8 -> (N, dim) float32, 행마다 길이 1 (코사인 = 내적)."""
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(imgs), self.batch):
                x = torch.from_numpy(imgs[i:i + self.batch]).to(self.device).permute(0, 3, 1, 2)
                x = (x.to(self.mean.dtype) / 255 - self.mean) / self.std
                if self.spec.kind == "dino":
                    f = self.model(pixel_values=x).pooler_output
                else:
                    f = self.model.get_image_features(pixel_values=x)
                    f = getattr(f, "pooler_output", f)  # transformers 5는 출력 객체로 감싸 줄 때가 있다
                f = f.float()
                out.append((f / f.norm(dim=1, keepdim=True)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.dim), np.float32)

    def peak_vram_mb(self):
        return self.torch.cuda.max_memory_allocated() / 2**20 if self.device == "cuda" else 0.0


# ---------------------------------------------------------------- 프레임·view

def iter_frames(path, fps=1.0):
    """영상에서 (시각 초, RGB uint8 배열)을 fps장/초로 내보낸다. 순서대로 디코드하며 시각을 지난 첫 프레임을 쓴다."""
    import av
    with av.open(str(path)) as c:
        if not c.streams.video:
            return
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        nxt = 0.0
        for fr in c.decode(s):
            if fr.time is None or fr.time < nxt:
                continue
            yield float(fr.time), fr.to_ndarray(format="rgb24")
            nxt = (int(fr.time * fps) + 1) / fps  # 다음 샘플 시각 (프레임 시각이 어긋나도 한 칸씩만 전진)


def view_boxes(h, w):
    """view -> (위, 아래, 왼, 오른) 자를 상자. 정사각형 변 s=min(h,w)."""
    s = min(h, w)
    cy, cx = (h - s) // 2, (w - s) // 2
    boxes = {"c": (cy, cy + s, cx, cx + s), "f": (0, h, 0, w)}
    if w >= h:   # 가로: 앞 = 왼쪽, 뒤 = 오른쪽
        boxes["a"], boxes["z"] = (0, s, 0, s), (0, s, w - s, w)
    else:        # 세로: 앞 = 위, 뒤 = 아래
        boxes["a"], boxes["z"] = (0, s, 0, s), (h - s, h, 0, s)
    return boxes


def trim_border(img, thr=4.0, min_keep=0.15):
    """가장자리의 단색 띠(검은/흰 레터박스 등)를 잘라 낸다. 줄(행/열) 하나의 화소 표준편차가 thr 미만이면 '띠'.
    질의가 원본을 작게 줄여 띠 위에 올린 경우(frame 변형) 가운데 정사각형이 띠로 채워지는 걸 막으려는 것.
    남는 넓이가 min_keep 미만이면(어두운 장면 전체가 '띠'로 보이는 경우) 자르지 않는다."""
    h, w = img.shape[:2]
    flat = img.reshape(h, w, -1).astype(np.float32)
    row_flat = flat.std(axis=(1, 2)) < thr
    col_flat = flat.std(axis=(0, 2)) < thr

    def edges(mask):
        n = len(mask)
        lo = 0
        while lo < n and mask[lo]:
            lo += 1
        hi = n
        while hi > lo and mask[hi - 1]:
            hi -= 1
        return lo, hi

    top, bot = edges(row_flat)
    left, right = edges(col_flat)
    if bot - top < 8 or right - left < 8 or (bot - top) * (right - left) < min_keep * h * w:
        return img
    return img[top:bot, left:right]


def make_views(img, kinds="cazf", flip=False):
    """프레임 하나 -> [(view 글자, (224,224,3) uint8)]. 상자가 같은 view는 하나만 (정사각형 프레임은 c=a=z)."""
    from PIL import Image
    h, w = img.shape[:2]
    boxes = view_boxes(h, w)
    out, seen = [], set()
    for k in kinds:
        b = boxes[k]
        if b in seen:
            continue
        seen.add(b)
        crop = img[b[0]:b[1], b[2]:b[3]]
        if flip:
            crop = crop[:, ::-1]
        # PIL의 bicubic은 줄일 때 앞뒤 화소를 평균해 주므로(안티앨리어싱) 360p -> 224가 번지지 않는다
        out.append((k, np.asarray(Image.fromarray(np.ascontiguousarray(crop)).resize((SIZE, SIZE), Image.BICUBIC))))
    return out


# ---------------------------------------------------------------- DB (모델마다 표 하나)

def tables(model):
    """모델 이름 -> (영상 표, 프레임 표). 표 이름을 f-string으로 SQL에 붙이므로 이름은 MODELS에 있는 것만 허용한다
    (SQL 값처럼 %s로 넘길 수 없는 자리라서, 외부 입력이 들어오면 주입 통로가 된다)."""
    if model not in MODELS:
        raise ValueError(f"모르는 모델: {model!r} (가능: {sorted(MODELS)})")
    return f"vis_src_{model}", f"vis_frame_{model}"


def index_params(model, fps=1.0, views="cazf"):
    """색인 결과를 좌우하는 설정 전부. 영상 행(src 표 params)에 저장하고, 질의 때 대조한다.
    다르면 벡터 공간이 달라 점수가 조용히 엉망이 되므로(예: 모델 파일이 바뀜, fps가 다름) 막는다."""
    m = MODELS[model]
    return {"model": model, "hf": m.hf, "revision": m.revision, "pooling": m.kind,
            "mean": list(m.mean), "std": list(m.std), "size": SIZE, "fps": fps, "views": views,
            "metric": "cosine"}  # 벡터는 길이 1로 정규화해 저장, HNSW는 halfvec_cosine_ops


# 질의 쪽 설정과 같아야 하는 항목 (views는 색인 전용: 질의는 일부 view만 써도 된다)
QUERY_MUST_MATCH = ("model", "hf", "revision", "pooling", "mean", "std", "size", "fps", "metric")


def check_params(conn, model, fps=1.0, keys=QUERY_MUST_MATCH, views=None):
    """색인에 쓴 설정과 지금 설정이 다르면 ValueError. 색인이 비었으면 통과.
    영상 행에 params가 없으면(옛 색인) 무엇으로 만들었는지 모르므로 이것도 거부한다."""
    src, _ = tables(model)
    row = conn.execute(f"SELECT params FROM {src} LIMIT 1").fetchone()
    if row is None:
        return
    if row[0] is None:
        raise ValueError(f"{src}에 params가 없는 색인: 어떤 설정으로 만들었는지 알 수 없다 (다시 색인할 것)")
    mine = index_params(model, fps, views or row[0].get("views"))
    diff = {k: (row[0].get(k), mine[k]) for k in keys if row[0].get(k) != mine[k]}
    if diff:
        raise ValueError(f"색인과 지금 설정이 다름 (키: (색인, 지금)): {diff}")


def ddl(model, dim):
    """모델별 표 만들기. 모델마다 벡터 길이가 달라 표를 나눈다. 같은 문장이 db/init/07_visual_emb.sql에도 있다(고른 모델)."""
    src, frame = tables(model)
    return f"""
CREATE TABLE IF NOT EXISTS {src} (
    src_id   INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    video_id TEXT NOT NULL UNIQUE,
    dur_sec  REAL,
    n_frames INTEGER,
    params   JSONB              -- index_params(): 모델·revision·fps·views 등. 질의 때 check_params가 대조
);
ALTER TABLE {src} ADD COLUMN IF NOT EXISTS params JSONB;  -- params 열이 생기기 전에 만든 표를 위해
CREATE TABLE IF NOT EXISTS {frame} (
    src_id INTEGER NOT NULL,
    t      REAL NOT NULL,       -- 프레임 시각(초)
    view   SMALLINT NOT NULL,   -- VIEW_IDS: 0=c 1=a 2=z 3=f
    emb    halfvec({dim}) NOT NULL
);
CREATE INDEX IF NOT EXISTS {frame}_hnsw ON {frame} USING hnsw (emb halfvec_cosine_ops);
"""


def connect(url):
    return psycopg.connect(url, autocommit=True)  # 영상마다 커밋 (audio_fp와 같은 이유)


def vec_text(v):
    """pgvector 문자열 표기 '[0.1,0.2,...]'. 소수 5자리면 halfvec(정밀도 약 3자리)에 충분."""
    return "[" + ",".join(f"{x:.5f}" for x in v) + "]"


def index_rows(conn, model, video_id, dur_sec, times, view_ids, embs, params):
    """영상 하나의 프레임 벡터를 한 트랜잭션으로 넣는다. 이미 색인된 영상이면 False.
    times/view_ids/embs는 같은 길이 (행 하나 = 프레임 x view). params = index_params(...)."""
    from psycopg.types.json import Jsonb
    src, frame = tables(model)
    with conn.transaction():
        row = conn.execute(
            f"INSERT INTO {src} (video_id, dur_sec, n_frames, params) VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (video_id) DO NOTHING RETURNING src_id",
            (video_id, dur_sec, len(set(times)), Jsonb(params))).fetchone()
        if row is None:
            return False
        with conn.cursor().copy(f"COPY {frame} (src_id, t, view, emb) FROM STDIN") as cp:
            for t, v, e in zip(times, view_ids, embs):
                cp.write_row((row[0], t, v, vec_text(e)))
    return True


# ---------------------------------------------------------------- 질의

@dataclass(frozen=True)
class QueryParams:
    topk: int = 20            # 질의 벡터 하나당 가져올 이웃 수
    ef_search: int = 100      # HNSW가 훑는 후보 수 (topk보다 커야 정확)
    sim_floor: float = 0.3    # 이보다 덜 닮은 이웃은 표를 못 준다 (모델마다 '남남' 유사도 수준이 달라 조정)
    win: int = 2              # 오프셋을 ±win초 이웃까지 합쳐 센다 (배속·1초 간격 표본화로 오프셋이 흔들린다)
    max_per_frame: int = 5    # 한 질의 프레임이 한 영상에서 만들 수 있는 (오프셋) 표는 가장 닮은 몇 개까지만 (정지 화면이 표를 독점 못 하게)
    query_views: str = "cz"   # 질의 쪽 view
    flip: bool = False        # True면 좌우반전 질의도 같이 찾는다 (dev에서 이득이 없어 기본 꺼짐: DINOv2는 반전에 거의 불변)
    power: float = 1.0        # 표 = (유사도-sim_floor)^power. 1보다 크면 거의 똑같은 프레임에 표가 더 몰린다
    trim: float = 0.0         # >0이면 질의 프레임의 단색 가장자리 띠를 이 표준편차 기준으로 자른다 (0 = 안 함)
    index_views: str = "cazf"  # 이웃 중 이 view의 색인 행만 표로 쓴다 (비교 실험용; 색인은 항상 전부 넣는다)


KNN_SQL = """
SELECT q.i, n.src_id, n.t, n.sim, n.view
FROM unnest(%s::text[]) WITH ORDINALITY AS q(v, i)
CROSS JOIN LATERAL (
    SELECT src_id, t, view, 1 - (emb <=> q.v::halfvec) AS sim
    FROM {frame} ORDER BY emb <=> q.v::halfvec LIMIT %s
) n
"""


def knn_db(conn, model, vecs, qp):
    """벡터 N개 -> 이웃 행 배열 [(질의 번호 0..N-1, src_id, t, sim, view)] (pgvector HNSW)."""
    _, frame = tables(model)
    conn.execute(f"SET hnsw.ef_search = {int(qp.ef_search)}")
    rows = conn.execute(KNN_SQL.format(frame=frame), ([vec_text(v) for v in vecs], qp.topk)).fetchall()
    if not rows:
        return np.zeros((0, 5))
    a = np.array(rows, dtype=np.float64)
    a[:, 0] -= 1  # ORDINALITY는 1부터
    return a


def vote(nn, q_frame, q_time, qp):
    """이웃 행들 -> {src_id: (점수, 시작 초, 표를 준 질의 프레임 수)}.

    nn[:,0] = 질의 벡터 번호, q_frame[번호] = 질의 프레임 번호, q_time[프레임] = 질의 프레임 시각.
    표 = (유사도 - sim_floor). 같은 프레임이 같은 (영상, 오프셋)에 여러 view·반전·이웃으로 표를 줘도 가장 큰 것 하나만 센다.
    오프셋 c의 점수 = c±win 안에서 프레임마다 가장 큰 표를 더한 값."""
    if len(nn) and qp.index_views != "cazf":
        nn = nn[np.isin(nn[:, 4].astype(int), [VIEW_IDS[k] for k in qp.index_views])]
    if len(nn) == 0:
        return {}
    qi = nn[:, 0].astype(int)
    frame = np.asarray(q_frame)[qi]
    off = nn[:, 2] - np.asarray(q_time)[frame]
    w = np.maximum(nn[:, 3] - qp.sim_floor, 0) ** qp.power
    keep = w > 0
    src, frame, off, w = nn[keep, 1].astype(int), frame[keep], off[keep], w[keep]
    # (영상, 프레임, 반올림한 오프셋)마다 최대 표와 그 오프셋 값
    best = {}
    for s, f, o, x in zip(src, frame, off, w):
        k = (s, f, int(round(o)))
        if k not in best or x > best[k][0]:
            best[k] = (x, o)
    # 프레임 하나가 한 영상에서 낼 수 있는 표 수를 제한: 정지 화면에서는 같은 장면 수십 장이 모두 이웃이라 오프셋이 사방에 퍼진다
    per = {}
    for (s, f, b), (x, o) in best.items():
        per.setdefault((s, f), []).append((x, b, o))
    by_src = {}
    for (s, f), lst in per.items():
        for x, b, o in sorted(lst, reverse=True)[:qp.max_per_frame]:
            by_src.setdefault(s, {}).setdefault(b, {})[f] = (x, o)
    result = {}
    for s, bins in by_src.items():
        top = None
        for c in bins:
            frames = {}  # 프레임 -> 창 안 최대 (표, 오프셋)
            for b in range(c - qp.win, c + qp.win + 1):
                for f, (x, o) in bins.get(b, {}).items():
                    if f not in frames or x > frames[f][0]:
                        frames[f] = (x, o)
            score = sum(x for x, _ in frames.values())
            if top is None or score > top[0]:
                xs = np.array([x for x, _ in frames.values()])
                os_ = np.array([o for _, o in frames.values()])
                top = (score, float((xs * os_).sum() / xs.sum()), len(frames))
        result[s] = top
    return result


def query_frames(conn, model, embedder, frames, qp=QueryParams(), fps=1.0, top_k=10, tm=None):
    """질의 프레임 [(시각, RGB 배열)] -> (후보 목록 [{video_id, start_sec, score, n_frames}], 시간 내역 dict).
    시간 내역(초): embed·knn·vote — 어디가 느린지 보려고. 색인과 설정이 다르면 ValueError."""
    tm = {} if tm is None else tm
    check_params(conn, model, fps)
    if not frames:
        return [], tm
    t0 = time.perf_counter()
    imgs, q_frame = [], []
    for fi, (_, img) in enumerate(frames):
        if qp.trim > 0:
            img = trim_border(img, qp.trim)
        for flip in ([False, True] if qp.flip else [False]):
            for _, v in make_views(img, qp.query_views, flip):
                imgs.append(v)
                q_frame.append(fi)
    vecs = embedder.embed(np.stack(imgs))
    tm["embed"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    nn = knn_db(conn, model, vecs, qp)
    tm["knn"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    res = vote(nn, q_frame, [t for t, _ in frames], qp)
    top = sorted(res.items(), key=lambda kv: -kv[1][0])[:top_k]
    src, _ = tables(model)
    names = dict(conn.execute(f"SELECT src_id, video_id FROM {src} WHERE src_id = ANY(%s)",
                              ([s for s, _ in top],)).fetchall())
    cands = [{"video_id": names[s], "start_sec": round(max(st, 0.0), 2), "score": round(float(sc), 3), "n_frames": n}
             for s, (sc, st, n) in top]
    tm["vote"] = time.perf_counter() - t0
    return cands, tm


def query_clip(conn, model, embedder, path, qp=QueryParams(), fps=1.0, top_k=10):
    """질의 영상 파일 -> query_frames와 같은 결과 (+ 시간 내역에 decode)."""
    t0 = time.perf_counter()
    frames = list(iter_frames(path, fps))
    tm = {"decode": time.perf_counter() - t0}
    return query_frames(conn, model, embedder, frames, qp, fps, top_k, tm)
