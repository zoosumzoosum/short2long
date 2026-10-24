-- 화면 매칭 색인 (4주 차, 후보 B): 고른 모델 DINOv2-base(768차원)의 표만 둔다. 같은 문장이 app/visual_emb.py ddl()에서 만들어진다.
-- 비교용 표(dinov2s·clip·siglip)는 서비스에 넣지 않는다: eval.run_vis index가 호스트 GPU에서 직접 만든다.

CREATE TABLE IF NOT EXISTS vis_src_dinov2b (
    src_id   INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    video_id TEXT NOT NULL UNIQUE,
    dur_sec  REAL,
    n_frames INTEGER,
    params   JSONB              -- index_params(): 모델·revision·fps·views 등. 질의 때 check_params가 대조
);
ALTER TABLE vis_src_dinov2b ADD COLUMN IF NOT EXISTS params JSONB;  -- params 열이 생기기 전에 만든 표를 위해
CREATE TABLE IF NOT EXISTS vis_frame_dinov2b (
    src_id INTEGER NOT NULL,
    t      REAL NOT NULL,       -- 프레임 시각(초)
    view   SMALLINT NOT NULL,   -- VIEW_IDS: 0=c 1=a 2=z 3=f
    emb    halfvec(768) NOT NULL
);
CREATE INDEX IF NOT EXISTS vis_frame_dinov2b_hnsw ON vis_frame_dinov2b USING hnsw (emb halfvec_cosine_ops);

