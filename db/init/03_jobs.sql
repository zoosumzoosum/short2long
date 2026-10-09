-- 수집 주문서 (설계 결정 ⑧): list_channel → 영상마다 fetch_video
CREATE TABLE IF NOT EXISTS jobs (
    id          BIGSERIAL PRIMARY KEY,                -- 주문서 번호
    kind        TEXT NOT NULL,                        -- 'list_channel' | 'fetch_video'
    target      TEXT NOT NULL,                        -- 채널 ID 또는 영상 ID
    status      TEXT NOT NULL DEFAULT 'pending',      -- pending → running → done / failed
    attempts    INTEGER NOT NULL DEFAULT 0,           -- 시도 횟수 (무한 재시도 방지)
    error       TEXT,                                 -- 실패 이유
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 아직 안 끝난(pending·running) 주문서는 (종류, 대상)마다 하나만. 끝난 건 다시 주문 가능
CREATE UNIQUE INDEX IF NOT EXISTS jobs_one_active
    ON jobs (kind, target)
    WHERE status IN ('pending', 'running');
