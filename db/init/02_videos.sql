-- 수집한 유튜브 영상 한 편 = 한 행 (설계 결정 ④: ORM 없이 SQL로 직접 관리)
CREATE TABLE IF NOT EXISTS videos (
    video_id      TEXT PRIMARY KEY,                    -- 유튜브 영상 ID
    channel_id    TEXT NOT NULL,                       -- 채널 ID
    title         TEXT NOT NULL,                       -- 일본어 원제 (공식 API 기준)
    description   TEXT,                                -- 설명란: 원 출처 정보가 있는 곳
    published_at  TIMESTAMPTZ,                         -- 업로드 시각: "처음 올라온 곳" 판단
    duration_sec  INTEGER,                             -- 길이(초)
    collected_at  TIMESTAMPTZ NOT NULL DEFAULT now()   -- 우리가 수집한 시각
);
