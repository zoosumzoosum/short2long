-- 어떤 영상을 왜 골랐는지 (결정 ⑪): 형식(롱폼·쇼츠·팬 클립)과 계층 샘플링의 층. 평가셋을 층별로 나눌 때 씀
CREATE TABLE IF NOT EXISTS selected_videos (
    video_id     TEXT PRIMARY KEY,
    channel_url  TEXT NOT NULL,
    format       TEXT NOT NULL,          -- 'long' | 'short' | 'fan'
    stratum      TEXT,                   -- 쇼츠의 멤버 층 (예: 真鍋凪咲, (태그 없음))
    selected_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
