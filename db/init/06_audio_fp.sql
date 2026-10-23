-- 소리 핑거프린트 색인 (랜드마크 해시, Shazam 방식)
-- 영상 하나 = audio_fp_src 한 행, 그 영상에서 뽑은 해시 하나 = audio_fp 한 행.
-- 같은 해시가 수백만 행 중 어디에 있는지 찾는 것이 질의의 전부라 hash에 인덱스를 둔다.

CREATE TABLE IF NOT EXISTS audio_fp_src (
    src_id    INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,  -- audio_fp에서 쓰는 짧은 번호 (영상 ID 문자열을 수천만 번 반복 저장하지 않으려고)
    video_id  TEXT NOT NULL UNIQUE,                              -- 유튜브 영상 ID (videos.video_id와 같은 값; 수집 전 평가에서도 쓰려고 FK는 걸지 않음)
    dur_sec   REAL,                                              -- 소리 길이(초)
    n_hashes  INTEGER                                            -- 이 영상에서 뽑은 해시 수
);

CREATE TABLE IF NOT EXISTS audio_fp (
    hash    INTEGER NOT NULL,   -- (피크1 주파수 8비트 | 피크2 주파수 8비트 | 시간차 6비트) = 22비트
    src_id  INTEGER NOT NULL,   -- audio_fp_src.src_id
    t       INTEGER NOT NULL    -- 피크1이 있는 프레임 번호 (미세 칸 단위: 초 = t * 256 / 4 / 8000)
);

CREATE INDEX IF NOT EXISTS audio_fp_hash_idx ON audio_fp (hash);
