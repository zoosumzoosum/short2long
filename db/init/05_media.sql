-- 내려받은 영상 파일 (결정 ⑫): 특징을 뽑은 뒤 파일은 지우고 경로도 비움
ALTER TABLE videos ADD COLUMN IF NOT EXISTS media_path  TEXT;
ALTER TABLE videos ADD COLUMN IF NOT EXISTS media_bytes BIGINT;
