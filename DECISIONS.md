# 설계 결정 기록

형식: 번호 / 날짜 / 결정 / 고른 이유 / 버린 대안

## ① Docker는 WSL 안에 Docker Engine으로 (2026-09-27)
- 결정: Windows의 Docker Desktop 대신 WSL2 Ubuntu 안에 Docker Engine을 직접 설치
- 이유: 실제 리눅스 GPU 서버에 구성하는 방식과 같다. Windows 쪽 메모리를 따로 쓰지 않는다
- 버린 대안: Docker Desktop (설치는 쉽지만 메모리를 더 쓰고 서버 환경과 거리가 있음)

## ② 저장소는 처음부터 공개 (2026-09-27)
- 결정: GitHub 공개 저장소로 시작
- 이유: 진행 과정이 커밋 날짜로 남는다. 보는 사람이 있으면 꾸준히 하게 된다
- 버린 대안: 1차 완성 후 공개 전환
- 조건: 영상·자막 데이터와 `.env`는 올리지 않는다 (`.gitignore`)

## ③ DB는 pgvector 공식 이미지(PostgreSQL 17) (2026-09-27)
- 결정: `pgvector/pgvector:pg17`
- 이유: 일반 PostgreSQL에 벡터 검색 확장(pgvector)이 미리 들어 있다. 메타데이터·작업 상태·벡터를 DB 하나에서 다룬다
- 버린 대안: 벡터 전용 DB(Qdrant·Milvus 등) — 규모(100~150편)에 비해 관리할 것이 늘어남

## ④ DB 접속은 우선 psycopg로 직접 (2026-09-27, 2주 차에 재검토)
- 결정: `/health`는 psycopg로 SQL 한 줄만 실행
- 이유: 지금은 연결 확인만 필요하다. ORM(SQLAlchemy)을 쓸지는 테이블을 만드는 2주 차에 정한다
