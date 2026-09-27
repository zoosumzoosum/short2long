# short2long

한국어 장면 묘사로 일본어 영상의 출처 구간을 찾는 다국어 검색 서비스.
쇼츠·릴스로 퍼진 클립을 보고 "원본이 뭐지?"라고 궁금할 때, 원본 영상과 해당 구간, 앞뒤 맥락을 찾아 준다.
테스트 도메인은 일본 아이돌 CUTIE STREET·MORE STAR의 유튜브 영상이다.

> 개발 중 (2026-09-27 시작). 진행 상황은 커밋 기록과 `DECISIONS.md`에 남긴다.

## 실행

```bash
cp .env.example .env   # 값 채우기
docker compose up --build
curl http://localhost:8000/health
```

## 데이터

영상·자막 데이터는 저장소에 올리지 않는다. 결과는 유튜브 링크와 시각으로만 돌려준다.
