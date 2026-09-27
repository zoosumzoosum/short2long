import psycopg
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.config import settings

app = FastAPI(title="short2long")


@app.get("/health")
def health():
    try:
        with psycopg.connect(settings.database_url, connect_timeout=3) as conn:
            row = conn.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
    except psycopg.Error as e:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "db": "unreachable", "detail": type(e).__name__},
        )
    return {"status": "ok", "db": "ok", "pgvector": row[0] if row else None}
