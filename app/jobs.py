"""jobs 표에 주문서 넣기 (설계 결정 ⑧)."""


def add_job(conn, kind, target):
    """주문서를 넣는다. 같은 (종류, 대상)이 이미 대기·진행 중이면 건너뛴다 (jobs_one_active). 넣었으면 True"""
    cur = conn.execute(
        """
        INSERT INTO jobs (kind, target) VALUES (%s, %s)
        ON CONFLICT (kind, target) WHERE status IN ('pending', 'running') DO NOTHING
        """,
        (kind, target),
    )
    return cur.rowcount == 1
