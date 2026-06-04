"""Per-call telemetry — one row per LLM call in gateway.db."""
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

DB_PATH = str(Path(__file__).resolve().parents[3] / "gateway.db")


@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init() -> None:
    with conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                latency_ms INTEGER DEFAULT 0,
                status TEXT,
                error TEXT,
                prompt_chars INTEGER DEFAULT 0,
                response_chars INTEGER DEFAULT 0,
                override TEXT,
                attempted TEXT
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_calls_ts ON calls(ts DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_calls_prov_ts ON calls(provider, ts DESC)")


def log_call(
    *,
    provider: str,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    latency_ms: int = 0,
    status: str = "ok",
    error: str | None = None,
    prompt_chars: int = 0,
    response_chars: int = 0,
    override: str | None = None,
    attempted: str | None = None,
) -> int:
    with conn() as c:
        cur = c.execute(
            """INSERT INTO calls
               (ts, provider, model, input_tokens, output_tokens, latency_ms,
                status, error, prompt_chars, response_chars, override, attempted)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (time.time(), provider, model, input_tokens, output_tokens, latency_ms,
             status, error, prompt_chars, response_chars, override, attempted),
        )
        return cur.lastrowid


def update_call(
    call_id: int,
    *,
    status: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    latency_ms: int = 0,
    error: str | None = None,
    response_chars: int = 0,
    attempted: str | None = None,
) -> None:
    with conn() as c:
        c.execute(
            """UPDATE calls
               SET status=?, input_tokens=?, output_tokens=?, latency_ms=?,
                   error=?, response_chars=?, attempted=COALESCE(?, attempted)
               WHERE id=?""",
            (
                status, input_tokens, output_tokens, latency_ms,
                error, response_chars, attempted, call_id,
            ),
        )


def recent(limit: int = 100, provider: str | None = None, status: str | None = None) -> list[dict]:
    q = "SELECT * FROM calls"
    where, args = [], []
    if provider:
        where.append("provider=?")
        args.append(provider)
    if status:
        where.append("status=?")
        args.append(status)
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY ts DESC LIMIT ?"
    args.append(limit)
    with conn() as c:
        return [dict(r) for r in c.execute(q, args).fetchall()]


def aggregate_today() -> dict[str, dict]:
    now = time.time()
    day_start = now - (now % 86400)
    with conn() as c:
        rows = c.execute(
            """SELECT provider,
                      COUNT(*) AS calls,
                      SUM(CASE WHEN status='ok' THEN 1 ELSE 0 END) AS ok_calls,
                      SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS errors,
                      SUM(CASE WHEN status='running' THEN 1 ELSE 0 END) AS running,
                      SUM(input_tokens) AS in_tok,
                      SUM(output_tokens) AS out_tok,
                      AVG(CASE WHEN status!='running' THEN latency_ms END) AS avg_latency,
                      MAX(ts) AS last_ts
                 FROM calls WHERE ts >= ? GROUP BY provider""",
            (day_start,),
        ).fetchall()
        return {r["provider"]: dict(r) for r in rows}
