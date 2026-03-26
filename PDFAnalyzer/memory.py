import json
import sqlite3
from pathlib import Path

MEMORY_DB_PATH = Path("memory.db")

# connect to sqlite
def get_memory_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(MEMORY_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# initialize memory
def init_memory() -> None:
    with get_memory_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evidence_memory (
                session_id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

# handles saving most recent messages
def save_message(session_id: str, role: str, content: str) -> None:
    with get_memory_connection() as conn:
        conn.execute(
            "INSERT INTO chat_memory (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )

# handles saving the retrieved information(evidence) for a given session
def save_evidence(session_id: str, query: str, evidence_json: str) -> None:
    with get_memory_connection() as conn:
        conn.execute(
            """
            INSERT INTO evidence_memory (session_id, query, evidence_json)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                query = excluded.query,
                evidence_json = excluded.evidence_json,
                created_at = CURRENT_TIMESTAMP
            """,
            (session_id, query, evidence_json),
        )

# obtain the the retrieved information(evidence) for a given session 
def get_latest_evidence(session_id: str) -> dict | None:
    with get_memory_connection() as conn:
        row = conn.execute(
            """
            SELECT query, evidence_json
            FROM evidence_memory
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()

    if not row:
        return None

    return {
        "query": row["query"],
        "evidence_json": row["evidence_json"],
    }



# obtain recent messages
def get_recent_messages(session_id: str, limit: int = 6) -> list[dict]:
    with get_memory_connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content
            FROM chat_memory
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()

    rows.reverse()
    return [{"role": row["role"], "content": row["content"]} for row in rows]

def get_last_user_message(session_id: str) -> str | None:
    with get_memory_connection() as conn:
        row = conn.execute(
            """
            SELECT content
            FROM chat_memory
            WHERE session_id = ? AND role = 'user'
            ORDER BY id DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()

    return row["content"] if row else None

"""
Summarize the retrieved information for a given query
to be exposed to the model. 
"""
def summarize_cached_evidence(evidence_json: str, max_chunks: int = 3, max_chars: int = 220) -> str:
    if not evidence_json:
        return "None"

    try:
        payload = json.loads(evidence_json)
    except json.JSONDecodeError:
        return "Cached evidence is available, but the summary could not be parsed."

    evidence_query = payload.get("query") or "Unknown"
    chunks = payload.get("chunks") or []

    lines = [f"Cached evidence was retrieved for: {evidence_query}"]
    if not chunks:
        lines.append("No chunks are stored in the cached evidence.")
        return "\n".join(lines)

    lines.append("Top cached chunks:")
    for chunk in chunks[:max_chunks]:
        citation = chunk.get("citation") or f"{chunk.get('document_name', 'unknown')} p.{chunk.get('page_number', '?')}"
        excerpt = " ".join(str(chunk.get("content", "")).split())
        if len(excerpt) > max_chars:
            excerpt = excerpt[:max_chars].rstrip() + "..."
        lines.append(f"- {citation}: {excerpt}")

    return "\n".join(lines)

"""
Returns the query, corresponding raw evidence extracted from documents and the 
summarized evidence. 
"""

def get_cached_evidence_context(session_id: str) -> dict | None:
    latest_evidence = get_latest_evidence(session_id)
    if not latest_evidence:
        return None

    return {
        "query": latest_evidence["query"],
        "evidence_json": latest_evidence["evidence_json"],
        "summary": summarize_cached_evidence(latest_evidence["evidence_json"]),
    }