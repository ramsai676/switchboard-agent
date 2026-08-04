"""Durable state for Switchboard.

Everything the agent learns lives here: the experts it can page, the questions
it has been asked, the pages it has sent out, and — the part that matters — the
answers it has learned so it never has to ask a human the same thing twice.

SQLite because the agent must survive a restart mid-page. An escalation timer
that forgets who it paged is worse than no escalation at all.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "switchboard.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS experts (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    channel       TEXT NOT NULL,      -- email | telegram | discord | slack
    address       TEXT NOT NULL,      -- email address, chat id, user id
    skills        TEXT NOT NULL,      -- JSON list of skill tags
    timezone      TEXT,
    paged_count   INTEGER NOT NULL DEFAULT 0,
    answered_count INTEGER NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS questions (
    id              TEXT PRIMARY KEY,
    text            TEXT NOT NULL,
    asker_channel   TEXT NOT NULL,
    asker_message_id TEXT,
    asker_conversation_id TEXT,
    asker_label     TEXT,
    status          TEXT NOT NULL,    -- answered_from_memory | paging | resolved | abandoned
    resolved_answer TEXT,
    resolved_by     TEXT,             -- expert id, or 'memory'
    skill           TEXT,
    created_at      REAL NOT NULL,
    resolved_at     REAL
);

CREATE TABLE IF NOT EXISTS pages (
    id             TEXT PRIMARY KEY,
    question_id    TEXT NOT NULL,
    expert_id      TEXT NOT NULL,
    channel        TEXT NOT NULL,
    status         TEXT NOT NULL,     -- sent | accepted | declined | answered | timed_out
    conversation_id TEXT,
    sent_at        REAL NOT NULL,
    responded_at   REAL,
    FOREIGN KEY (question_id) REFERENCES questions(id)
);

-- The compounding part: every human answer becomes reusable knowledge.
CREATE TABLE IF NOT EXISTS knowledge (
    id            TEXT PRIMARY KEY,
    question_text TEXT NOT NULL,
    answer_text   TEXT NOT NULL,
    skill         TEXT,
    taught_by     TEXT,
    reuse_count   INTEGER NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT NOT NULL,
    payload    TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pages_question ON pages(question_id);
CREATE INDEX IF NOT EXISTS idx_pages_status   ON pages(status);
CREATE INDEX IF NOT EXISTS idx_questions_status ON questions(status);
"""


def _now() -> float:
    return time.time()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class Expert:
    id: str
    name: str
    channel: str
    address: str
    skills: list[str]
    timezone: str | None = None
    paged_count: int = 0
    answered_count: int = 0

    @property
    def reliability(self) -> float:
        """Share of pages this expert actually answered. Unpaged experts start
        optimistic so a new hire is not starved of questions forever."""
        if self.paged_count == 0:
            return 0.75
        return self.answered_count / self.paged_count


class Store:
    def __init__(self, path: Path | str = DEFAULT_DB) -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL so the dashboard can read while the agent writes.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    @contextmanager
    def _tx(self):
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    # ---- events ---------------------------------------------------------

    def log(self, kind: str, **payload) -> None:
        """Append to the event log. The dashboard tails this to animate."""
        with self._tx() as cur:
            cur.execute(
                "INSERT INTO events (kind, payload, created_at) VALUES (?,?,?)",
                (kind, json.dumps(payload), _now()),
            )

    def events_after(self, seq: int, limit: int = 200) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM events WHERE seq > ? ORDER BY seq LIMIT ?", (seq, limit)
        ).fetchall()
        return [
            {"seq": r["seq"], "kind": r["kind"], "at": r["created_at"], **json.loads(r["payload"])}
            for r in rows
        ]

    # ---- experts --------------------------------------------------------

    def add_expert(
        self, name: str, channel: str, address: str, skills: list[str], timezone: str | None = None
    ) -> Expert:
        expert = Expert(new_id("exp"), name, channel, address, skills, timezone)
        with self._tx() as cur:
            cur.execute(
                "INSERT INTO experts (id,name,channel,address,skills,timezone,created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (expert.id, name, channel, address, json.dumps(skills), timezone, _now()),
            )
        self.log("expert.added", expert_id=expert.id, name=name, channel=channel, skills=skills)
        return expert

    def _row_to_expert(self, r: sqlite3.Row) -> Expert:
        return Expert(
            id=r["id"], name=r["name"], channel=r["channel"], address=r["address"],
            skills=json.loads(r["skills"]), timezone=r["timezone"],
            paged_count=r["paged_count"], answered_count=r["answered_count"],
        )

    def experts(self) -> list[Expert]:
        rows = self._conn.execute("SELECT * FROM experts ORDER BY created_at").fetchall()
        return [self._row_to_expert(r) for r in rows]

    def expert(self, expert_id: str) -> Expert | None:
        r = self._conn.execute("SELECT * FROM experts WHERE id=?", (expert_id,)).fetchone()
        return self._row_to_expert(r) if r else None

    def expert_by_address(self, address: str) -> Expert | None:
        r = self._conn.execute(
            "SELECT * FROM experts WHERE lower(address)=lower(?)", (address,)
        ).fetchone()
        return self._row_to_expert(r) if r else None

    def bump_expert(self, expert_id: str, *, paged: bool = False, answered: bool = False) -> None:
        with self._tx() as cur:
            if paged:
                cur.execute(
                    "UPDATE experts SET paged_count=paged_count+1 WHERE id=?", (expert_id,)
                )
            if answered:
                cur.execute(
                    "UPDATE experts SET answered_count=answered_count+1 WHERE id=?", (expert_id,)
                )

    # ---- questions ------------------------------------------------------

    def add_question(
        self, text: str, asker_channel: str, asker_message_id: str | None,
        asker_conversation_id: str | None, asker_label: str | None, skill: str | None,
    ) -> str:
        qid = new_id("q")
        with self._tx() as cur:
            cur.execute(
                "INSERT INTO questions (id,text,asker_channel,asker_message_id,"
                "asker_conversation_id,asker_label,status,skill,created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (qid, text, asker_channel, asker_message_id, asker_conversation_id,
                 asker_label, "paging", skill, _now()),
            )
        self.log("question.asked", question_id=qid, text=text,
                 channel=asker_channel, asker=asker_label, skill=skill)
        return qid

    def question(self, qid: str) -> dict | None:
        r = self._conn.execute("SELECT * FROM questions WHERE id=?", (qid,)).fetchone()
        return dict(r) if r else None

    def resolve_question(self, qid: str, answer: str, resolved_by: str, status: str) -> None:
        with self._tx() as cur:
            cur.execute(
                "UPDATE questions SET status=?, resolved_answer=?, resolved_by=?, resolved_at=?"
                " WHERE id=?",
                (status, answer, resolved_by, _now(), qid),
            )
        self.log("question.resolved", question_id=qid, resolved_by=resolved_by, status=status)

    def open_questions(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM questions WHERE status='paging' ORDER BY created_at"
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- pages ----------------------------------------------------------

    def add_page(self, question_id: str, expert_id: str, channel: str,
                 conversation_id: str | None) -> str:
        pid = new_id("pg")
        with self._tx() as cur:
            cur.execute(
                "INSERT INTO pages (id,question_id,expert_id,channel,status,conversation_id,sent_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (pid, question_id, expert_id, channel, "sent", conversation_id, _now()),
            )
        self.bump_expert(expert_id, paged=True)
        self.log("page.sent", page_id=pid, question_id=question_id,
                 expert_id=expert_id, channel=channel)
        return pid

    def set_page_status(self, page_id: str, status: str) -> None:
        with self._tx() as cur:
            cur.execute(
                "UPDATE pages SET status=?, responded_at=? WHERE id=?", (status, _now(), page_id)
            )
        self.log("page.status", page_id=page_id, status=status)

    def page(self, page_id: str) -> dict | None:
        r = self._conn.execute("SELECT * FROM pages WHERE id=?", (page_id,)).fetchone()
        return dict(r) if r else None

    def pages_for_question(self, qid: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM pages WHERE question_id=? ORDER BY sent_at", (qid,)
        ).fetchall()
        return [dict(r) for r in rows]

    def open_page_for_expert_conversation(self, conversation_id: str) -> dict | None:
        """Find the page a human is replying to, by the conversation it lives in."""
        r = self._conn.execute(
            "SELECT * FROM pages WHERE conversation_id=? AND status IN ('sent','accepted')"
            " ORDER BY sent_at DESC LIMIT 1",
            (conversation_id,),
        ).fetchone()
        return dict(r) if r else None

    def stale_pages(self, older_than_seconds: float) -> list[dict]:
        cutoff = _now() - older_than_seconds
        rows = self._conn.execute(
            "SELECT * FROM pages WHERE status IN ('sent','accepted') AND sent_at < ?", (cutoff,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- knowledge ------------------------------------------------------

    def learn(self, question_text: str, answer_text: str, skill: str | None,
              taught_by: str | None) -> str:
        kid = new_id("kn")
        with self._tx() as cur:
            cur.execute(
                "INSERT INTO knowledge (id,question_text,answer_text,skill,taught_by,created_at)"
                " VALUES (?,?,?,?,?,?)",
                (kid, question_text, answer_text, skill, taught_by, _now()),
            )
        self.log("knowledge.learned", knowledge_id=kid, question=question_text,
                 skill=skill, taught_by=taught_by)
        return kid

    def knowledge(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM knowledge ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]

    def bump_reuse(self, knowledge_id: str) -> None:
        with self._tx() as cur:
            cur.execute(
                "UPDATE knowledge SET reuse_count=reuse_count+1 WHERE id=?", (knowledge_id,)
            )

    # ---- metrics --------------------------------------------------------

    def stats(self) -> dict:
        q = self._conn.execute(
            "SELECT COUNT(*) c, "
            " SUM(CASE WHEN status='answered_from_memory' THEN 1 ELSE 0 END) deflected, "
            " SUM(CASE WHEN status='resolved' THEN 1 ELSE 0 END) human_resolved, "
            " SUM(CASE WHEN status='paging' THEN 1 ELSE 0 END) in_flight "
            "FROM questions"
        ).fetchone()
        total = q["c"] or 0
        deflected = q["deflected"] or 0
        humans_spared = self._conn.execute(
            "SELECT COALESCE(SUM(reuse_count),0) s FROM knowledge"
        ).fetchone()["s"]
        return {
            "questions_total": total,
            "deflected": deflected,
            "human_resolved": q["human_resolved"] or 0,
            "in_flight": q["in_flight"] or 0,
            "deflection_rate": round(deflected / total, 3) if total else 0.0,
            "knowledge_items": self._conn.execute(
                "SELECT COUNT(*) c FROM knowledge"
            ).fetchone()["c"],
            "human_pages_avoided": humans_spared,
            "experts": self._conn.execute("SELECT COUNT(*) c FROM experts").fetchone()["c"],
        }

    def close(self) -> None:
        self._conn.close()
