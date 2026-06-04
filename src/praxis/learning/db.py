"""learning.db — goals, plans, plan_days, mastery, check-ins."""
import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from praxis.learning.models import (
    CheckInResult,
    FeynmanResult,
    GoalRow,
    Plan,
    PlanDayRow,
    PlanRow,
    MasteryRow,
    ResourceCheck,
)


DB_PATH = str(Path(__file__).resolve().parents[3] / "learning.db")


@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init() -> None:
    with conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                level TEXT NOT NULL,
                deadline_days INTEGER NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id INTEGER NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
                version INTEGER NOT NULL DEFAULT 1,
                summary TEXT,
                reasoning TEXT,
                self_check TEXT,
                reasoning_types_json TEXT,
                caveats_json TEXT,
                total_days INTEGER NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS plan_days (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_id INTEGER NOT NULL REFERENCES plans(id) ON DELETE CASCADE,
                day_num INTEGER NOT NULL,
                topic TEXT NOT NULL,
                objective TEXT NOT NULL,
                concepts_json TEXT NOT NULL,
                activities_json TEXT NOT NULL,
                resources_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
            );
            CREATE INDEX IF NOT EXISTS idx_plan_days_plan ON plan_days(plan_id, day_num);

            CREATE TABLE IF NOT EXISTS mastery (
                goal_id INTEGER NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
                concept TEXT NOT NULL,
                score REAL NOT NULL DEFAULT 0.0,
                samples INTEGER NOT NULL DEFAULT 0,
                last_assessed REAL,
                PRIMARY KEY (goal_id, concept)
            );

            CREATE TABLE IF NOT EXISTS check_ins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_day_id INTEGER NOT NULL REFERENCES plan_days(id) ON DELETE CASCADE,
                result_json TEXT NOT NULL,
                overall_score REAL NOT NULL,
                completed_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_check_ins_day ON check_ins(plan_day_id, completed_at DESC);

            CREATE TABLE IF NOT EXISTS feynman_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id INTEGER NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
                concept TEXT NOT NULL,
                result_json TEXT NOT NULL,
                final_understanding REAL NOT NULL,
                completed_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_feynman_goal
                ON feynman_sessions(goal_id, concept, completed_at DESC);

            CREATE TABLE IF NOT EXISTS resources (
                plan_day_id INTEGER PRIMARY KEY REFERENCES plan_days(id) ON DELETE CASCADE,
                check_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
        """)


# ---------- Goals ----------

def create_goal(text: str, level: str, deadline_days: int) -> int:
    with conn() as c:
        cur = c.execute(
            "INSERT INTO goals (text, level, deadline_days, created_at) VALUES (?,?,?,?)",
            (text, level, deadline_days, time.time()),
        )
        return cur.lastrowid


def get_goal(goal_id: int) -> GoalRow | None:
    with conn() as c:
        r = c.execute("SELECT * FROM goals WHERE id=?", (goal_id,)).fetchone()
    if not r:
        return None
    return GoalRow(
        id=r["id"], text=r["text"], level=r["level"],
        deadline_days=r["deadline_days"],
        created_at=datetime.fromtimestamp(r["created_at"]),
    )


def delete_goal(goal_id: int) -> bool:
    """Delete a goal and all its plans/days/mastery/check-ins (FK cascade). Returns True if a row was removed."""
    with conn() as c:
        cur = c.execute("DELETE FROM goals WHERE id=?", (goal_id,))
        return cur.rowcount > 0


def list_goals() -> list[GoalRow]:
    with conn() as c:
        rows = c.execute("SELECT * FROM goals ORDER BY created_at DESC").fetchall()
    return [
        GoalRow(
            id=r["id"], text=r["text"], level=r["level"],
            deadline_days=r["deadline_days"],
            created_at=datetime.fromtimestamp(r["created_at"]),
        )
        for r in rows
    ]


# ---------- Plans ----------

def save_plan(goal_id: int, plan: Plan) -> int:
    """Insert a Plan + all PlanDays. Marks any prior plan as inactive."""
    with conn() as c:
        c.execute("UPDATE plans SET is_active=0 WHERE goal_id=?", (goal_id,))
        prev = c.execute(
            "SELECT COALESCE(MAX(version), 0) AS v FROM plans WHERE goal_id=?",
            (goal_id,),
        ).fetchone()
        version = (prev["v"] if prev else 0) + 1
        cur = c.execute(
            """INSERT INTO plans
               (goal_id, version, summary, reasoning, self_check,
                reasoning_types_json, caveats_json, total_days, is_active, created_at)
               VALUES (?,?,?,?,?,?,?,?,1,?)""",
            (goal_id, version, plan.summary, plan.reasoning, plan.self_check,
             json.dumps(plan.reasoning_types), json.dumps(plan.caveats),
             plan.total_days, time.time()),
        )
        plan_id = cur.lastrowid
        for d in plan.days:
            c.execute(
                """INSERT INTO plan_days
                   (plan_id, day_num, topic, objective, concepts_json, activities_json, resources_json)
                   VALUES (?,?,?,?,?,?,?)""",
                (plan_id, d.day_num, d.topic, d.objective,
                 json.dumps(d.concepts), json.dumps(d.activities),
                 json.dumps(d.suggested_resources)),
            )
        # Initialize mastery rows for each unique concept
        seen: set[str] = set()
        for d in plan.days:
            for con in d.concepts:
                key = con.strip().lower()
                if key in seen:
                    continue
                seen.add(key)
                c.execute(
                    """INSERT OR IGNORE INTO mastery (goal_id, concept, score, samples)
                       VALUES (?,?,0.0,0)""",
                    (goal_id, con),
                )
        return plan_id


def get_active_plan(goal_id: int) -> PlanRow | None:
    with conn() as c:
        r = c.execute(
            "SELECT * FROM plans WHERE goal_id=? AND is_active=1 ORDER BY version DESC LIMIT 1",
            (goal_id,),
        ).fetchone()
    if not r:
        return None
    return PlanRow(
        id=r["id"], goal_id=r["goal_id"], version=r["version"],
        summary=r["summary"] or "", total_days=r["total_days"],
        reasoning=r["reasoning"] or "",
        self_check=r["self_check"] or "",
        reasoning_types=json.loads(r["reasoning_types_json"] or "[]"),
        caveats=json.loads(r["caveats_json"] or "[]"),
        created_at=datetime.fromtimestamp(r["created_at"]),
    )


def get_plan_days(plan_id: int) -> list[PlanDayRow]:
    with conn() as c:
        rows = c.execute(
            "SELECT * FROM plan_days WHERE plan_id=? ORDER BY day_num", (plan_id,),
        ).fetchall()
    return [_row_to_plan_day(r) for r in rows]


def get_plan_day(plan_day_id: int) -> PlanDayRow | None:
    with conn() as c:
        r = c.execute("SELECT * FROM plan_days WHERE id=?", (plan_day_id,)).fetchone()
    return _row_to_plan_day(r) if r else None


def get_plan_day_by_num(plan_id: int, day_num: int) -> PlanDayRow | None:
    with conn() as c:
        r = c.execute(
            "SELECT * FROM plan_days WHERE plan_id=? AND day_num=?",
            (plan_id, day_num),
        ).fetchone()
    return _row_to_plan_day(r) if r else None


def _row_to_plan_day(r) -> PlanDayRow:
    return PlanDayRow(
        id=r["id"], plan_id=r["plan_id"], day_num=r["day_num"],
        topic=r["topic"], objective=r["objective"],
        concepts=json.loads(r["concepts_json"]),
        activities=json.loads(r["activities_json"]),
        suggested_resources=json.loads(r["resources_json"]),
        status=r["status"],
    )


def set_plan_day_status(plan_day_id: int, status: str) -> None:
    with conn() as c:
        c.execute("UPDATE plan_days SET status=? WHERE id=?", (status, plan_day_id))


# ---------- Mastery ----------

def get_mastery(goal_id: int) -> list[MasteryRow]:
    with conn() as c:
        rows = c.execute(
            "SELECT * FROM mastery WHERE goal_id=? ORDER BY score ASC, concept ASC",
            (goal_id,),
        ).fetchall()
    return [
        MasteryRow(
            concept=r["concept"], score=r["score"], samples=r["samples"],
            last_assessed=datetime.fromtimestamp(r["last_assessed"]) if r["last_assessed"] else None,
        )
        for r in rows
    ]


def update_mastery(goal_id: int, concept: str, new_sample_score: float) -> None:
    """Update mastery for a concept using a running average."""
    with conn() as c:
        r = c.execute(
            "SELECT score, samples FROM mastery WHERE goal_id=? AND concept=?",
            (goal_id, concept),
        ).fetchone()
        if r:
            samples = r["samples"] + 1
            # Exponential moving average with weight 0.4 on the new sample — recency biased
            new_score = 0.6 * r["score"] + 0.4 * new_sample_score
            c.execute(
                """UPDATE mastery SET score=?, samples=?, last_assessed=?
                   WHERE goal_id=? AND concept=?""",
                (new_score, samples, time.time(), goal_id, concept),
            )
        else:
            c.execute(
                """INSERT INTO mastery (goal_id, concept, score, samples, last_assessed)
                   VALUES (?,?,?,1,?)""",
                (goal_id, concept, new_sample_score, time.time()),
            )


# ---------- Check-ins ----------

def save_check_in(plan_day_id: int, result: CheckInResult) -> int:
    with conn() as c:
        cur = c.execute(
            """INSERT INTO check_ins (plan_day_id, result_json, overall_score, completed_at)
               VALUES (?,?,?,?)""",
            (plan_day_id, result.model_dump_json(), result.overall_score, time.time()),
        )
        return cur.lastrowid


def latest_check_in(plan_day_id: int) -> CheckInResult | None:
    with conn() as c:
        r = c.execute(
            "SELECT * FROM check_ins WHERE plan_day_id=? ORDER BY completed_at DESC LIMIT 1",
            (plan_day_id,),
        ).fetchone()
    if not r:
        return None
    return CheckInResult.model_validate_json(r["result_json"])


def goal_id_for_plan_day(plan_day_id: int) -> int | None:
    """Resolve the owning goal_id for a plan day (plan_day → plan → goal)."""
    with conn() as c:
        r = c.execute(
            """SELECT p.goal_id AS goal_id
               FROM plan_days d JOIN plans p ON p.id = d.plan_id
               WHERE d.id = ?""",
            (plan_day_id,),
        ).fetchone()
    return r["goal_id"] if r else None


# ---------- Feynman sessions ----------

def save_feynman(goal_id: int, result: FeynmanResult) -> int:
    with conn() as c:
        cur = c.execute(
            """INSERT INTO feynman_sessions (goal_id, concept, result_json, final_understanding, completed_at)
               VALUES (?,?,?,?,?)""",
            (goal_id, result.concept, result.model_dump_json(), result.final_understanding, time.time()),
        )
        return cur.lastrowid


def latest_feynman(goal_id: int, concept: str) -> FeynmanResult | None:
    with conn() as c:
        r = c.execute(
            """SELECT result_json FROM feynman_sessions
               WHERE goal_id=? AND concept=? ORDER BY completed_at DESC LIMIT 1""",
            (goal_id, concept),
        ).fetchone()
    if not r:
        return None
    return FeynmanResult.model_validate_json(r["result_json"])


# ---------- Verified resources ----------

def save_resources(plan_day_id: int, check: ResourceCheck) -> None:
    with conn() as c:
        c.execute(
            """INSERT INTO resources (plan_day_id, check_json, updated_at) VALUES (?,?,?)
               ON CONFLICT(plan_day_id) DO UPDATE SET check_json=excluded.check_json,
                                                      updated_at=excluded.updated_at""",
            (plan_day_id, check.model_dump_json(), time.time()),
        )


def get_resources(plan_day_id: int) -> ResourceCheck | None:
    with conn() as c:
        r = c.execute("SELECT check_json FROM resources WHERE plan_day_id=?", (plan_day_id,)).fetchone()
    if not r:
        return None
    return ResourceCheck.model_validate_json(r["check_json"])
