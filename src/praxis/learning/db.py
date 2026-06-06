"""learning.db — goals, plans, plan_days, mastery, check-ins."""
import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from praxis.learning.models import (
    CheckInResult,
    FlashcardDraft,
    FlashcardRow,
    FeynmanResult,
    GoalRow,
    NoteRow,
    Plan,
    PlanDayRow,
    PlanRow,
    MasteryRow,
    ResourceCheck,
    UserRow,
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
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL DEFAULT 1 REFERENCES users(id),
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

            CREATE TABLE IF NOT EXISTS concept_edges (
                goal_id INTEGER NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                relation TEXT NOT NULL DEFAULT 'prerequisite',
                PRIMARY KEY (goal_id, source, target)
            );

            CREATE TABLE IF NOT EXISTS flashcards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id INTEGER NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
                concept TEXT NOT NULL,
                front TEXT NOT NULL,
                back TEXT NOT NULL,
                due_at REAL NOT NULL,
                stability REAL NOT NULL DEFAULT 1.0,
                difficulty REAL NOT NULL DEFAULT 5.0,
                reps INTEGER NOT NULL DEFAULT 0,
                lapses INTEGER NOT NULL DEFAULT 0,
                last_reviewed REAL,
                UNIQUE(goal_id, front)
            );
            CREATE INDEX IF NOT EXISTS idx_flashcards_due ON flashcards(goal_id, due_at);

            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id INTEGER NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
                filename TEXT NOT NULL,
                content TEXT NOT NULL,
                concepts_json TEXT NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS socratic_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id INTEGER NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
                concept TEXT NOT NULL,
                transcript_json TEXT NOT NULL,
                understanding REAL NOT NULL,
                completed INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS replan_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id INTEGER NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
                reason TEXT NOT NULL,
                weak_concepts_json TEXT NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS roadmap_targets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id INTEGER NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                target_type TEXT NOT NULL DEFAULT 'checkpoint',
                target_value REAL NOT NULL DEFAULT 1.0,
                concept TEXT,
                completed INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL
            );
        """)
        c.execute(
            "INSERT OR IGNORE INTO users (id, name, created_at) VALUES (1, 'Local learner', ?)",
            (time.time(),),
        )
        columns = {r["name"] for r in c.execute("PRAGMA table_info(goals)").fetchall()}
        if "user_id" not in columns:
            c.execute("ALTER TABLE goals ADD COLUMN user_id INTEGER NOT NULL DEFAULT 1")
        day_columns = {r["name"] for r in c.execute("PRAGMA table_info(plan_days)").fetchall()}
        for name, definition in (
            ("checkpoint", "TEXT NOT NULL DEFAULT ''"),
            ("success_criteria_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("estimated_minutes", "INTEGER NOT NULL DEFAULT 60"),
            ("difficulty", "TEXT NOT NULL DEFAULT 'core'"),
        ):
            if name not in day_columns:
                c.execute(f"ALTER TABLE plan_days ADD COLUMN {name} {definition}")
        c.execute(
            """DELETE FROM flashcards
               WHERE front LIKE 'What is %, in your own words?'
                  OR front LIKE 'How would you apply %?'"""
        )
        plan_rows = c.execute("SELECT id, goal_id FROM plans WHERE is_active=1").fetchall()
        for plan_row in plan_rows:
            edge_count = c.execute(
                "SELECT COUNT(*) AS n FROM concept_edges WHERE goal_id=?",
                (plan_row["goal_id"],),
            ).fetchone()["n"]
            if edge_count:
                continue
            previous: list[str] = []
            day_rows = c.execute(
                "SELECT concepts_json FROM plan_days WHERE plan_id=? ORDER BY day_num",
                (plan_row["id"],),
            ).fetchall()
            for day_row in day_rows:
                current = json.loads(day_row["concepts_json"])
                for target in current:
                    for source in previous[-6:]:
                        if source.lower() != target.lower():
                            c.execute(
                                """INSERT OR IGNORE INTO concept_edges
                                   (goal_id, source, target, relation)
                                   VALUES (?,?,?,'prerequisite')""",
                                (plan_row["goal_id"], source, target),
                            )
                previous.extend(current)


# ---------- Goals ----------

def create_user(name: str) -> int:
    with conn() as c:
        cur = c.execute(
            "INSERT INTO users (name, created_at) VALUES (?,?)",
            (name.strip(), time.time()),
        )
        return cur.lastrowid


def list_users() -> list[UserRow]:
    with conn() as c:
        rows = c.execute("SELECT * FROM users ORDER BY id").fetchall()
    return [
        UserRow(id=r["id"], name=r["name"], created_at=datetime.fromtimestamp(r["created_at"]))
        for r in rows
    ]


def create_goal(text: str, level: str, deadline_days: int, user_id: int = 1) -> int:
    with conn() as c:
        cur = c.execute(
            "INSERT INTO goals (user_id, text, level, deadline_days, created_at) VALUES (?,?,?,?,?)",
            (user_id, text, level, deadline_days, time.time()),
        )
        return cur.lastrowid


def get_goal(goal_id: int) -> GoalRow | None:
    with conn() as c:
        r = c.execute("SELECT * FROM goals WHERE id=?", (goal_id,)).fetchone()
    if not r:
        return None
    return GoalRow(
        id=r["id"], user_id=r["user_id"], text=r["text"], level=r["level"],
        deadline_days=r["deadline_days"],
        created_at=datetime.fromtimestamp(r["created_at"]),
    )


def delete_goal(goal_id: int) -> bool:
    """Delete a goal and all its plans/days/mastery/check-ins (FK cascade). Returns True if a row was removed."""
    with conn() as c:
        cur = c.execute("DELETE FROM goals WHERE id=?", (goal_id,))
        return cur.rowcount > 0


def list_goals(user_id: int | None = None) -> list[GoalRow]:
    with conn() as c:
        if user_id is None:
            rows = c.execute("SELECT * FROM goals ORDER BY created_at DESC").fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM goals WHERE user_id=? ORDER BY created_at DESC", (user_id,),
            ).fetchall()
    return [
        GoalRow(
            id=r["id"], user_id=r["user_id"], text=r["text"], level=r["level"],
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
                   (plan_id, day_num, topic, objective, concepts_json, activities_json,
                    resources_json, checkpoint, success_criteria_json, estimated_minutes, difficulty)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (plan_id, d.day_num, d.topic, d.objective,
                 json.dumps(d.concepts), json.dumps(d.activities),
                 json.dumps(d.suggested_resources), d.checkpoint,
                 json.dumps(d.success_criteria), d.estimated_minutes, d.difficulty),
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
        c.execute("DELETE FROM concept_edges WHERE goal_id=?", (goal_id,))
        previous: list[str] = []
        for d in plan.days:
            current = [x.strip() for x in d.concepts if x.strip()]
            for target in current:
                for source in previous[-6:]:
                    if source.lower() != target.lower():
                        c.execute(
                            """INSERT OR IGNORE INTO concept_edges
                               (goal_id, source, target, relation) VALUES (?,?,?,'prerequisite')""",
                            (goal_id, source, target),
                        )
            previous.extend(current)
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
        checkpoint=r["checkpoint"] or f"Create evidence that demonstrates: {r['objective']}",
        success_criteria=json.loads(r["success_criteria_json"] or "[]"),
        estimated_minutes=r["estimated_minutes"] or 60,
        difficulty=r["difficulty"] or "core",
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


# ---------- Shared concept graph ----------

def concept_graph(goal_id: int) -> dict:
    plan = get_active_plan(goal_id)
    if not plan:
        return {"nodes": [], "edges": []}
    days = get_plan_days(plan.id)
    mastery = {m.concept.lower(): m for m in get_mastery(goal_id)}
    day_map: dict[str, list[int]] = {}
    names: dict[str, str] = {}
    for day in days:
        for concept in day.concepts:
            key = concept.lower()
            names[key] = concept
            day_map.setdefault(key, []).append(day.day_num)
    with conn() as c:
        edges = [
            dict(r) for r in c.execute(
                "SELECT source, target, relation FROM concept_edges WHERE goal_id=?",
                (goal_id,),
            ).fetchall()
        ]
    prereqs: dict[str, list[str]] = {}
    for edge in edges:
        prereqs.setdefault(edge["target"].lower(), []).append(edge["source"])
    nodes = []
    for key, name in names.items():
        m = mastery.get(key)
        nodes.append({
            "concept": name,
            "mastery": m.score if m else 0.0,
            "samples": m.samples if m else 0,
            "day_nums": day_map[key],
            "prerequisites": prereqs.get(key, []),
        })
    return {"nodes": nodes, "edges": edges}


# ---------- Spaced repetition ----------

def save_flashcards(goal_id: int, cards: list[FlashcardDraft]) -> int:
    created = 0
    with conn() as c:
        for card in cards:
            numeric_difficulty = {"foundation": 3.5, "application": 5.0, "analysis": 6.5}[card.difficulty]
            cur = c.execute(
                """INSERT OR IGNORE INTO flashcards
                   (goal_id, concept, front, back, due_at, difficulty)
                   VALUES (?,?,?,?,?,?)""",
                (goal_id, card.concept, card.front, card.back, time.time(), numeric_difficulty),
            )
            created += cur.rowcount
    return created


def due_flashcards(goal_id: int, limit: int = 20) -> list[FlashcardRow]:
    with conn() as c:
        rows = c.execute(
            """SELECT * FROM flashcards WHERE goal_id=? AND due_at<=?
               ORDER BY due_at, difficulty DESC LIMIT ?""",
            (goal_id, time.time(), limit),
        ).fetchall()
    return [_flashcard_row(r) for r in rows]


def all_flashcards(goal_id: int) -> list[FlashcardRow]:
    with conn() as c:
        rows = c.execute(
            "SELECT * FROM flashcards WHERE goal_id=? ORDER BY due_at", (goal_id,),
        ).fetchall()
    return [_flashcard_row(r) for r in rows]


def _flashcard_row(r) -> FlashcardRow:
    return FlashcardRow(
        id=r["id"], goal_id=r["goal_id"], concept=r["concept"],
        front=r["front"], back=r["back"],
        due_at=datetime.fromtimestamp(r["due_at"]),
        stability=r["stability"], difficulty=r["difficulty"],
        reps=r["reps"], lapses=r["lapses"],
    )


def review_flashcard(card_id: int, rating: int) -> FlashcardRow | None:
    with conn() as c:
        r = c.execute("SELECT * FROM flashcards WHERE id=?", (card_id,)).fetchone()
        if not r:
            return None
        stability = float(r["stability"])
        difficulty = float(r["difficulty"])
        reps = r["reps"] + 1
        lapses = r["lapses"]
        if rating == 1:
            interval_days = 5 / 1440
            stability = max(0.25, stability * 0.45)
            difficulty = min(10.0, difficulty + 0.8)
            lapses += 1
        elif rating == 2:
            interval_days = max(1.0, stability * 1.2)
            stability *= 1.15
            difficulty = min(10.0, difficulty + 0.2)
        elif rating == 3:
            interval_days = max(1.0, stability * 2.5)
            stability *= 1.8
            difficulty = max(1.0, difficulty - 0.15)
        else:
            interval_days = max(3.0, stability * 4.0)
            stability *= 2.4
            difficulty = max(1.0, difficulty - 0.5)
        due_at = time.time() + interval_days * 86400
        c.execute(
            """UPDATE flashcards SET due_at=?, stability=?, difficulty=?, reps=?,
               lapses=?, last_reviewed=? WHERE id=?""",
            (due_at, stability, difficulty, reps, lapses, time.time(), card_id),
        )
        goal_id, concept = r["goal_id"], r["concept"]
    update_mastery(goal_id, concept, {1: 0.15, 2: 0.45, 3: 0.75, 4: 0.95}[rating])
    with conn() as c:
        return _flashcard_row(c.execute("SELECT * FROM flashcards WHERE id=?", (card_id,)).fetchone())


# ---------- Notes / ingestion ----------

def save_note(goal_id: int, filename: str, content: str, concepts: list[str]) -> int:
    with conn() as c:
        cur = c.execute(
            """INSERT INTO notes (goal_id, filename, content, concepts_json, created_at)
               VALUES (?,?,?,?,?)""",
            (goal_id, filename, content, json.dumps(concepts), time.time()),
        )
        for concept in concepts:
            c.execute(
                "INSERT OR IGNORE INTO mastery (goal_id, concept, score, samples) VALUES (?,?,0,0)",
                (goal_id, concept),
            )
        return cur.lastrowid


def list_notes(goal_id: int) -> list[NoteRow]:
    with conn() as c:
        rows = c.execute(
            "SELECT * FROM notes WHERE goal_id=? ORDER BY created_at DESC", (goal_id,),
        ).fetchall()
    return [
        NoteRow(
            id=r["id"], goal_id=r["goal_id"], filename=r["filename"], content=r["content"],
            concepts=json.loads(r["concepts_json"]), created_at=datetime.fromtimestamp(r["created_at"]),
        )
        for r in rows
    ]


# ---------- Automatic replanning ----------

def maybe_auto_replan(goal_id: int) -> dict | None:
    weak = [m.concept for m in get_mastery(goal_id) if m.samples >= 2 and m.score < 0.4]
    if not weak:
        return None
    with conn() as c:
        recent = c.execute(
            "SELECT created_at FROM replan_events WHERE goal_id=? ORDER BY created_at DESC LIMIT 1",
            (goal_id,),
        ).fetchone()
        if recent and time.time() - recent["created_at"] < 21600:
            return None
        plan = c.execute(
            "SELECT id FROM plans WHERE goal_id=? AND is_active=1 ORDER BY version DESC LIMIT 1",
            (goal_id,),
        ).fetchone()
        if not plan:
            return None
        day = c.execute(
            """SELECT * FROM plan_days WHERE plan_id=? AND status!='done'
               ORDER BY day_num LIMIT 1""",
            (plan["id"],),
        ).fetchone()
        if not day:
            return None
        activities = json.loads(day["activities_json"])
        activity = "Targeted review: revisit " + ", ".join(weak[:4]) + " before new material."
        if activity not in activities:
            activities.insert(0, activity)
            c.execute(
                "UPDATE plan_days SET activities_json=? WHERE id=?",
                (json.dumps(activities), day["id"]),
            )
        c.execute(
            """INSERT INTO replan_events (goal_id, reason, weak_concepts_json, created_at)
               VALUES (?,?,?,?)""",
            (goal_id, "mastery drift", json.dumps(weak[:8]), time.time()),
        )
    return {"plan_day_id": day["id"], "weak_concepts": weak[:8], "activity": activity}


def replan_events(goal_id: int) -> list[dict]:
    with conn() as c:
        rows = c.execute(
            "SELECT * FROM replan_events WHERE goal_id=? ORDER BY created_at DESC", (goal_id,),
        ).fetchall()
    return [
        {
            "id": r["id"], "reason": r["reason"],
            "weak_concepts": json.loads(r["weak_concepts_json"]),
            "created_at": datetime.fromtimestamp(r["created_at"]).isoformat(),
        }
        for r in rows
    ]


# ---------- Interactive roadmap ----------

def roadmap(goal_id: int) -> dict:
    plan = get_active_plan(goal_id)
    if not plan:
        return {"overall_progress": 0, "current_day": None, "checkpoints": [], "targets": []}
    days = get_plan_days(plan.id)
    mastery = {m.concept.lower(): m.score for m in get_mastery(goal_id)}
    checkpoints = []
    complete_count = 0
    current_day = None
    for day in days:
        scores = [mastery.get(c.lower(), 0.0) for c in day.concepts]
        current_mastery = sum(scores) / len(scores) if scores else 0.0
        if day.status == "done":
            progress = 1.0
            complete_count += 1
            state = "complete"
        elif day.status == "in_progress":
            progress = max(0.2, current_mastery)
            state = "active"
            current_day = current_day or day.day_num
        else:
            progress = min(0.75, current_mastery * 0.8)
            state = "ready" if current_day is None and day.day_num == complete_count + 1 else "locked"
            if state == "ready":
                current_day = day.day_num
        checkpoints.append({
            "id": day.id,
            "day_num": day.day_num,
            "topic": day.topic,
            "objective": day.objective,
            "checkpoint": day.checkpoint or f"Demonstrate: {day.objective}",
            "success_criteria": day.success_criteria or [
                "Complete the planned artifact or demonstration.",
                "Explain the key decisions without relying on notes.",
            ],
            "estimated_minutes": day.estimated_minutes,
            "difficulty": day.difficulty,
            "status": day.status,
            "state": state,
            "progress": round(progress, 3),
            "current_mastery": round(current_mastery, 3),
            "target_mastery": 0.8,
            "concepts": day.concepts,
        })
    target_rows = list_roadmap_targets(goal_id)
    target_progress = sum(1 for t in target_rows if t["completed"])
    denominator = len(days) + len(target_rows)
    overall = (complete_count + target_progress) / denominator if denominator else 0.0
    return {
        "overall_progress": round(overall, 3),
        "current_day": current_day,
        "checkpoints": checkpoints,
        "targets": target_rows,
    }


def create_roadmap_target(
    goal_id: int,
    title: str,
    description: str,
    target_type: str,
    target_value: float,
    concept: str | None,
) -> dict:
    with conn() as c:
        cur = c.execute(
            """INSERT INTO roadmap_targets
               (goal_id, title, description, target_type, target_value, concept, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (goal_id, title.strip(), description.strip(), target_type, target_value, concept, time.time()),
        )
        target_id = cur.lastrowid
    return get_roadmap_target(target_id)


def get_roadmap_target(target_id: int) -> dict | None:
    with conn() as c:
        row = c.execute("SELECT * FROM roadmap_targets WHERE id=?", (target_id,)).fetchone()
    return _roadmap_target(row) if row else None


def list_roadmap_targets(goal_id: int) -> list[dict]:
    with conn() as c:
        rows = c.execute(
            "SELECT * FROM roadmap_targets WHERE goal_id=? ORDER BY completed, created_at",
            (goal_id,),
        ).fetchall()
    return [_roadmap_target(row) for row in rows]


def update_roadmap_target(target_id: int, completed: bool) -> dict | None:
    with conn() as c:
        cur = c.execute(
            "UPDATE roadmap_targets SET completed=? WHERE id=?",
            (int(completed), target_id),
        )
        if not cur.rowcount:
            return None
    return get_roadmap_target(target_id)


def delete_roadmap_target(target_id: int) -> bool:
    with conn() as c:
        return c.execute("DELETE FROM roadmap_targets WHERE id=?", (target_id,)).rowcount > 0


def _roadmap_target(row) -> dict:
    return {
        "id": row["id"],
        "goal_id": row["goal_id"],
        "title": row["title"],
        "description": row["description"],
        "target_type": row["target_type"],
        "target_value": row["target_value"],
        "concept": row["concept"],
        "completed": bool(row["completed"]),
        "created_at": datetime.fromtimestamp(row["created_at"]).isoformat(),
    }
