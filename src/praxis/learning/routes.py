"""Learning API: goals, plans, check-ins, mastery.

Two response styles per route:
  - JSON for HTMX-less / programmatic callers
  - HTML fragment when `HX-Request: true` is set (HTMX swaps it in)
"""
import json
import re
import uuid

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from praxis.learning import agents, db
from praxis.learning.search import search_many
from praxis.learning.models import (
    CheckInAnswer,
    CheckInPlan,
    CheckInResult,
    CreateRoadmapTargetRequest,
    CreateGoalRequest,
    CreateUserRequest,
    FeynmanExchange,
    FeynmanReplyRequest,
    FeynmanResult,
    AdaptiveQuizAnswerRequest,
    AdaptiveQuizRequest,
    Plan,
    ReviewFlashcardRequest,
    RefinePlanRequest,
    SocraticReplyRequest,
    SocraticStartRequest,
    StartFeynmanRequest,
    SubmitAnswerRequest,
    UpdateRoadmapTargetRequest,
)


router = APIRouter(prefix="/api")

# In-process cache of in-progress check-in sessions: {plan_day_id: CheckInPlan}
# Avoids re-asking the agent to generate the same questions on each answer submission.
# Single-user app — process-local is fine.
_active_checkins: dict[int, CheckInPlan] = {}
_active_answers: dict[int, list[CheckInAnswer]] = {}

# In-process Feynman sessions, keyed by an opaque token returned from /feynman/start.
# value: {goal_id, concept, level, exchanges: list[FeynmanExchange], pending_question, turns}
_feynman_sessions: dict[str, dict] = {}
MAX_FEYNMAN_TURNS = 6
_adaptive_sessions: dict[str, dict] = {}
_socratic_sessions: dict[str, dict] = {}
MAX_SOCRATIC_TURNS = 8


# ---------- Goals & Plans ----------

@router.post("/goals")
async def create_goal(req: CreateGoalRequest):
    goal_id = db.create_goal(req.text, req.level, req.deadline_days, req.user_id)
    try:
        plan = await agents.generate_plan(req.text, req.level, req.deadline_days, provider=req.provider)
    except Exception as e:
        raise HTTPException(502, f"planner failed: {e}")
    plan_id = db.save_plan(goal_id, plan)
    return {"goal_id": goal_id, "plan_id": plan_id, "summary": plan.summary, "total_days": plan.total_days}


def _sse(event_type: str, payload: dict) -> str:
    return f"data: {json.dumps({'type': event_type, **payload})}\n\n"


@router.post("/goals/stream")
async def create_goal_stream(req: CreateGoalRequest):
    """SSE: streams the planner agent's tokens live, then saves & emits goal_id."""
    goal_id = db.create_goal(req.text, req.level, req.deadline_days, req.user_id)

    async def gen():
        try:
            yield _sse("start", {"goal_id": goal_id, "provider": req.provider or "default"})
            async for kind, value in agents.stream_plan(
                req.text, req.level, req.deadline_days, provider=req.provider,
            ):
                if kind == "token":
                    yield _sse("token", {"content": value})
                elif kind == "done":
                    plan_id = db.save_plan(goal_id, value)
                    yield _sse("done", {
                        "goal_id": goal_id,
                        "plan_id": plan_id,
                        "summary": value.summary,
                        "total_days": value.total_days,
                    })
        except Exception as e:
            yield _sse("error", {"message": str(e)[:500]})

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/goals/{goal_id}/refine/stream")
async def refine_plan_stream(goal_id: int, req: RefinePlanRequest):
    """SSE: streams a refined plan; saves as a new version on completion."""
    plan_row = db.get_active_plan(goal_id)
    if not plan_row:
        raise HTTPException(404, "no active plan for this goal")
    days = db.get_plan_days(plan_row.id)

    # Reconstruct a Plan instance from DB rows so the agent gets full context
    prev_plan = Plan(
        reasoning=plan_row.reasoning,
        self_check=plan_row.self_check,
        reasoning_types=plan_row.reasoning_types,
        caveats=plan_row.caveats,
        goal=db.get_goal(goal_id).text,
        summary=plan_row.summary,
        total_days=plan_row.total_days,
        days=[d.__class__.__mro__[0](**d.model_dump()) for d in days]
        if False else _plan_days_to_pydantic(days),
    )

    async def gen():
        try:
            yield _sse("start", {"goal_id": goal_id, "provider": req.provider or "default"})
            async for kind, value in agents.stream_refine_plan(
                prev_plan, req.instruction, provider=req.provider,
            ):
                if kind == "token":
                    yield _sse("token", {"content": value})
                elif kind == "done":
                    new_plan_id = db.save_plan(goal_id, value)
                    yield _sse("done", {
                        "goal_id": goal_id,
                        "plan_id": new_plan_id,
                        "summary": value.summary,
                        "total_days": value.total_days,
                    })
        except Exception as e:
            yield _sse("error", {"message": str(e)[:500]})

    return StreamingResponse(gen(), media_type="text/event-stream")


def _plan_days_to_pydantic(rows) -> list:
    """Convert PlanDayRow list to PlanDay list (drop DB-only fields)."""
    from praxis.learning.models import PlanDay
    return [
        PlanDay(
            day_num=r.day_num, topic=r.topic, objective=r.objective,
            concepts=r.concepts, activities=r.activities,
            suggested_resources=r.suggested_resources,
            checkpoint=r.checkpoint, success_criteria=r.success_criteria,
            estimated_minutes=r.estimated_minutes, difficulty=r.difficulty,
        )
        for r in rows
    ]


@router.get("/providers")
async def list_available_providers(request: Request):
    """Returns the providers configured + the agent-specific defaults from settings."""
    from praxis.config import settings
    rt = request.app.state.router
    return {
        "providers": list(rt.providers.keys()),
        "order": rt.order,
        "planner_default": settings.planner_provider,
        "checkin_default": settings.checkin_provider,
        "grader_default": settings.grader_provider,
    }


@router.get("/goals")
async def list_goals(user_id: int | None = None):
    goals = db.list_goals(user_id)
    return [g.model_dump(mode="json") for g in goals]


@router.delete("/goals/{goal_id}")
async def delete_goal(goal_id: int):
    if not db.delete_goal(goal_id):
        raise HTTPException(404, "goal not found")
    return {"deleted": goal_id}


@router.get("/goals/{goal_id}/plan")
async def get_plan(goal_id: int):
    plan = db.get_active_plan(goal_id)
    if not plan:
        raise HTTPException(404, "no active plan")
    days = db.get_plan_days(plan.id)
    return {
        "plan": plan.model_dump(mode="json"),
        "days": [d.model_dump(mode="json") for d in days],
        "mastery": [m.model_dump(mode="json") for m in db.get_mastery(goal_id)],
    }


# ---------- Check-in flow ----------

@router.post("/days/{plan_day_id}/checkin/start")
async def start_check_in(plan_day_id: int):
    """Generate the question set for this day's check-in (cached in-process)."""
    if plan_day_id in _active_checkins:
        return _active_checkins[plan_day_id].model_dump(mode="json")

    day = db.get_plan_day(plan_day_id)
    if not day:
        raise HTTPException(404, "plan day not found")

    with db.conn() as c:
        plan_row = c.execute("SELECT goal_id FROM plans WHERE id=?", (day.plan_id,)).fetchone()
    if not plan_row:
        raise HTTPException(404, "plan not found")
    goal_id = plan_row["goal_id"]

    mastery_rows = db.get_mastery(goal_id)
    mastery_map = {m.concept: m.score for m in mastery_rows}

    try:
        plan = await agents.plan_check_in(
            topic=day.topic,
            objective=day.objective,
            concepts=day.concepts,
            mastery=mastery_map,
        )
    except Exception as e:
        raise HTTPException(502, f"check-in planner failed: {e}")

    _active_checkins[plan_day_id] = plan
    _active_answers[plan_day_id] = []
    db.set_plan_day_status(plan_day_id, "in_progress")
    return plan.model_dump(mode="json")


@router.post("/days/{plan_day_id}/checkin/answer")
async def submit_answer(plan_day_id: int, req: SubmitAnswerRequest):
    """Grade one answer. Caller submits one question at a time."""
    plan = _active_checkins.get(plan_day_id)
    if not plan:
        raise HTTPException(400, "no active check-in for this day — call /start first")
    if req.question_index >= len(plan.questions):
        raise HTTPException(400, "question_index out of range")

    question = plan.questions[req.question_index]
    try:
        grade = await agents.grade_answer(
            question_prompt=question.prompt,
            concept=question.concept,
            kind=question.kind,
            answer_text=req.answer_text,
            expected_elements=question.expected_elements,
        )
    except Exception as e:
        raise HTTPException(502, f"grader failed: {e}")

    # Pull goal_id for mastery write
    day = db.get_plan_day(plan_day_id)
    with db.conn() as c:
        plan_row = c.execute("SELECT goal_id FROM plans WHERE id=?", (day.plan_id,)).fetchone()
    goal_id = plan_row["goal_id"]

    db.update_mastery(goal_id, question.concept, grade.score)
    for gap in grade.gaps:
        db.update_mastery(goal_id, gap, max(0.0, grade.score - 0.2))
    auto_replan = db.maybe_auto_replan(goal_id)

    answer = CheckInAnswer(question=question, answer_text=req.answer_text, grade=grade)
    _active_answers[plan_day_id].append(answer)

    is_last = req.question_index == len(plan.questions) - 1
    result_dump = None
    if is_last:
        answers = _active_answers[plan_day_id]
        overall = sum(a.grade.score for a in answers) / len(answers) if answers else 0.0
        next_focus: list[str] = []
        for a in answers:
            if a.grade.score < 0.7:
                next_focus.append(a.question.concept)
            next_focus.extend(a.grade.gaps)
        next_focus = list(dict.fromkeys(next_focus))[:6]
        result = CheckInResult(
            plan_day_id=plan_day_id,
            answers=answers,
            overall_score=overall,
            next_focus=next_focus,
        )
        db.save_check_in(plan_day_id, result)
        db.set_plan_day_status(plan_day_id, "done")
        _active_checkins.pop(plan_day_id, None)
        _active_answers.pop(plan_day_id, None)
        result_dump = result.model_dump(mode="json")

    return {
        "grade": grade.model_dump(mode="json"),
        "is_last": is_last,
        "result": result_dump,
        "auto_replan": auto_replan,
    }


@router.get("/days/{plan_day_id}/checkin/latest")
async def latest_result(plan_day_id: int):
    result = db.latest_check_in(plan_day_id)
    if not result:
        raise HTTPException(404, "no completed check-in yet")
    return result.model_dump(mode="json")


# ---------- Feynman checks (Mode 2) ----------

def _feynman_opening(concept: str) -> str:
    return (
        f"In your own words, explain **{concept}** as if you were teaching it to a curious "
        f"friend who has never heard of it. Cover not just what it is, but how and why it works."
    )


@router.post("/feynman/start")
async def feynman_start(req: StartFeynmanRequest):
    """Open a Feynman session for one concept. Returns a session token + the opening question."""
    goal = db.get_goal(req.goal_id)
    if not goal:
        raise HTTPException(404, "goal not found")

    token = uuid.uuid4().hex
    question = _feynman_opening(req.concept)
    _feynman_sessions[token] = {
        "goal_id": req.goal_id,
        "concept": req.concept,
        "level": goal.level,
        "exchanges": [],
        "pending_question": question,
        "turns": 0,
    }
    return {"session": token, "concept": req.concept, "question": question}


@router.post("/feynman/reply")
async def feynman_reply(req: FeynmanReplyRequest):
    """Submit one explanation. The tutor evaluates and either probes again or marks it solid."""
    state = _feynman_sessions.get(req.session)
    if not state:
        raise HTTPException(400, "no active Feynman session — call /feynman/start first")

    try:
        turn = await agents.feynman_turn(
            concept=state["concept"],
            level=state["level"],
            exchanges=state["exchanges"],
            current_question=state["pending_question"],
            current_answer=req.answer_text,
        )
    except Exception as e:
        raise HTTPException(502, f"feynman tutor failed: {e}")

    state["exchanges"].append(
        FeynmanExchange(question=state["pending_question"], answer=req.answer_text, turn=turn)
    )
    state["turns"] += 1

    hit_cap = state["turns"] >= MAX_FEYNMAN_TURNS
    done = turn.verdict == "solid" or hit_cap

    if not done:
        # Probe again. Fall back to a generic nudge if the model left follow_up empty.
        next_q = turn.follow_up.strip() or (
            f"Can you go one level deeper on the part you're least sure about in {state['concept']}?"
        )
        state["pending_question"] = next_q
        return {
            "turn": turn.model_dump(mode="json"),
            "done": False,
            "question": next_q,
            "turns": state["turns"],
            "max_turns": MAX_FEYNMAN_TURNS,
        }

    # Session complete — persist, update mastery, drop in-process state.
    result = FeynmanResult(
        goal_id=state["goal_id"],
        concept=state["concept"],
        exchanges=state["exchanges"],
        final_understanding=turn.understanding,
        solved=turn.verdict == "solid",
    )
    db.save_feynman(state["goal_id"], result)
    db.update_mastery(state["goal_id"], state["concept"], turn.understanding)
    auto_replan = db.maybe_auto_replan(state["goal_id"])
    _feynman_sessions.pop(req.session, None)
    return {
        "turn": turn.model_dump(mode="json"),
        "done": True,
        "result": result.model_dump(mode="json"),
        "turns": state["turns"],
        "max_turns": MAX_FEYNMAN_TURNS,
        "auto_replan": auto_replan,
    }


@router.get("/goals/{goal_id}/feynman/latest")
async def feynman_latest(goal_id: int, concept: str):
    """Most recent completed Feynman session for a concept (query param `concept`)."""
    result = db.latest_feynman(goal_id, concept)
    if not result:
        raise HTTPException(404, "no completed Feynman session for this concept yet")
    return result.model_dump(mode="json")


# ---------- Resource verification (web-search tool) ----------

@router.post("/days/{plan_day_id}/resources/verify")
async def verify_day_resources(plan_day_id: int):
    """Run a real web search for this day's suggested resources, then have the verifier
    agent confirm which exist and attach real URLs. Saves and returns the result."""
    day = db.get_plan_day(plan_day_id)
    if not day:
        raise HTTPException(404, "plan day not found")

    # Build search queries: each suggested resource, plus a topic-level fallback.
    suggested = day.suggested_resources or []
    queries = [s for s in suggested][:6]
    if not queries:
        queries = [f"{day.topic} {c}" for c in day.concepts[:3]] or [day.topic]
    # Always add one authoritative topic query to surface resources the planner missed.
    queries.append(f"best resource to learn {day.topic}")

    try:
        hits_by_query = await search_many(queries, per_query=4)
    except Exception as e:
        raise HTTPException(502, f"web search failed: {e}")

    try:
        check = await agents.verify_resources(
            topic=day.topic,
            concepts=day.concepts,
            suggested=suggested,
            hits_by_query=hits_by_query,
        )
    except Exception as e:
        raise HTTPException(502, f"resource verifier failed: {e}")

    db.save_resources(plan_day_id, check)
    return check.model_dump(mode="json")


@router.get("/days/{plan_day_id}/resources")
async def get_day_resources(plan_day_id: int):
    check = db.get_resources(plan_day_id)
    if not check:
        raise HTTPException(404, "no verified resources yet for this day")
    return check.model_dump(mode="json")


# ---------- Users and shared concept graph ----------

@router.get("/users")
async def list_users():
    return [u.model_dump(mode="json") for u in db.list_users()]


@router.post("/users")
async def create_user(req: CreateUserRequest):
    try:
        user_id = db.create_user(req.name)
    except Exception as e:
        raise HTTPException(409, f"could not create user: {e}")
    return {"id": user_id, "name": req.name.strip()}


@router.get("/goals/{goal_id}/concept-graph")
async def get_concept_graph(goal_id: int):
    if not db.get_goal(goal_id):
        raise HTTPException(404, "goal not found")
    return db.concept_graph(goal_id)


# ---------- Spaced-repetition flashcards (Mode 3) ----------

@router.post("/goals/{goal_id}/flashcards/generate")
async def generate_flashcards(goal_id: int):
    goal = db.get_goal(goal_id)
    if not goal:
        raise HTTPException(404, "goal not found")
    plan = db.get_active_plan(goal_id)
    if not plan:
        raise HTTPException(400, "this goal has no active plan")
    days = db.get_plan_days(plan.id)
    mastery = {m.concept: m.score for m in db.get_mastery(goal_id)}
    context = [
        {
            "day": day.day_num,
            "topic": day.topic,
            "objective": day.objective,
            "concepts": day.concepts,
            "checkpoint": day.checkpoint,
        }
        for day in days
    ]
    try:
        deck = await agents.generate_flashcards(goal.text, goal.level, context, mastery)
    except Exception as e:
        raise HTTPException(502, f"flashcard designer failed: {e}")
    return {"created": db.save_flashcards(goal_id, deck.cards), "rationale": deck.rationale}


@router.get("/goals/{goal_id}/flashcards")
async def get_flashcards(goal_id: int, due_only: bool = False, limit: int = 50):
    cards = db.due_flashcards(goal_id, limit) if due_only else db.all_flashcards(goal_id)
    return [c.model_dump(mode="json") for c in cards]


@router.post("/flashcards/{card_id}/review")
async def review_flashcard(card_id: int, req: ReviewFlashcardRequest):
    card = db.review_flashcard(card_id, req.rating)
    if not card:
        raise HTTPException(404, "flashcard not found")
    replan = db.maybe_auto_replan(card.goal_id)
    return {"card": card.model_dump(mode="json"), "auto_replan": replan}


# ---------- Adaptive quiz (Mode 4) ----------

@router.post("/adaptive/start")
async def adaptive_start(req: AdaptiveQuizRequest):
    goal = db.get_goal(req.goal_id)
    if not goal:
        raise HTTPException(404, "goal not found")
    graph = db.concept_graph(req.goal_id)
    concepts = [n["concept"] for n in graph["nodes"]]
    mastery = {n["concept"]: n["mastery"] for n in graph["nodes"]}
    prereqs = {n["concept"]: n["prerequisites"] for n in graph["nodes"]}
    questions = agents.build_adaptive_questions(concepts, mastery, prereqs, req.count)
    if not questions:
        raise HTTPException(400, "this goal has no concepts yet")
    token = uuid.uuid4().hex
    _adaptive_sessions[token] = {
        "goal_id": req.goal_id,
        "questions": questions,
        "answers": [],
    }
    return {
        "session": token,
        "questions": [q.model_dump(mode="json") for q in questions],
        "strategy": "Questions progress from recall to application and edge cases based on mastery.",
    }


@router.post("/adaptive/answer")
async def adaptive_answer(req: AdaptiveQuizAnswerRequest):
    state = _adaptive_sessions.get(req.session)
    if not state:
        raise HTTPException(400, "no active adaptive quiz")
    questions = state["questions"]
    if req.question_index >= len(questions):
        raise HTTPException(400, "question_index out of range")
    if req.question_index != len(state["answers"]):
        raise HTTPException(400, "questions must be answered once and in order")
    question = questions[req.question_index]
    try:
        grade = await agents.grade_answer(
            question.prompt, question.concept, question.kind, req.answer_text,
            question.expected_elements,
        )
    except Exception as e:
        raise HTTPException(502, f"grader failed: {e}")
    db.update_mastery(state["goal_id"], question.concept, grade.score)
    state["answers"].append({"question": question.model_dump(), "grade": grade.model_dump()})
    done = len(state["answers"]) >= len(questions)
    replan = db.maybe_auto_replan(state["goal_id"])
    if done:
        _adaptive_sessions.pop(req.session, None)
    return {
        "grade": grade.model_dump(mode="json"),
        "done": done,
        "answered": len(state["answers"]),
        "total": len(questions),
        "auto_replan": replan,
    }


# ---------- Socratic explainer (Mode 5) ----------

@router.post("/socratic/start")
async def socratic_start(req: SocraticStartRequest):
    goal = db.get_goal(req.goal_id)
    if not goal:
        raise HTTPException(404, "goal not found")
    token = uuid.uuid4().hex
    question = (
        f"Before defining it formally, what problem do you think {req.concept} is meant to solve?"
    )
    _socratic_sessions[token] = {
        "goal_id": req.goal_id, "level": goal.level, "concept": req.concept,
        "transcript": [{"role": "tutor", "text": question}], "turns": 0,
    }
    return {"session": token, "question": question, "max_turns": MAX_SOCRATIC_TURNS}


@router.post("/socratic/reply")
async def socratic_reply(req: SocraticReplyRequest):
    state = _socratic_sessions.get(req.session)
    if not state:
        raise HTTPException(400, "no active Socratic session")
    state["transcript"].append({"role": "learner", "text": req.answer_text})
    try:
        turn = await agents.socratic_turn(
            state["concept"], state["level"], state["transcript"], req.answer_text,
        )
    except Exception as e:
        raise HTTPException(502, f"Socratic tutor failed: {e}")
    state["turns"] += 1
    done = turn.complete or state["turns"] >= MAX_SOCRATIC_TURNS
    state["transcript"].append({
        "role": "tutor", "text": turn.question, "feedback": turn.feedback,
        "understanding": turn.understanding,
    })
    if done:
        db.update_mastery(state["goal_id"], state["concept"], turn.understanding)
        replan = db.maybe_auto_replan(state["goal_id"])
        _socratic_sessions.pop(req.session, None)
    else:
        replan = None
    return {
        "turn": turn.model_dump(mode="json"), "done": done,
        "turns": state["turns"], "max_turns": MAX_SOCRATIC_TURNS,
        "auto_replan": replan,
    }


# ---------- File/note ingestion ----------

def _extract_note_concepts(content: str, known: list[str]) -> list[str]:
    lowered = content.lower()
    matched = [c for c in known if c.lower() in lowered]
    headings = re.findall(r"(?m)^#{1,4}\s+(.{2,80})$", content)
    emphasized = re.findall(r"\*\*([^*\n]{2,80})\*\*", content)
    candidates = matched + headings + emphasized
    clean: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        value = re.sub(r"\s+", " ", value).strip(" .:#*-")
        key = value.lower()
        if 2 <= len(value) <= 80 and key not in seen:
            seen.add(key)
            clean.append(value)
    return clean[:30]


@router.post("/goals/{goal_id}/notes")
async def ingest_note(
    goal_id: int,
    file: UploadFile = File(...),
):
    if not db.get_goal(goal_id):
        raise HTTPException(404, "goal not found")
    raw = await file.read()
    if len(raw) > 2_000_000:
        raise HTTPException(413, "file is too large; maximum is 2 MB")
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(415, "only UTF-8 text, Markdown, CSV, and JSON files are supported")
    graph = db.concept_graph(goal_id)
    known = [n["concept"] for n in graph["nodes"]]
    concepts = _extract_note_concepts(content, known)
    note_id = db.save_note(goal_id, file.filename or "note.txt", content, concepts)
    return {"id": note_id, "filename": file.filename, "concepts": concepts}


@router.get("/goals/{goal_id}/notes")
async def get_notes(goal_id: int):
    return [n.model_dump(mode="json") for n in db.list_notes(goal_id)]


@router.post("/goals/{goal_id}/auto-replan")
async def auto_replan(goal_id: int):
    if not db.get_goal(goal_id):
        raise HTTPException(404, "goal not found")
    return {"change": db.maybe_auto_replan(goal_id)}


@router.get("/goals/{goal_id}/replan-events")
async def get_replan_events(goal_id: int):
    return db.replan_events(goal_id)


# ---------- Interactive roadmap ----------

@router.get("/goals/{goal_id}/roadmap")
async def get_roadmap(goal_id: int):
    if not db.get_goal(goal_id):
        raise HTTPException(404, "goal not found")
    return db.roadmap(goal_id)


@router.post("/goals/{goal_id}/targets")
async def create_roadmap_target(goal_id: int, req: CreateRoadmapTargetRequest):
    if not db.get_goal(goal_id):
        raise HTTPException(404, "goal not found")
    return db.create_roadmap_target(
        goal_id, req.title, req.description, req.target_type, req.target_value, req.concept,
    )


@router.patch("/targets/{target_id}")
async def update_roadmap_target(target_id: int, req: UpdateRoadmapTargetRequest):
    target = db.update_roadmap_target(target_id, req.completed)
    if not target:
        raise HTTPException(404, "target not found")
    return target


@router.delete("/targets/{target_id}")
async def delete_roadmap_target(target_id: int):
    if not db.delete_roadmap_target(target_id):
        raise HTTPException(404, "target not found")
    return {"deleted": target_id}
