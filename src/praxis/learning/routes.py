"""Learning API: goals, plans, check-ins, mastery.

Two response styles per route:
  - JSON for HTMX-less / programmatic callers
  - HTML fragment when `HX-Request: true` is set (HTMX swaps it in)
"""
import json
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from praxis.learning import agents, db
from praxis.learning.search import search_many
from praxis.learning.models import (
    CheckInAnswer,
    CheckInPlan,
    CheckInResult,
    CreateGoalRequest,
    FeynmanExchange,
    FeynmanReplyRequest,
    FeynmanResult,
    Plan,
    RefinePlanRequest,
    StartFeynmanRequest,
    SubmitAnswerRequest,
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


# ---------- Goals & Plans ----------

@router.post("/goals")
async def create_goal(req: CreateGoalRequest):
    goal_id = db.create_goal(req.text, req.level, req.deadline_days)
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
    goal_id = db.create_goal(req.text, req.level, req.deadline_days)

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
async def list_goals():
    goals = db.list_goals()
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
    _feynman_sessions.pop(req.session, None)
    return {
        "turn": turn.model_dump(mode="json"),
        "done": True,
        "result": result.model_dump(mode="json"),
        "turns": state["turns"],
        "max_turns": MAX_FEYNMAN_TURNS,
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
