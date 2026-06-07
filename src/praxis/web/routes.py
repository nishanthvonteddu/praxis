"""Web UI routes — server-rendered HTML pages."""
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from praxis.learning import agents, db


ROOT = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(ROOT / "templates"))

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    users = db.list_users()
    requested_user = request.query_params.get("user_id")
    user_id = int(requested_user) if requested_user and requested_user.isdigit() else 1
    if not any(u.id == user_id for u in users):
        user_id = 1
    goals = db.list_goals(user_id)
    rt = request.app.state.router
    from praxis.config import settings
    return templates.TemplateResponse(
        request, "index.html",
        {
            "goals": goals,
            "users": users,
            "active_user_id": user_id,
            "providers": list(rt.providers.keys()),
            "planner_default": settings.planner_provider,
        },
    )


@router.post("/goals/new")
async def create_goal_form(
    text: str = Form(...),
    level: str = Form("beginner"),
    deadline_days: int = Form(...),
    user_id: int = Form(1),
):
    if len(text.strip()) < 8:
        raise HTTPException(400, "Goal text too short (min 8 chars).")
    goal_id = db.create_goal(text.strip(), level, deadline_days, user_id)
    try:
        plan = await agents.generate_plan(text.strip(), level, deadline_days)
    except Exception as e:
        raise HTTPException(502, f"planner failed: {e}")
    db.save_plan(goal_id, plan)
    return RedirectResponse(url=f"/goals/{goal_id}", status_code=303)


@router.get("/goals/{goal_id}", response_class=HTMLResponse)
async def view_goal(request: Request, goal_id: int):
    goal = db.get_goal(goal_id)
    if not goal:
        raise HTTPException(404, "goal not found")
    plan = db.get_active_plan(goal_id)
    days = db.get_plan_days(plan.id) if plan else []
    mastery = db.get_mastery(goal_id)
    roadmap = db.roadmap(goal_id)
    current_day_num = roadmap.get("current_day")
    current_day = next((day for day in days if day.day_num == current_day_num), None)
    if not current_day and days:
        current_day = days[-1]
    due_cards = db.due_flashcards(goal_id, 20)
    return templates.TemplateResponse(
        request, "plan.html",
        {
            "goal": goal,
            "plan": plan,
            "days": days,
            "mastery": mastery,
            "roadmap": roadmap,
            "current_day": current_day,
            "due_count": len(due_cards),
        },
    )


@router.get("/days/{plan_day_id}", response_class=HTMLResponse)
async def view_day(request: Request, plan_day_id: int):
    day = db.get_plan_day(plan_day_id)
    if not day:
        raise HTTPException(404, "day not found")
    with db.conn() as c:
        plan_row = c.execute("SELECT goal_id FROM plans WHERE id=?", (day.plan_id,)).fetchone()
    goal_id = plan_row["goal_id"]
    goal = db.get_goal(goal_id)
    latest = db.latest_check_in(plan_day_id)
    resources = db.get_resources(plan_day_id)
    return templates.TemplateResponse(
        request, "day.html",
        {"day": day, "goal": goal, "latest": latest, "resources": resources},
    )


@router.get("/goals/{goal_id}/feynman", response_class=HTMLResponse)
async def feynman_page(request: Request, goal_id: int, concept: str):
    goal = db.get_goal(goal_id)
    if not goal:
        raise HTTPException(404, "goal not found")
    latest = db.latest_feynman(goal_id, concept)
    return templates.TemplateResponse(
        request, "feynman.html",
        {"goal": goal, "concept": concept, "latest": latest},
    )


@router.get("/goals/{goal_id}/lab", response_class=HTMLResponse)
async def learning_lab(request: Request, goal_id: int):
    goal = db.get_goal(goal_id)
    if not goal:
        raise HTTPException(404, "goal not found")
    graph = db.concept_graph(goal_id)
    cards = db.due_flashcards(goal_id, 20)
    notes = db.list_notes(goal_id)
    events = db.replan_events(goal_id)
    return templates.TemplateResponse(
        request, "lab.html",
        {
            "goal": goal,
            "graph": graph,
            "cards": cards,
            "notes": notes,
            "events": events,
        },
    )


@router.get("/goals/{goal_id}/roadmap", response_class=HTMLResponse)
async def roadmap_page(request: Request, goal_id: int):
    goal = db.get_goal(goal_id)
    if not goal:
        raise HTTPException(404, "goal not found")
    roadmap = db.roadmap(goal_id)
    return templates.TemplateResponse(
        request, "roadmap.html",
        {"goal": goal, "roadmap": roadmap},
    )


@router.get("/status", response_class=HTMLResponse)
async def status_page(request: Request):
    return templates.TemplateResponse(request, "status.html", {})
