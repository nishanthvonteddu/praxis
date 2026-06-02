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
    goals = db.list_goals()
    rt = request.app.state.router
    from praxis.config import settings
    return templates.TemplateResponse(
        request, "index.html",
        {
            "goals": goals,
            "providers": list(rt.providers.keys()),
            "planner_default": settings.planner_provider,
        },
    )


@router.post("/goals/new")
async def create_goal_form(
    text: str = Form(...),
    level: str = Form("beginner"),
    deadline_days: int = Form(...),
):
    if len(text.strip()) < 8:
        raise HTTPException(400, "Goal text too short (min 8 chars).")
    goal_id = db.create_goal(text.strip(), level, deadline_days)
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
    return templates.TemplateResponse(
        request, "plan.html",
        {"goal": goal, "plan": plan, "days": days, "mastery": mastery},
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
    return templates.TemplateResponse(
        request, "day.html",
        {"day": day, "goal": goal, "latest": latest},
    )


@router.get("/status", response_class=HTMLResponse)
async def status_page(request: Request):
    return templates.TemplateResponse(request, "status.html", {})
