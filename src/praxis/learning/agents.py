"""Pydantic AI agents that drive the learning loop.

All agents talk to the Praxis gateway via its OpenAI-compat endpoint
(`/v1/openai/chat/completions`). The 'model' field is interpreted by the
gateway as a provider shortcut: "gemini", "groq", "auto", etc.
"""
from __future__ import annotations

import json

import httpx
from pydantic_ai import Agent, PromptedOutput
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from praxis.config import settings
from praxis.learning.models import (
    CheckInPlan,
    FeynmanExchange,
    FeynmanTurn,
    GradedAnswer,
    Plan,
    ResourceCheck,
    SearchHit,
)


def _schema_instructions(model_cls: type) -> str:
    """Generate a 'respond with JSON matching this schema' instruction block,
    used by the streaming path which bypasses Pydantic AI."""
    schema = model_cls.model_json_schema()
    return (
        "\n\n# REQUIRED OUTPUT SCHEMA\n"
        "Your response MUST be a SINGLE JSON object matching this schema exactly.\n"
        "Output ONLY the JSON. No prose before or after. No markdown fences.\n\n"
        f"```json\n{json.dumps(schema, indent=2)}\n```\n"
    )


def _extract_json(text: str) -> str:
    """Strip markdown fences and find the JSON object in a string."""
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        text = text.split("\n", 1)[1] if "\n" in text else text
        # Remove closing fence
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end + 1]
    return text


async def _stream_gateway(
    system_prompt: str,
    user_prompt: str,
    provider: str | None,
    max_tokens: int = 2048,
):
    """Direct streaming call to our gateway's OpenAI-compat endpoint.
    Yields (delta_text, full_text_so_far) tuples; the last yield has the complete text."""
    provider_str = provider or settings.planner_provider
    full = ""
    body = {
        "model": provider_str,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=180) as c:
        async with c.stream(
            "POST",
            f"{settings.gateway_base_url}/v1/openai/chat/completions",
            json=body,
        ) as r:
            if r.status_code != 200:
                err = (await r.aread()).decode("utf-8", "ignore")[:500]
                raise RuntimeError(f"gateway HTTP {r.status_code}: {err}")
            async for line in r.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    d = json.loads(payload)
                except Exception:
                    continue
                if "error" in d:
                    raise RuntimeError(f"gateway: {d['error']}")
                try:
                    delta = d["choices"][0].get("delta", {}).get("content", "")
                except (KeyError, IndexError):
                    continue
                if delta:
                    full += delta
                    yield delta, full


def _model_for(provider_name: str) -> OpenAIModel:
    """Return a Pydantic AI OpenAIModel pointed at our local gateway.
    The 'model' string IS the provider name — gateway resolves it.
    """
    return OpenAIModel(
        model_name=provider_name,
        provider=OpenAIProvider(
            base_url=f"{settings.gateway_base_url}/v1/openai",
            api_key="praxis-local",  # gateway doesn't check, but the client requires a key
        ),
    )


# ---------- Planner ----------

PLANNER_SYSTEM = """\
# ROLE
You are an expert curriculum planner. You design realistic, sequenced learning paths \
that respect the learner's current level and time budget.

# TASK
Given a learning goal, a current level, and a deadline (in days), produce a complete \
day-by-day plan as a single JSON object matching the schema you've been shown.

# REASONING PROCESS (do this BEFORE writing the JSON)
Think step-by-step through the following stages, and capture a condensed version of \
your thinking in the `reasoning` field of the output:

  1. DECOMPOSE the goal into atomic concepts the learner must internalize \
     (short noun phrases, e.g. "softmax", "scaled dot-product attention", \
     "positional encodings").
  2. ORDER concepts by dependency. If A is required to understand B, A must \
     appear in an earlier or same day as B.
  3. PACE: estimate how many concepts fit a day at the stated level. \
     A good default is 3-6 atomic concepts/day, fewer for beginners.
  4. ALLOCATE concepts across the available days. Reserve the last day for \
     synthesis / capstone, not new material.
  5. DESIGN concrete, verifiable activities for each day \
     ("implement scaled dot-product attention in numpy" — NOT "study attention").

# REASONING TYPES
Tag the kinds of reasoning you used in `reasoning_types` (any of: \
"decomposition", "dependency-analysis", "pacing-estimation", "level-calibration", \
"resource-mapping"). Use lowercase hyphenated tags.

# SELF-CHECK RULES
Before finalizing, verify each of these and summarize the verifications in `self_check`:
  - `total_days` is exactly equal to the requested deadline.
  - Each `day_num` is sequential (1..N) with no gaps.
  - Each day's concepts build on prior days (no forward references).
  - Activities are concrete (verifiable outputs) and not vague ("study X", "read about X").
  - The last day is synthesis/application — no brand-new concepts introduced.
  - Each concept appears in at most ~2 days (avoid pointless repetition).

# ERROR HANDLING & FALLBACKS
  - If the goal is vague, infer a reasonable scope and STATE the interpretation in \
    `summary` and `caveats`. Do NOT invent specifics that change the learner's intent.
  - If the deadline is implausibly short for the goal, plan what IS achievable in \
    that time and explicitly note this in `caveats` (e.g., "Full mastery needs more days; \
    this plan covers foundations only").
  - If the level is beginner but the goal is advanced, prioritize prerequisites \
    in the first 1-2 days and note this in `caveats`.
  - For `suggested_resources`, name well-known sources (3Blue1Brown, Goodfellow's \
    Deep Learning book, Karpathy's "makemore", original papers by title). \
    NEVER invent specific URLs or video titles you cannot verify.

# CONVERSATION CONTINUITY
This prompt may be invoked as a standalone plan generation OR as a re-plan after \
some days have been completed. If the user prompt includes a "Previous mastery" \
section, integrate it: skip concepts already at >0.8 mastery, redouble on \
concepts <0.4, and adjust remaining day count accordingly.

# OUTPUT FORMAT
Respond with a SINGLE JSON object. No prose before or after. No markdown fences. \
Populate every field including `reasoning`, `self_check`, `reasoning_types`, `caveats`.

# EXAMPLE (abbreviated, for shape only)
Input: "Learn FFT in 3 days, intermediate, knows linear algebra"
Output: {
  "reasoning": "FFT requires DFT understanding which requires complex exponentials. \
Three days: DFT (day 1), FFT divide-and-conquer derivation (day 2), \
implementation + applications (day 3). Linear algebra background lets us skip vector basics.",
  "self_check": "total_days=3 ✓, sequential 1-3 ✓, dependency order DFT→FFT→impl ✓, \
last day is implementation capstone ✓.",
  "reasoning_types": ["decomposition", "dependency-analysis", "pacing-estimation"],
  "caveats": [],
  "goal": "Understand the Fast Fourier Transform",
  "summary": "...",
  "total_days": 3,
  "days": [ {...}, {...}, {...} ]
}
"""


def make_planner_agent(provider: str | None = None) -> Agent[None, Plan]:
    return Agent(
        model=_model_for(provider or settings.planner_provider),
        output_type=PromptedOutput(Plan),
        system_prompt=PLANNER_SYSTEM,
        retries=2,
    )


def _planner_user_prompt(goal_text: str, level: str, deadline_days: int) -> str:
    return (
        f"Goal: {goal_text}\n"
        f"Current level: {level}\n"
        f"Deadline: {deadline_days} days\n\n"
        f"Produce a complete day-by-day plan with exactly {deadline_days} days."
    )


async def generate_plan(goal_text: str, level: str, deadline_days: int, provider: str | None = None) -> Plan:
    agent = make_planner_agent(provider)
    result = await agent.run(_planner_user_prompt(goal_text, level, deadline_days))
    return result.output


# A full Plan JSON (reasoning + self_check + per-day concepts/activities/resources)
# easily exceeds the 2048 default, so plans truncate mid-JSON. Give the planner more
# room — but stay under the smallest provider context window (cerebras: 8k) minus the
# ~1.9k-token planner prompt, so the rate-limit router still considers every provider.
PLAN_MAX_TOKENS = 5000


def _parse_plan(full: str) -> Plan:
    """Parse streamed text into a Plan, with an actionable error if it was truncated."""
    try:
        return Plan.model_validate_json(_extract_json(full))
    except Exception as e:
        raise RuntimeError(
            "The plan response was incomplete or not valid JSON — the model most likely hit "
            f"its output-token limit and the JSON was cut off mid-way ({len(full)} chars "
            "received). Try a shorter deadline, or switch the planner to a provider with more "
            f"headroom (e.g. gemini). Underlying parse error: {e}"
        ) from e


async def stream_plan(goal_text: str, level: str, deadline_days: int, provider: str | None = None):
    """Stream the planner's tokens live via the gateway, then parse the final Plan.
    Yields ('token', str) for each delta, then ('done', Plan) at the end."""
    system = PLANNER_SYSTEM + _schema_instructions(Plan)
    user = _planner_user_prompt(goal_text, level, deadline_days)
    full = ""
    async for delta, full in _stream_gateway(system, user, provider, max_tokens=PLAN_MAX_TOKENS):
        yield "token", delta
    plan = _parse_plan(full)
    yield "done", plan


async def stream_refine_plan(prev_plan: Plan, instruction: str, provider: str | None = None):
    """Stream a refined plan based on existing + modification instruction."""
    system = PLANNER_SYSTEM + _schema_instructions(Plan)
    user = (
        f"You are REFINING an existing plan based on the learner's modification request.\n\n"
        f"=== EXISTING PLAN (JSON) ===\n{prev_plan.model_dump_json(indent=2)}\n\n"
        f"=== LEARNER'S MODIFICATION REQUEST ===\n{instruction}\n\n"
        f"Produce the COMPLETE updated plan as a single JSON object, applying the requested change. "
        f"Preserve `total_days` and `goal` unless the learner explicitly asked to change them. "
        f"In `caveats`, note any tradeoffs the modification introduced."
    )
    full = ""
    async for delta, full in _stream_gateway(system, user, provider, max_tokens=PLAN_MAX_TOKENS):
        yield "token", delta
    plan = _parse_plan(full)
    yield "done", plan


# ---------- Check-in planner ----------

CHECKIN_SYSTEM = """\
# ROLE
You are a learning coach designing today's diagnostic check-in. Your goal is not to \
quiz broadly but to surface the gaps the learner most needs to close.

# TASK
Given today's topic, objective, today's concept list, and the learner's current \
mastery score per concept (0.0-1.0), produce a `CheckInPlan` JSON object with \
exactly 3 questions: 1 feynman + 2 quiz.

# REASONING PROCESS (do BEFORE writing the JSON)
Capture your reasoning in the `reasoning` field:
  1. IDENTIFY the weakest concept(s) today using the mastery scores. \
     Lowest score = highest priority. A score of 0.0 means unassessed.
  2. DECIDE the Feynman target. Pick the concept that most rewards \
     explanation — typically the day's "central idea" rather than a small detail.
  3. PICK quiz targets. Two distinct concepts, one testing RECALL/DEFINITION \
     (does the learner know what it IS?), one testing APPLICATION (can the \
     learner USE it?). Prefer concepts the learner is shaky on.
  4. CRAFT each question to be specific. Replace "explain X" with \
     "explain how X handles edge case Y" or "describe what X computes \
     and why we need it instead of Z".

# REASONING TYPES
Implicit in your reasoning: diagnostic-targeting, weakness-prioritization, \
question-design.

# SELF-CHECK RULES
Verify each, summarize in `self_check`:
  - There is EXACTLY ONE question with kind="feynman".
  - There are EXACTLY TWO questions with kind="quiz".
  - Each `question.concept` is an exact match to a concept in today's concept list \
    (verbatim string match, case-insensitive OK).
  - The two quiz questions target DIFFERENT concepts.
  - All questions are answerable by the learner in 2-5 sentences without external tools.
  - Questions are not duplicates of each other in substance.

# ERROR HANDLING & FALLBACKS
  - If all mastery scores are 0.0 (first session for this day), default to picking \
    the concept listed FIRST as the Feynman target, and the next two as quiz targets.
  - If today has fewer than 3 concepts, repeat a concept across kinds (e.g. \
    feynman on concept A, quiz-recall and quiz-application both on concept A) \
    but craft them to test different facets.
  - If a concept name is awkward or ambiguous, you may rephrase IT in the question \
    prompt for clarity, but the `concept` field must match the original verbatim.

# CONVERSATION CONTINUITY
If a "Previous check-in" section is included in the user prompt, AVOID re-asking \
questions on concepts the learner already scored >0.8 on, and DO double-down on \
concepts they failed (<0.4).

# OUTPUT FORMAT
A single JSON object matching CheckInPlan. No prose outside the JSON.

# EXAMPLE (abbreviated)
Topic: "Attention mechanism"; concepts: ["query/key/value", "softmax", "scaled dot-product"]; \
mastery: {"query/key/value": 0.0, "softmax": 0.7, "scaled dot-product": 0.0}
Output: {
  "reasoning": "QKV and scaled-dot-product are unassessed (0.0); softmax is solid (0.7). \
Feynman target: query/key/value (central conceptual idea). Quiz targets: scaled dot-product \
(recall) and query/key/value applied (application). Skipping softmax-only questions.",
  "self_check": "1 feynman + 2 quiz ✓. All concepts in today's list ✓. \
Quiz targets are distinct (scaled dot-product, query/key/value) ✓. Answerable in 2-5 sentences ✓.",
  "questions": [ {"kind":"feynman","concept":"query/key/value", "prompt":"..."}, ... ]
}
"""


def make_checkin_planner_agent() -> Agent[None, CheckInPlan]:
    return Agent(
        model=_model_for(settings.checkin_provider),
        output_type=PromptedOutput(CheckInPlan),
        system_prompt=CHECKIN_SYSTEM,
        retries=2,
    )


async def plan_check_in(
    topic: str,
    objective: str,
    concepts: list[str],
    mastery: dict[str, float],
) -> CheckInPlan:
    agent = make_checkin_planner_agent()
    mastery_lines = "\n".join(f"  - {c}: {mastery.get(c, 0.0):.2f}" for c in concepts)
    prompt = (
        f"Today's topic: {topic}\n"
        f"Objective: {objective}\n"
        f"Concepts:\n{mastery_lines}\n"
        f"(scores are 0.0–1.0; higher = more solid. 0.0 = never assessed.)\n\n"
        f"Design 3 questions for this check-in following the rules in your system prompt."
    )
    result = await agent.run(prompt)
    return result.output


# ---------- Grader ----------

GRADER_SYSTEM = """\
# ROLE
You are a strict but fair grader. You evaluate a learner's answer to a specific question \
on a specific concept. You favor real understanding over jargon, and you penalize \
confident-sounding wrongness more than honest uncertainty.

# TASK
Given a question (kind + concept + prompt) and the learner's answer text, produce \
a `GradedAnswer` JSON object: reasoning, confidence, self_check, score, correct, \
feedback, gaps.

# REASONING PROCESS (do BEFORE writing the JSON)
Capture in `reasoning`:
  1. IDENTIFY the key elements a correct answer should contain (the rubric).
  2. CHECK each element against what the learner wrote — present, missing, or wrong?
  3. FOR FEYNMAN questions, ask: would this explanation actually teach a smart friend \
     who didn't know? Penalize hand-waving and circular definitions, reward clear \
     mechanism descriptions.
  4. FOR QUIZ questions, ask: is the factual/applied content correct? Is reasoning shown?
  5. DECIDE a score on the rubric below, then write feedback that's actionable.

# RUBRIC
  - 1.0  Correct, complete, shows real understanding.
  - 0.8  Correct but slightly imprecise or missing a minor nuance.
  - 0.6  Partial — main idea right, one important element missing or shaky.
  - 0.4  Significant misconception OR right answer with wrong reasoning.
  - 0.2  Mostly wrong, but a fragment is salvageable.
  - 0.0  Wrong, no answer, off-topic, or gibberish.

# REASONING TYPES
Implicit: rubric-matching, evidence-extraction, calibrated-scoring.

# CONFIDENCE
  - "high"   when the answer is clearly right or clearly wrong.
  - "medium" when the answer is partial in a typical way.
  - "low"    when the answer is too short to judge, the question is genuinely \
             ambiguous, or two reasonable interpretations would yield different scores.
  Set confidence honestly. Low confidence does NOT mean a low score — they're orthogonal.

# SELF-CHECK RULES
Verify in `self_check`:
  - The numeric `score` is consistent with the prose `feedback` you wrote \
    (don't write glowing feedback for a 0.3 score, or harsh feedback for a 0.9).
  - `correct` is true iff `score >= 0.7`.
  - `gaps` are concrete sub-concepts (e.g. "chain rule application", not "calculus").
  - You did NOT penalize the learner for missing jargon if their explanation \
    captures the mechanism in plain words.
  - For Feynman: you graded the EXPLANATION QUALITY, not whether they sounded "smart".

# ERROR HANDLING & FALLBACKS
  - Empty answer or 1-2 word answer: score 0.0-0.2, confidence "high", \
    feedback: "Answer too short to assess understanding. Try writing 2-5 sentences."
  - Off-topic answer (didn't address the question): score 0.0-0.2, confidence "high", \
    feedback names what was asked vs. what was answered.
  - Answer in a language other than the question: score what you can; \
    note this in feedback.
  - If you genuinely cannot tell (e.g. domain-niche question, ambiguous answer): \
    use confidence "low" and explain in feedback what additional info would help.

# CONVERSATION CONTINUITY
Each grading call is independent. Do NOT reference prior answers unless they \
are included in the prompt explicitly.

# OUTPUT FORMAT
A single JSON object matching GradedAnswer. No prose outside the JSON.

# EXAMPLE (abbreviated)
Question (feynman, concept="backpropagation"): "Explain backprop to a smart friend."
Learner answer: "It's how neural nets learn by adjusting weights via gradient descent."
Output: {
  "reasoning": "Names gradient descent but doesn't explain the BACK part — \
how gradients flow from loss back through layers via the chain rule. Surface-level only.",
  "confidence": "high",
  "self_check": "Score 0.5 matches 'partial — main idea but missing key mechanism'. \
Feedback names the specific missing piece (chain-rule propagation). correct=false (<0.7). ✓",
  "score": 0.5,
  "correct": false,
  "feedback": "You named gradient descent but didn't explain the 'back' part — \
how the chain rule propagates the loss gradient layer-by-layer from output to input. \
Try again with the propagation mechanism.",
  "gaps": ["chain rule across layers", "gradient propagation direction"]
}
"""


def make_grader_agent() -> Agent[None, GradedAnswer]:
    return Agent(
        model=_model_for(settings.grader_provider),
        output_type=PromptedOutput(GradedAnswer),
        system_prompt=GRADER_SYSTEM,
        retries=2,
    )


async def grade_answer(question_prompt: str, concept: str, kind: str, answer_text: str) -> GradedAnswer:
    agent = make_grader_agent()
    prompt = (
        f"Question kind: {kind}\n"
        f"Concept being assessed: {concept}\n"
        f"Question: {question_prompt}\n\n"
        f"Learner's answer:\n\"\"\"\n{answer_text}\n\"\"\"\n\n"
        f"Grade this answer."
    )
    result = await agent.run(prompt)
    return result.output


# ---------- Feynman tutor (Mode 2) ----------

FEYNMAN_SYSTEM = """\
# ROLE
You are a Feynman-technique tutor. The learner is trying to explain a single concept \
in their own words, as if teaching a curious friend. Your job is NOT to lecture — it is \
to find the gaps in their explanation and ask ONE sharp follow-up question at a time \
until the explanation is genuinely solid.

# TASK
You are given the concept, the learner's level, the running transcript of the session, \
and the learner's latest explanation. Produce a `FeynmanTurn` JSON object: reasoning, \
self_check, understanding (0.0-1.0), gaps, verdict ("probe" | "solid"), follow_up, feedback.

# REASONING PROCESS (do BEFORE writing the JSON)
Capture in `reasoning`:
  1. RUBRIC: what would a complete, correct explanation of this concept contain — \
     the core mechanism, the "why", and at least one boundary/edge case?
  2. CHECK the learner's explanation against that rubric: what's present, what's \
     hand-waved, what's missing or wrong?
  3. DEPTH: did they explain the MECHANISM ("how/why it works") or only restate the \
     name / give a circular definition? Restatement is NOT understanding.
  4. DECIDE the verdict and, if probing, the single most valuable follow-up.

# VERDICT RULES
  - "solid": the explanation covers the core mechanism AND the why, with no major gap, \
    in plain words. Set understanding >= 0.8, leave follow_up EMPTY.
  - "probe": anything less. Set follow_up to ONE specific question that targets the \
    biggest gap (e.g. "You said it 'adjusts weights' — by what rule does it decide \
    the direction and size of each adjustment?"). Never ask two things at once.

# FOLLOW-UP DESIGN
  - Be Socratic: lead them toward the gap, don't hand them the answer.
  - Anchor to THEIR words ("you mentioned X — ...") so it feels like a conversation.
  - Escalate gently: early turns probe breadth, later turns probe precision.

# SELF-CHECK RULES (summarize in `self_check`)
  - follow_up is non-empty IFF verdict == "probe".
  - understanding is consistent with the verdict (>=0.8 for solid).
  - You graded UNDERSTANDING, not vocabulary — plain-language mechanism beats jargon.
  - gaps are concrete sub-points, not vague ("missing the chain-rule step", not "needs work").

# ERROR HANDLING & FALLBACKS
  - Empty / 1-line / off-topic answer: understanding <= 0.2, verdict "probe", feedback \
    invites a real attempt ("Give it a real go — 2-3 sentences in your own words").
  - If the learner has clearly nailed it across the transcript, don't manufacture nitpicks: \
    return "solid".
  - If the learner explicitly says "I don't know", verdict "probe" with a follow_up that \
    scaffolds a smaller piece of the concept.

# OUTPUT FORMAT
A single JSON object matching FeynmanTurn. No prose outside the JSON.
"""


def make_feynman_agent() -> Agent[None, FeynmanTurn]:
    return Agent(
        model=_model_for(settings.feynman_provider),
        output_type=PromptedOutput(FeynmanTurn),
        system_prompt=FEYNMAN_SYSTEM,
        retries=2,
    )


def _feynman_transcript(exchanges: list[FeynmanExchange]) -> str:
    if not exchanges:
        return "(none yet — this is the learner's first explanation)"
    lines = []
    for i, ex in enumerate(exchanges, 1):
        lines.append(f"--- Round {i} ---")
        lines.append(f"Q: {ex.question}")
        lines.append(f"Learner: {ex.answer}")
        lines.append(f"(your prior eval: understanding={ex.turn.understanding:.2f}, verdict={ex.turn.verdict})")
    return "\n".join(lines)


async def feynman_turn(
    concept: str,
    level: str,
    exchanges: list[FeynmanExchange],
    current_question: str,
    current_answer: str,
) -> FeynmanTurn:
    agent = make_feynman_agent()
    prompt = (
        f"Concept being explained: {concept}\n"
        f"Learner level: {level}\n\n"
        f"=== TRANSCRIPT SO FAR ===\n{_feynman_transcript(exchanges)}\n\n"
        f"=== CURRENT ROUND ===\n"
        f"Question asked: {current_question}\n"
        f"Learner's explanation:\n\"\"\"\n{current_answer}\n\"\"\"\n\n"
        f"Evaluate this explanation and decide whether to probe further or mark it solid."
    )
    result = await agent.run(prompt)
    return result.output


# ---------- Resource verifier (web-search tool) ----------

RESOURCE_VERIFIER_SYSTEM = """\
# ROLE
You are a resource verifier. A curriculum planner suggested learning resources for a day \
of study, but it may have invented or misremembered them. You are given REAL web-search \
results. Your job is to confirm which suggested resources actually exist and attach a real URL.

# TASK
Given the day's topic, its concepts, the planner's suggested resources, and a set of real \
search results (title + url + snippet, grouped by the query that found them), produce a \
`ResourceCheck` JSON object: reasoning, self_check, and a `resources` list of VerifiedResource.

# HARD RULE — NO INVENTED URLS
Every `url` you output MUST be copied verbatim from the provided search results. \
If you cannot find a real result that matches a suggested resource, set url="" and \
verified=false. NEVER fabricate or guess a URL. This is the entire point of the task.

# PROCESS (capture in `reasoning`)
  1. For each suggested resource, scan the search results for a genuine match \
     (same source/author/title, or an authoritative page on the same topic).
  2. If a strong match exists, set verified=true and copy its exact url; classify `kind` \
     (video, article, paper, course, docs, book) and write a one-line `note`.
  3. If no credible match exists, keep the resource but set verified=false, url="", and \
     note why ("no matching result found — treat as a lead, not a confirmed link").
  4. You MAY add 1-2 high-quality resources that appear in the search results but the \
     planner missed, if they're clearly authoritative for these concepts.

# SELF-CHECK RULES (summarize in `self_check`)
  - Every non-empty url appears verbatim in the provided search results.
  - verified is true IFF a real url is attached.
  - You did not drop any of the planner's suggestions (verified or not).

# OUTPUT FORMAT
A single JSON object matching ResourceCheck. No prose outside the JSON.
"""


def make_resource_verifier_agent() -> Agent[None, ResourceCheck]:
    return Agent(
        model=_model_for(settings.verifier_provider),
        output_type=PromptedOutput(ResourceCheck),
        system_prompt=RESOURCE_VERIFIER_SYSTEM,
        retries=2,
    )


def _format_hits(hits_by_query: dict[str, list[SearchHit]]) -> str:
    blocks = []
    for query, hits in hits_by_query.items():
        lines = [f"## Results for query: {query!r}"]
        if not hits:
            lines.append("  (no results)")
        for h in hits:
            lines.append(f"  - title: {h.title}\n    url: {h.url}\n    snippet: {h.snippet}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "(no search results available)"


async def verify_resources(
    topic: str,
    concepts: list[str],
    suggested: list[str],
    hits_by_query: dict[str, list[SearchHit]],
) -> ResourceCheck:
    agent = make_resource_verifier_agent()
    suggested_block = "\n".join(f"  - {s}" for s in suggested) or "  (the planner suggested none)"
    prompt = (
        f"Day topic: {topic}\n"
        f"Concepts: {', '.join(concepts)}\n\n"
        f"=== PLANNER'S SUGGESTED RESOURCES ===\n{suggested_block}\n\n"
        f"=== REAL WEB SEARCH RESULTS ===\n{_format_hits(hits_by_query)}\n\n"
        f"Verify each suggested resource against the search results. Attach only real URLs."
    )
    result = await agent.run(prompt)
    return result.output
