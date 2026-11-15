"""
Graph nodes, acts as adapters for calling the existing agents

a node is a defined action, reading what it needs from the state, and writing what it produces back to the state.
"""

import logging

from agenticresume.agents.auditor import audit
from agenticresume.agents.critic import critique
from agenticresume.agents.enthusiast import run_enthusiast
from agenticresume.agents.extractor import extract_profile
from agenticresume.agents.jobpost import parse_job_post
from agenticresume.agents.mapping import to_career_profile, to_job_post
from agenticresume.agents.pragmatist import run_pragmatist
from agenticresume.agents.recruiter import recommend
from agenticresume.agents.skeptic import run_skeptic
from agenticresume.agents.tailor import tailor
from agenticresume.domain.tailoring import redact_unfounded
from agenticresume.graph.state import ScreeningState
from agenticresume.settings import Settings

logger = logging.getLogger(__name__)


async def extract_node(state: ScreeningState, *, settings: Settings) -> ScreeningState:
    extraction = await extract_profile(settings, state["resume_text"])
    profile = to_career_profile(extraction, source_document="resume")
    return {"profile": profile}


async def parse_node(state: ScreeningState, *, settings: Settings) -> ScreeningState:
    parsed = await parse_job_post(settings, state["jd_text"])
    return {"job_post": to_job_post(parsed, raw_text=state["jd_text"])}


async def audit_node(state: ScreeningState, *, settings: Settings) -> ScreeningState:
    coverages = await audit(settings, state["profile"], state["job_post"])
    return {"coverages": coverages}


async def skeptic_node(state: ScreeningState, *, settings: Settings) -> ScreeningState:
    a = await run_skeptic(settings, state["job_post"], state["coverages"])
    return {"assessments": [a]}          # list-of-one → reducer concatenates the 3


async def enthusiast_node(state: ScreeningState, *, settings: Settings) -> ScreeningState:
    a = await run_enthusiast(settings, state["job_post"], state["coverages"])
    return {"assessments": [a]}


async def pragmatist_node(state: ScreeningState, *, settings: Settings) -> ScreeningState:
    a = await run_pragmatist(settings, state["job_post"], state["coverages"])
    return {"assessments": [a]}


async def recruiter_node(state: ScreeningState, *, settings: Settings) -> ScreeningState:
    result = await recommend(
        settings,
        state["profile"],
        state["job_post"],
        state["coverages"],
        state["assessments"],# merged list of all 3, thanks to the reducer + fan-in
    )
    return {"result": result}


async def tailor_node(state: ScreeningState, *, settings: Settings) -> ScreeningState:
    # critique is absent on the first pass and present on a revision — the tailor
    # treats a present one as repair instructions. state.get, since total=False.
    reframes = await tailor(
        settings,
        state["profile"],
        state["job_post"],
        state["coverages"],
        critique=state.get("critique"),
    )
    return {"tailoring": reframes}


async def critic_node(state: ScreeningState, *, settings: Settings) -> ScreeningState:
    # grades the plan the tailor just wrote, and bumps the pass counter the router
    # gates on. single writer of both keys, so last-write-wins is correct here.
    result = await critique(
        settings,
        state["profile"],
        state["job_post"],
        state["coverages"],
        state["tailoring"],
    )
    return {"critique": result, "revision_count": state.get("revision_count", 0) + 1}


def finalize_node(state: ScreeningState) -> ScreeningState:
    # pure, no LLM: the loop may accept at its cap with an overreaching move still
    # present. Enforce honesty before the plan leaves the graph. No-op if nothing
    # was flagged, so it runs harmlessly on the early-stop path too.
    guarded = redact_unfounded(state["tailoring"], state["critique"])
    return {"tailoring": guarded}