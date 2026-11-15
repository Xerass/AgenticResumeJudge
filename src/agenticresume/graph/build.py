"""Assembles and compiles the screening graph. This is where settings is injected."""

from functools import partial

from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph

from agenticresume.graph.nodes import (
    audit_node,
    critic_node,
    enthusiast_node,
    extract_node,
    finalize_node,
    parse_node,
    pragmatist_node,
    recruiter_node,
    skeptic_node,
    tailor_node,
)
from agenticresume.graph.state import ScreeningState
from agenticresume.settings import Settings


EARLY_STOP_SCORE = 8   # a plan graded this high (and honest) needs no more revision
MAX_PASSES = 2         # hard cap on tailor passes — the loop's termination guarantee


def route_after_recruiter(state: ScreeningState) -> str:
    """Pure, deterministic branch — no LLM. The verdict is the decision point:
    an invite is done; a hold/reject routes into the advocate to tailor."""
    return "tailor" if state["result"].decision != "invite" else "done"


def route_after_critic(state: ScreeningState) -> str:
    """The reflection gate — also pure. Two stop conditions, honesty first:
    the hard cap guarantees termination, and the early-stop fires only when the
    plan is BOTH good enough AND free of unfounded (overreaching) moves. A high
    score that still hides a fabrication loops back rather than shipping."""
    critique = state["critique"]
    if state.get("revision_count", 0) >= MAX_PASSES:
        return "accept"
    if critique.score >= EARLY_STOP_SCORE and not critique.has_unfounded:
        return "accept"
    return "revise"


def build_graph(settings: Settings) -> Runnable[ScreeningState, ScreeningState]:
    """Wire nodes into a compiled pipeline, binding settings onto each node."""
    g = StateGraph(ScreeningState)

    # partial() binds settings, turning each (state, *, settings) into (state) — the
    # signature LangGraph calls. This is the composition root: settings enters once.
    g.add_node("extract", partial(extract_node, settings=settings))
    g.add_node("parse", partial(parse_node, settings=settings))
    g.add_node("audit", partial(audit_node, settings=settings))
    g.add_node("skeptic", partial(skeptic_node, settings=settings))
    g.add_node("enthusiast", partial(enthusiast_node, settings=settings))
    g.add_node("pragmatist", partial(pragmatist_node, settings=settings))
    g.add_node("recruiter", partial(recruiter_node, settings=settings))
    g.add_node("tailor", partial(tailor_node, settings=settings))
    g.add_node("critic", partial(critic_node, settings=settings))
    g.add_node("finalize", finalize_node)  # pure, LLM-free — no settings to bind

    g.add_edge(START, "extract")
    g.add_edge(START, "parse")            # extract ‖ parse

    g.add_edge("extract", "audit")
    g.add_edge("parse", "audit")          # audit waits for BOTH (fan-in barrier)

    g.add_edge("audit", "skeptic")
    g.add_edge("audit", "enthusiast")
    g.add_edge("audit", "pragmatist")     # fan-out: 3 judges in parallel

    g.add_edge("skeptic", "recruiter")
    g.add_edge("enthusiast", "recruiter")
    g.add_edge("pragmatist", "recruiter") # fan-in: recruiter waits for all 3

    # verdict-gated branch: invite ends the run; hold/reject goes to the tailor
    g.add_conditional_edges(
        "recruiter",
        route_after_recruiter,
        {"tailor": "tailor", "done": END},
    )

    # the reflection loop: every tailor pass is graded, then either accepted or
    # sent back for one more pass. "revise" is the backward edge that makes this
    # a cycle rather than a DAG — bounded by MAX_PASSES in route_after_critic.
    g.add_edge("tailor", "critic")
    g.add_conditional_edges(
        "critic",
        route_after_critic,
        {"revise": "tailor", "accept": "finalize"},
    )
    # honesty guard runs on every accepted plan, then the graph ends
    g.add_edge("finalize", END)

    return g.compile()