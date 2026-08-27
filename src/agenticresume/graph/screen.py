"""Assembles and compiles the screening graph. This is where settings is injected."""

from functools import partial

from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph

from agenticresume.graph.nodes import (
    audit_node,
    enthusiast_node,
    extract_node,
    parse_node,
    pragmatist_node,
    recruiter_node,
    skeptic_node,
)
from agenticresume.graph.state import ScreeningState
from agenticresume.settings import Settings


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

    g.add_edge("recruiter", END)

    return g.compile()