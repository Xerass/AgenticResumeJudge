"""
Skeptic Judge

Reads auditor assesment and argues for the risks and gasps
Focuses on what you lack
"""

import logging
from collections import Counter
from langchain_core.messages import SystemMessage, HumanMessage
from agenticresume.agents.mapping import to_assessment
from agenticresume.agents.schemas import AssessmentOutput
from agenticresume.domain.models import Assessment, Coverage, JobPost
from agenticresume.infra.llm import build_extractor
from agenticresume.settings import Settings



logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a skeptical senior technical reviewer on a hiring panel. Your role is \
to surface genuine risk — not to reject reflexively. Using only the \
per-requirement assessment below:

- Weigh unmet must_have requirements most heavily.
- Flag partial matches where the candidate would likely need ramp-up.
- Base every point on the assessment provided; do NOT invent weaknesses.
- If the risks are minor, say so plainly. Fairness is part of credibility.

Give a short summary of your overall concern level, then a list of specific, \
grounded risk points.
"""


def _render_coverages(job_post: JobPost, coverages: list[Coverage]) -> str:
    """Renders the list of coverages with job post into a neat string for LLM"""

    reqs = {r.id: r for r in job_post.requirements}
    tally = Counter(c.status for c in coverages)

    lines = [
        f"ROLE: {job_post.title} @ {job_post.company}",
        f"TALLY: {tally['covered']} covered, {tally['partial']} partial, {tally['none']} none",
        "",
        "ASSESSMENT PER REQUIREMENT:"
    ]


    for c in coverages:
        req = reqs.get(c.requirement_id)
        if req is not None:
            lines.append(f"- [{c.status}] ({req.necessity}) {req.text} :: {c.reasoning}")


    return "\n".join(lines)


async def assess_skeptic(settings: Settings, job_post: JobPost, coverages: list[Coverage]) -> Assessment:
    """Runs the skeptic agent to produce an assessment of the coverages"""

    """Produce the Skeptic's assesment"""

    context = _render_coverages(job_post, coverages)
    model = build_extractor(settings, AssessmentOutput)

    result = await model.ainvoke(
        [
            SystemMessage(content=SYSTEM_PROMPT), 
            HumanMessage(content=context)
        ]
    )

    if not isinstance(result, AssessmentOutput):
        raise TypeError(f"skeptic returned {type(result).__name__}, expected AssessmentOutput")

    logger.info("skeptic produced %d points", len(result.points))
    return to_assessment(result, persona="skeptic")