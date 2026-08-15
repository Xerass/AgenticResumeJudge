"""
Recruiter Agent
A synthesis layer, gets input from the other agents and creates
a final verdict based on the decision model
"""

import logging

from agenticresume.domain.models import (
    AnalysisResult,
    Assessment,
    CareerProfile,
    Coverage,
    JobPost,
    Requirement,
)
from agenticresume.agents.mapping import to_analysis_result
from agenticresume.agents.schemas import RecruiterOutput
from agenticresume.domain.scoring import unmet_must_haves, weighted_coverage
from agenticresume.infra.llm import invoke_structured
from agenticresume.settings import Settings


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the lead recruiter making the final hiring call. You are given the \
per-requirement coverage, three panel opinions (skeptic = risk, enthusiast = \
potential, pragmatist = role fit), a deterministic coverage score, and any unmet \
must-have requirements (dealbreakers). Rules:

- Weigh the panel against the evidence. Unmet must-haves are strong grounds for \
rejection unless clearly outweighed.
- Decide exactly one of: 'invite' or 'reject' or 'hold'.
- Justify in 2-3 sentences, naming the factors that drove the call.
- Do not invent facts beyond what is provided.
"""

def _render_context(
    job_post: JobPost,
    coverages: list[Coverage],
    assessments: list[Assessment],
    score: float,
    dealbreakers: list[Requirement],
) -> str:
    reqs = {r.id: r for r in job_post.requirements}
    lines = [
        f"ROLE: {job_post.title} @ {job_post.company}",
        f"DETERMINISTIC COVERAGE SCORE: {score:.0%}",
        "",
        "UNMET MUST-HAVES (dealbreakers):",
    ]
    lines += [f"  - {d.text}" for d in dealbreakers] or ["  (none)"]

    lines += ["", "COVERAGE:"]
    for c in coverages:
        req = reqs.get(c.requirement_id)
        if req is not None:
            lines.append(f"  [{c.status}] ({req.necessity}) {req.text}")

    lines += ["", "PANEL OPINIONS:"]
    for a in assessments:
        pts = "; ".join(a.points)
        lines.append(f"  {a.persona.upper()}: {a.summary} ({pts})")

    return "\n".join(lines)


async def recommend(
    settings: Settings,
    profile: CareerProfile,
    job_post: JobPost,
    coverages: list[Coverage],
    assessments: list[Assessment],
) -> AnalysisResult:
    """Synthesize the panel into a final verdict."""
    cov = tuple(coverages)
    reqs = job_post.requirements

    score = weighted_coverage(cov, reqs)# deterministic half
    dealbreakers = unmet_must_haves(cov, reqs)

    context = _render_context(job_post, coverages, assessments, score, dealbreakers)
    output = await invoke_structured(settings, RecruiterOutput, SYSTEM_PROMPT, context)

    logger.info("verdict: %s (score %.0f%%)", output.decision, score * 100)
    return to_analysis_result(
        output,
        profile_id=profile.id,
        job_post_id=job_post.id,
        coverages=coverages,
        assessments=assessments,
        score=score,
    )