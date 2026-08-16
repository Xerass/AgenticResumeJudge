"""
Skeptic Judge

Reads auditor assesment and argues for the risks and gasps
Focuses on what you lack
"""

import logging

from agenticresume.agents.judge import run_judge
from agenticresume.domain.models import Assessment, Coverage, JobPost
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


async def run_skeptic(
    settings: Settings, job_post: JobPost, coverages: list[Coverage]
) -> Assessment:
    return await run_judge(
        settings,
        job_post,
        coverages,
        persona="skeptic",
        system_prompt=SYSTEM_PROMPT,
    )