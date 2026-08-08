"""
Enthusiast Judge

Focuses on growth potential and upside. Looks for ways to make a hire work, rather than reasons to reject.
"""

import logging

from agenticresume.agents.judge import run_judge

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an enthusiastic hiring advocate on the panel — the candidate's champion, \
but an honest one. Using only the per-requirement assessment below:

- Highlight covered requirements and the strongest evidence.
- For partial or missing skills, argue for transferable/adjacent experience and \
demonstrated ability to ramp up — but only where the assessment supports it.
- Do NOT fabricate qualifications or claim an unmet requirement is met.
- Frame growth potential realistically.

Give a short summary of your overall enthusiasm, then a list of specific, \
grounded points in the candidate's favor.
"""

async def run_enthusiast(settings, job_post, coverages):
    return await run_judge(
        settings,
        job_post,
        coverages,
        persona="enthusiast",
        system_prompt=SYSTEM_PROMPT,
    )