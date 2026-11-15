"""Critic Agent: the reflection step. Reads the tailor's plan back against the
candidate's real facts and the weak requirements, then grades it. Its output is
built for the reviser to act on — a score the loop gates on, and one targeted
note per move that still needs work. It writes nothing new; it only judges."""

import logging

from agenticresume.agents.mapping import to_critique
from agenticresume.agents.schemas import CritiqueOutput
from agenticresume.domain.models import CareerProfile, Coverage, Critique, JobPost, Reframe
from agenticresume.infra.llm import invoke_structured
from agenticresume.settings import Settings

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are a hiring-committee reviewer grading a résumé TAILORING PLAN. Each move \
rewords the candidate's EXISTING facts to match a weak job requirement more \
closely. Judge the plan on two axes, honesty first:

- HONESTY: a move may use ONLY information present in its cited [F#] fact. If the \
proposed text names a skill, number, tool, employer, or outcome not in that fact, \
mark it 'unfounded'. This is the most serious failure — fabrication defeats the \
whole system.
- TARGETING: a truthful move should mirror the requirement's wording and make the \
match obvious. If it is vague, generic, or buries the match, mark it 'weak'.
- A move that is both truthful and sharply targeted is 'strong'. A 'gap' honestly \
admitted (no fact supports the requirement) is correct behaviour, not a failure.

Return an overall score from 1 to 10 (10 = every move truthful and sharply \
targeted), a one-sentence summary, and one note ONLY for each move that is 'weak' \
or 'unfounded' — say exactly what the reviser should change. Reference each \
requirement by its [R#] number. Omit notes for moves that are already strong.
"""


def _render_context(
    profile: CareerProfile,
    job_post: JobPost,
    coverages: list[Coverage],
    reframes: list[Reframe],
) -> str:
    """Show the facts as ground truth, then the plan under review. [R#]/[F#]
    numbering matches the tailor's (requirements in job-post order, facts in
    active-facts order) so the critic's notes resolve back to the same moves."""

    facts = profile.active_facts
    requirements = job_post.requirements
    req_index = {r.id: i for i, r in enumerate(requirements, 1)}
    fact_index = {f.id: i for i, f in enumerate(facts, 1)}
    req_text = {r.id: r.text for r in requirements}

    lines = [
        f"CANDIDATE: {profile.full_name}",
        "",
        "FACTS (ground truth — no move may exceed what its cited fact says):",
    ]
    for i, f in enumerate(facts, 1):
        lines.append(f"[F{i}] {f.text}")

    lines += ["", "TAILORING PLAN UNDER REVIEW:"]
    for r in reframes:
        rn = req_index.get(r.requirement_id, "?")
        target = req_text.get(r.requirement_id, "?")
        if r.move == "gap":
            lines.append(f'[R{rn}] GAP — "{target}" :: {r.rationale}')
        else:
            fn = fact_index.get(r.source_fact_id, "?")
            lines.append(
                f'[R{rn}] {r.move.upper()} from [F{fn}] -> "{r.proposed_text}" '
                f'(requirement: "{target}")'
            )

    return "\n".join(lines)


async def critique(
    settings: Settings,
    profile: CareerProfile,
    job_post: JobPost,
    coverages: list[Coverage],
    reframes: list[Reframe],
) -> Critique:
    """Grade a tailoring plan. An empty plan is nothing to revise, so it scores a
    perfect 10 — the loop reads that as 'stop', not 'start over'."""

    if not reframes:
        logger.info("no reframes to critique; plan is vacuously complete")
        return Critique(score=10, summary="No weakpoints to tailor; nothing to revise.")

    context = _render_context(profile, job_post, coverages, reframes)
    output = await invoke_structured(settings, CritiqueOutput, SYSTEM_PROMPT, context)

    result = to_critique(output, job_post.requirements)
    logger.info(
        "critique: %d/10 over %d moves, %d flagged%s",
        result.score,
        len(reframes),
        len(result.notes),
        " (unfounded present)" if result.has_unfounded else "",
    )
    return result
