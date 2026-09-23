"""Tailor Agent: the first advocate. Walks the weak coverages and proposes
truthful ways to present the candidate's EXISTING history against the JD.
Reframes facts into variants; never invents new ones."""

import logging

from agenticresume.agents.mapping import reframes_from_tailor
from agenticresume.agents.schemas import TailorOutput
from agenticresume.domain.models import CareerProfile, Coverage, Fact, JobPost, Reframe, Requirement
from agenticresume.domain.scoring import _NECESSITY_WEIGHT  # rank weakpoints by what moves the score
from agenticresume.infra.llm import invoke_structured
from agenticresume.settings import Settings

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are a résumé strategist. For each WEAK requirement, help the candidate \
present their EXISTING history in the strongest truthful way. Rules:

- You may only reword facts already provided. NEVER add a skill, number, \
employer, tool, or achievement that is not present in the cited fact.
- move "reframe": cite the [F#] you are rewording and give the variant in \
proposed_text, mirroring the requirement's language.
- move "surface": cite an existing [F#] the audit did not credit for this \
requirement; a rewrite is optional.
- move "gap": if nothing truthful supports it, say so plainly. Do not invent \
evidence to fill it.
- Reference the requirement by its [R#] number. One move per weak requirement.
"""


def _render_context(
    profile: CareerProfile,
    weak: list[Coverage],
    requirements: tuple[Requirement, ...],
    facts: tuple[Fact, ...],
) -> str:
    """Show ALL facts (so 'surface' can find missed ones) but only the weak
    requirements — each keeping its original [R#] so the mapper resolves it."""

    where = {r.id: f"{r.title} @ {r.company}" for r in profile.roles}
    where |= {p.id: f"Project: {p.name}" for p in profile.projects}
    req_index = {r.id: i for i, r in enumerate(requirements, 1)}

    lines = [f"CANDIDATE: {profile.full_name}", "", "FACTS (all available):"]
    for i, f in enumerate(facts, 1):
        lines.append(f"[F{i}] ({where.get(f.context_id, '?')}) {f.text}")

    lines += ["", "WEAK REQUIREMENTS (strongest-mattering first):"]
    for c in weak:
        req = next(r for r in requirements if r.id == c.requirement_id)
        lines.append(
            f"[R{req_index[req.id]}] ({req.kind}, {req.necessity}) {req.text} "
            f":: audit says {c.status} — {c.reasoning}"
        )

    return "\n".join(lines)


async def tailor(
    settings: Settings,
    profile: CareerProfile,
    job_post: JobPost,
    coverages: list[Coverage],
) -> list[Reframe]:
    """Produce a truthful tailoring plan over the weak requirements."""

    facts = profile.active_facts
    requirements = job_post.requirements
    necessity = {r.id: r.necessity for r in requirements}

    # weakpoint recognition: anything not fully covered, must-haves first.
    weak = sorted(
        (c for c in coverages if c.status != "covered"),
        key=lambda c: _NECESSITY_WEIGHT[necessity[c.requirement_id]],
        reverse=True,
    )
    if not weak:
        logger.info("no weakpoints to tailor")
        return []

    context = _render_context(profile, weak, requirements, facts)
    output = await invoke_structured(settings, TailorOutput, SYSTEM_PROMPT, context)

    reframes = reframes_from_tailor(output, facts, requirements)
    logger.info("tailored %d weakpoints into %d moves", len(weak), len(reframes))
    return reframes