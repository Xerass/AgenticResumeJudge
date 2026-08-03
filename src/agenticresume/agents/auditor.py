"""Auditor Agent: Asesses a candidate's CareerProfile against a JobPosts' 
requirements and produces one coverage item per requirement. Facts and requirements 
are ID'd"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from agenticresume.agents.mapping import coverages_from_audit
from agenticresume.agents.schemas import AuditorOutput
from agenticresume.domain.models import CareerProfile, Coverage, JobPost
from agenticresume.infra.llm import build_extractor
from agenticresume.settings import Settings


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are a hiring auditor. For EACH requirement, decide how well the candidate \
meets it, using only the facts and skills provided. Rules:

- Judge every requirement. Reference it by its [R#] number.
- status: "covered" if clearly met, "partial" if partly or adjacently met \
(e.g. a related skill), "none" if unmet.
- Cite the [F#] facts that justify a covered/partial judgment. If nothing \
supports it, cite nothing and use "none".
- Do not invent evidence. Only cite facts that genuinely apply.
- Give a one-sentence reason per requirement.
"""


def _render_context(profile: CareerProfile, facts: tuple, requirements: tuple)->str:
    """builds a context string for the auditor prompt"""

    where = {r.id: f"{r.title} @ {r.company}" for r in profile.roles}
    where |= {p.id: f"Project: {p.name}" for p in profile.projects}

    skills = ", ".join(s.display_name for s in profile.skills) or "(none listed)"
    lines = [f"CANDIDATE: {profile.full_name}", f"SKILLS: {skills}", "", "FACTS:"]
    for i, f in enumerate(facts, 1):
        lines.append(f"[F{i}] ({where.get(f.context_id, '?')}) {f.text}")

    lines += ["", "REQUIREMENTS:"]
    for i, req in enumerate(requirements, 1):
        yrs = f", {req.year_required}+ yrs" if req.year_required else ""
        lines.append(f"[R{i}] ({req.kind}, {req.necessity}{yrs}) {req.text}")

    return "\n".join(lines)

async def audit(settings: Settings, profile: CareerProfile, job_post: JobPost) -> list[Coverage]:
    """Audits one Coverage per requirement"""

    facts = profile.active_facts
    requirements = job_post.requirements

    context = _render_context(profile, facts, requirements)
    model = build_extractor(settings, AuditorOutput)

    result = await model.ainvoke(
        [
            SystemMessage(content = SYSTEM_PROMPT),
            HumanMessage(content = context)
        ]
    )

    if not isinstance(result, AuditorOutput):
        raise TypeError(f"auditor returned {type(result).__name__}, expected AuditorOutput")

    #pass to the mapper, after extraction of result (AuditorOutput)
    coverages = coverages_from_audit(result, facts, requirements)


    logger.info("audited %d requirements", len(coverages))
    return coverages