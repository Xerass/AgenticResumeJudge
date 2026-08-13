from typing import Counter

from langchain_core.messages import HumanMessage, SystemMessage
from agenticresume.agents.mapping import to_assessment
from agenticresume.agents.schemas import AssessmentOutput
from agenticresume.domain.models import Coverage, JobPost, Assessment, JudgePersona
from agenticresume.infra.llm import build_extractor, invoke_structured
from agenticresume.settings import Settings



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



async def run_judge(settings: Settings, job_post: JobPost, coverages: list[Coverage], *, persona:JudgePersona, system_prompt: str) -> Assessment:
    """Runs the Judge LLM, given a judge Persona and system prompt"""
    context = _render_coverages(job_post, coverages)
    output = await invoke_structured(settings, AssessmentOutput, system_prompt, context)

    return to_assessment(output, persona=persona)