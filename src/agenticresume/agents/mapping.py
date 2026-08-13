""" Transforms the Wire Schema of LLMs into an actual output"""


from datetime import date, datetime
from uuid import UUID

from agenticresume.agents.schemas import AssessmentOutput, ExtractionOutput, JobPostOutput, AuditorOutput, RecruiterOutput
from agenticresume.domain.models import (
    AnalysisResult,
    Assessment,
    CareerProfile, 
    Fact,
    JudgePersona, 
    Project, 
    Role, 
    Skill,
    JobPost,
    Requirement,
    Coverage
    )

def _parse_month(value: str) -> date | None:
    """Tolerant date parse. 'YYYY-MM' or 'YYYY' -> date; anything else -> None."""
    value = value.strip()
    for fmt in ("%Y-%m", "%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None

def _clean_bullets(bullets: list[str]) -> list[str]:
    """Drop empties and case-insensitive duplicates, preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for b in bullets:
        b = b.strip()
        if b and b.lower() not in seen:
            seen.add(b.lower())
            out.append(b)
    return out

def _dedup_skills(raw: list[str]) -> tuple[Skill, ...]:
    """Build Skills, deduped by canonical name, order preserved."""
    out: list[Skill] = []
    for s in raw:
        if s.strip():
            skill = Skill.of(s)
            if skill not in out:  #skill.__eq__ compares canonical names
                out.append(skill)
    return tuple(out)

def to_career_profile(extraction: ExtractionOutput, *, source_document: str = "") -> CareerProfile:
    """Translates extracted wire data into a domain profile"""

    roles: list[Role] = []
    projects: list[Project] = []
    facts: list[Fact] = []


    for r in extraction.roles:
        role = Role(
            title=r.role or "Unknown role",
            company=r.company or "Unknown company",
            started=_parse_month(r.started),
            ended=_parse_month(r.ended),
        ) 

        roles.append(role)

        skills = _dedup_skills(r.skills)

        for text in _clean_bullets(r.bullets):
            facts.append(
                Fact(
                    text=text,
                    context_kind="role",
                    context_id=role.id, #attach context of where skill was  used
                    skills=skills,
                    source_document=source_document,
                )
            )

    for p in extraction.projects:
        project = Project(name=p.name or "Unknown project", summary=p.summary)
        projects.append(project)
        skills = _dedup_skills(p.skills)
        for text in _clean_bullets(p.bullets):
            facts.append(
                Fact(
                    text=text,
                    context_kind="project",
                    context_id=project.id,
                    skills=skills,
                    source_document=source_document,
                )
            )


    return CareerProfile(
        full_name = extraction.full_name or "Unknown Candidate",
        roles = tuple(roles),
        projects = tuple(projects),
        facts = tuple(facts)
    )


        
def to_job_post(extraction: JobPostOutput, *, raw_text: str = "") -> JobPost:
    """Transofrms wire schema of jobPostOutput into domain JobPost"""

    requirements: list[Requirement] = []


    for r in extraction.requirements:
        #text is the verbatim requiremeent of the requirment object
        text = r.text.strip()
        if not text:
            continue

        kind = r.kind
        skill: Skill | None = None

        if kind == "skill":
            name = r.skill.strip()
            if name:
                skill = Skill.of(name)
            else:
                #model tagged it as skill but named none, reclassify it as a domain skill
                kind = "domain"


        requirements.append(
            Requirement(
                text=text,
                kind=kind,
                necessity=r.necessity,
                skill=skill,
                year_required = r.years_required
            )
        )

    return JobPost(
        company = extraction.company.strip() or "Unknown Company",
        title = extraction.title.strip() or "Unknown Title",
        raw_text = raw_text.strip(),
        requirements = tuple(requirements)
    )

def coverages_from_audit(
    output: AuditorOutput,
    facts: tuple[Fact, ...],
    requirements: tuple[Requirement, ...],        
) -> list[Coverage]:


    """Translates index-based audit into real IDs"""

    coverages: list[Coverage] = []
    seen: set[UUID] = set()

    for item in output.coverage:
        ri = item.requirement_index - 1
        if not 0 <= ri < len(requirements):
            continue #hallucincated index, drop it

        req = requirements[ri]
        if req.id in seen:
            continue #duplicated coverage

        seen.add(req.id)

        evidence = tuple(
            facts[fi - 1].id
            for fi in item.evidence_indices
            if 0 <= fi - 1 < len(facts)  # drop hallucinated fact indices
        )

        status = item.status
        if not evidence and status != "none":
            status = "none" #no evidence (forced on no evidence and status = none)
        elif evidence and status == "none":
            status = "partial" 

        coverages.append(
            Coverage(
                requirement_id=req.id,
                status=status,
                evidence=evidence,
                reasoning=item.reasoning,
            )
        )


    #any reqs the auditor skipped is uncovered
    for req in requirements:
        if req.id not in seen:
            coverages.append(
                Coverage(requirement_id=req.id, status="none", reasoning="not assessed")
            )

    return coverages

def to_assessment(output: AssessmentOutput, *, persona: JudgePersona) -> Assessment:
    """Stamps a persona onto a judge's raw output and cleans it"""

    points = tuple(p.strip() for p in output.points if p.strip())
    summary = output.summary.strip() or "No assessment produced."  # summary is NonEmptyStr
    return Assessment(persona=persona, summary=summary, points=points)


def to_analysis_result(
    output: RecruiterOutput,
    *,
    profile_id: UUID,
    job_post_id: UUID,
    coverages: list[Coverage],
    assessments: list[Assessment],
    score: float,
) -> AnalysisResult:
    """Transforms the recruiter output into a domain AnalysisResult"""
    return AnalysisResult(
        profile_id=profile_id,
        job_post_id=job_post_id,
        coverages=tuple(coverages),
        assessments=tuple(assessments),
        decision=output.decision,
        score=score,  # deterministic, injected, not from the LLM
        rationale=output.rationale.strip() or "No rationale provided.",
    )