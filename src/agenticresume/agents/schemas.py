"""Wire schems -> shapes emitted by the LLM, not domain, simply formats for reliable extraction
    Does not contain IDs, no cross-referencing. A mapper will turn them into a domain object
"""

from pydantic import BaseModel, ConfigDict, Field
from agenticresume.domain.models import Necessity, RequirementKind
class Wire(BaseModel):
    """Base model for wire schemas"""
    model_config = ConfigDict(extra = "forbid")


#all of these are turned into JSON to send to an LLM
class ExtractedRole(Wire):
    #we need description fields since LLMs rely on this to understand what is being looked for
    role: str = Field(description="Job title, e.g. 'Senior Backend Engineer'")
    company: str = Field(description="Employer name")
    started: str = Field(description="Start date as ISO 'YYYY-MM' or 'YYYY'; empty if unknown")
    ended: str = Field(description="End date as ISO 'YYYY-MM'; empty string if current or unknown")

    #learning purpose: dont do default = [], this item is mutable we need a factory so schemas dont share the same list
    bullets: list[str] = Field(
        default_factory= list,
        description="Each concrete accomplishment or responsibility, verbatim, one per bullet"
    )

    skills: list[str] = Field(
        default_factory=list,
        description="Specific tools, languages, frameworks named in this role",
    )


class ExtractedProject(Wire):
    name: str = Field(description="Project title")
    summary: str = Field(default="", description="One-line description if present")
    bullets: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)


class ExtractionOutput(Wire):
    full_name: str = Field(description="Candidate's full name; empty string if not found")
    roles: list[ExtractedRole] = Field(default_factory=list)
    projects: list[ExtractedProject] = Field(default_factory=list)


class ExtractedRequirement(Wire):
    text: str = Field(description="The requirement stated verbatim, e.g. '5+ years of Python'")
    kind: RequirementKind = Field(
        description=(
            "Category: 'skill' (a named tool/tech/language), 'domain' (industry or "
            "field experience), 'credential' (degree/cert/license), 'soft' (a soft "
            "skill), or 'logistical' (location, work authorization, availability)"
        )
    )
    necessity: Necessity = Field(
        description="'must_have' if required/mandatory, 'nice_to_have' if preferred or a bonus"
    )
    skill: str = Field(
        default="",
        description="If kind is 'skill', the normalized skill name (e.g. 'React'); empty otherwise",
    )
    years_required: int | None = Field(
        default=None, description="Minimum years of experience if explicitly stated, else null"
    )

class JobPostOutput(Wire):
    company: str = Field(description="Hiring company name; empty string if not found")    
    title:str = Field(description = "Job title; empty string if not found")
    requirements: list[ExtractedRequirement] = Field(default_factory=list)



