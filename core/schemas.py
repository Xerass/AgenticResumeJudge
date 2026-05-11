# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
from typing import Literal

#strict schema set so pydantic can verify on model create and also 
#give models a strict format to follow

#constructor schemas

class ContactInfo(BaseModel):
    name: str
    email: str

class Experience(BaseModel):
    role: str
    company: str
    description: str
    skills_used: list[str]

class Project(BaseModel):
    title: str
    tech_stack: list[str]

class ConstructorOutput(BaseModel):
    contact: ContactInfo
    experience: list[Experience]
    projects: list[Project]



#auditor schemas
class SkillAnalysis(BaseModel):
    required_skill: str
    #literals force that the model strictly follow the type and value specified
    status: Literal["direct_match", "categorical_match", "missing"]
    candidate_skill: str | None = None
    category: str | None = None

class MatchSummary(BaseModel):
    direct_matches: int
    categorical_matches: int
    missing_skills: int
    match_rate: float = Field(ge=0.0, le=1.0)

class AuditorOutput(BaseModel):
    skill_analysis: list[SkillAnalysis]
    match_summary: MatchSummary
    assessment: Literal["strong_match", "partial_match", "weak_match"]

#Skeptic Schemas
class RedFlag(BaseModel):
    issue: str
    impact: Literal["critical", "major", "minor"]
    reasoning: str

class SkepticOutput(BaseModel):
    red_flags: list[RedFlag]
    velocity_risk: Literal["low_ramp_up", "medium_ramp_up", "high_ramp_up"]
    estimated_ramp_up_time: str
    risk_score: int = Field(ge=0, le=100)
    verdict_recommendation: Literal["reject", "proceed_with_caution", "interview"]

#enthusiast Schemas
class Defense(BaseModel):
    gap_identified: str
    defense_argument: str
    validity_score: int = Field(ge=1, le=10)

class EnthusiastOutput(BaseModel):
    defenses: list[Defense]
    hidden_value: list[str]
    culture_fit_prediction: str
    hiring_recommendation: Literal["strong_invite", "invite", "weak_invite"]