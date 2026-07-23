"""
Domain Models
Defines the "vocabulary" of the system
Pure data and rules, each is a class
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, date, datetime
from typing import Annotated, Literal, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

#"cookie cutter" vars

NonEmptyStr = Annotated[str, Field(min_length= 1)]

#define literals for string vars
RequirementKind = Literal["skill", "domain", "credential", "soft", "logistical"]
Necessity = Literal["must_have", "nice_to_have"]
CoverageStatus = Literal["covered", "partial", "none"]
ContextKind = Literal["role", "project"]
JudgePersona = Literal["skeptic", "enthusiast"]
FactStatus = Literal["active", "superseded"]

#vase class, defines configDict only
class Base(BaseModel):

    """Enforce immutability within models"""

    model_config =  ConfigDict(frozen = True, extra = "forbid")


def canonicalize(name: str) -> str:
    """Fold a skill name to its identity form. 'React.js ' -> 'reactjs'."""
    cleaned = re.sub(r"[^a-z0-9+#\s]", "", name.strip().lower())
    return re.sub(r"\s+", " ", cleaned)


# ------------ Entities -----------------------

class Skill(Base): 
    """ Defines a capability of the candidate, acts as facts for models to reference during eval"""

    display_name: NonEmptyStr
    canonical_name: NonEmptyStr
    
    #basically static (no self needed), acts as a factory (we will be spamming this)
    @classmethod 
    def of(cls, raw:str) -> Skill:
        #cleans up input before construction
        return cls(display_name = raw.strip(), canonical_name = canonicalize(raw))


    #since literally  "anything" can be a skill, strict model validation is unneded

    #what we do need is dedup of skill names, so check equivalency of their canonical names
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Skill) and self.canonical_name == other.canonical_name

    def __hash__(self) -> int:
        return hash(self.canonical_name)


class Role(Base):
    """Employment at a company over a period"""

    #uuid4-factory for role IDs
    id: UUID = Field(default_factory= uuid4)
    title: NonEmptyStr
    company: NonEmptyStr
    started: date
    ended: date | None #none = currently still there

    @model_validator(mode = "after")
    def _dates_are_ordered(self) -> Self:
        if self.ended is not None and self.ended < self.started:
            raise ValueError("ended must not precede started")
        return self


class Project(Base):
    """Work outside of employment"""

    id:UUID = Field(default_factory=uuid4)
    name: NonEmptyStr
    #just extract it raw
    summary: str = ""


class Fact(Base):
    """One atomic true thing you did, in your own words.

    The atom of the system: unit of extraction, of evidence, and of
    generation. `text` is never edited — rephrasings belong to a variant.
    """

    id: UUID = Field(default_factory=uuid4)
    text: NonEmptyStr

    context_kind: ContextKind
    context_id: UUID

    #type hinted as Tuple of Skill, ... means any amount
    skills: tuple[Skill, ...] = ()

    #be aware of source, model must always be open to update
    source_document: str = ""
    source_excerpt: str = ""

    status: FactStatus = "active"
    superseded_by: UUID | None = None

    @property
    def fingerprint(self) -> str:
        """Dedup key: normalized text scoped to its context."""
        normalized = re.sub(r"\s+", " ", self.text.strip().lower())
        return hashlib.sha256(f"{self.context_id}:{normalized}".encode()).hexdigest()

    @model_validator(mode="after")
    def _supersession_is_consistent(self) -> Self:
        if (self.status == "superseded") != (self.superseded_by is not None):
            raise ValueError("superseded facts must name their replacement, and vice versa")
        return self


class CareerProfile(Base):
    """"aggreagtes truth of the person"""

    id: UUID = Field(default_factory= uuid4)
    full_name: NonEmptyStr
    roles: tuple[Role, ...] = ()
    projects: tuple[Project, ...] = ()
    facts: tuple[Fact, ...] = ()

    @property
    def active_facts(self) -> tuple[Fact, ...]:
        #only return truths that are currently true for the you
        return tuple(f for f in self.facts if f.status == "active")


    @model_validator(mode = "after")
    def _facts_reference_known_contexts(self) -> Self:
        #facts must only refer to real roles or real projects
        role_ids = {r.id for r in self.roles}
        project_ids = {p.id for p in self.projects}
        for fact in self.facts:
            known = role_ids if fact.context_kind == "role" else project_ids
            if fact.context_id not in known:
                raise ValueError(
                    f"fact {fact.id} references unknown {fact.context_kind} {fact.context_id}"
                )
        return self



# --- JD Related Entities ----------------

class Requirement(Base):
    """Things required for by the Job Description"""


    id: UUID = Field(default_factory= uuid4)
    text: NonEmptyStr
    kind: RequirementKind
    necessity: Necessity
    skill: Skill | None = None
    #set lower bound to 0, obv
    year_required: int | None = Field(default = None, ge = 0)


    @model_validator(mode="after")
    def _skill_requirements_name_a_skill(self) -> Self:
        if self.kind == "skill" and self.skill is None:
            raise ValueError("skill-kind requirements must name a skill")
        return self


class JobPost(Base):
    """A role you are targeting."""

    id: UUID = Field(default_factory=uuid4)
    company: NonEmptyStr
    title: NonEmptyStr
    raw_text: str = ""
    requirements: tuple[Requirement, ...] = ()
