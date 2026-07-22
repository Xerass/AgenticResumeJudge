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

    



