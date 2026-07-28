"""
Extractor Agent

Turns Raw Resume into an Extraction Output Schema, does nothing else.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from agenticresume.agents.schemas import ExtractionOutput
from agenticresume.infra.llm import build_extractor
from agenticresume.settings import Settings


logger =logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a precise resume-parsing engine. Extract structured data from the \
resume text below. Follow these rules exactly:

- Transcribe, do not summarize. Each distinct accomplishment or responsibility \
becomes its own bullet, in the candidate's own words. Lightly clean formatting \
artifacts (stray bullets, line breaks) but do not reword.
- Never invent information. If something is not in the text, leave it empty. \
Do not guess dates, employers, titles, or skills.
- List a skill only if it is explicitly named. Normalize it to its common \
canonical form (e.g. "react.js" -> "React", "postgres" -> "PostgreSQL").
- Attribute each bullet and skill to the specific role or project it appears under.
- Dates: "YYYY-MM" when a month is given, "YYYY" when only a year. Use an empty \
string for the end date of a current role.
"""


async def extract_profile(settings: Settings, resume_text:str) -> ExtractionOutput:
    """Extracts structured profile data from resume"""

    model = build_extractor(settings, ExtractionOutput)

    result = await model.ainvoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=resume_text),
        ]
    )

    if not isinstance(result, ExtractionOutput):
        raise TypeError(f"extractor returned {type(result).__name__}, expected ExtractionOutput")

    logger.info(
        "extracted %d roles, %d projects for %r",
        len(result.roles),
        len(result.projects),
        result.full_name or "unknown",
    )

    return result