"""
Turns raw job-description text into a JobPostOutput schema
"""

import logging
from langchain_core.messages import HumanMessage, SystemMessage
from agenticresume.agents.schemas import JobPostOutput
from agenticresume.infra.llm import build_extractor
from agenticresume.settings import Settings

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are a precise job-description parser. Break the posting into individual \
requirements. Follow these rules exactly:

- One requirement per distinct expectation. Split compound sentences into \
separate requirements.
- Classify each requirement's kind:
  - skill: a named technology, tool, language, or framework
  - domain: experience in an industry or field (e.g. "fintech", "healthcare")
  - credential: a degree, certification, or license
  - soft: a soft skill such as communication or leadership
  - logistical: location, work authorization, availability, travel
- Classify necessity: "must_have" for required/mandatory items, "nice_to_have" \
for preferred, bonus, or "a plus" items.
- For skill requirements, also give the normalized skill name \
(e.g. "react.js" -> "React"). Leave it empty for non-skill requirements.
- Record minimum years of experience only when explicitly stated.
- Never invent requirements. Extract only what the posting states.
"""

async def parse_job_post(settings: Settings, jd_text: str) -> JobPostOutput:

    """Parses the job description into a structured requirement"""

    model = build_extractor(settings, JobPostOutput)

    result = await model.ainvoke(
        [
            SystemMessage(content = SYSTEM_PROMPT),
            HumanMessage(content=jd_text)
        ]
    )

    if not isinstance(result, JobPostOutput):
        raise TypeError(f"parser returned {type(result).__name__}, expected JobPostOutput")

    logger.info(
        "parsed %d requirements for %r", len(result.requirements), result.title or "unknown"
    )

    return result