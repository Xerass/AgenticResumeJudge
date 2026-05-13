import json
import logging

from langchain_core.messages import SystemMessage, HumanMessage

from core.llm import get_llm
from core.schemas import EnthusiastOutput

logger = logging.getLogger("agent.enthusiast")

SYSTEM_PROMPT = """
you are the 'growth potential' analyzer. your goal is to defend the candidate against the auditor's findings.

input context:
1. auditor_gaps: a list of skills the candidate is missing.
2. job_description: the target role requirements.

your task:
iterate through every 'missing skill' found by the auditor.
look for a 'conceptual equivalent' in the candidate's background.

examples:
- missing: 'react', candidate has: 'vue'. defense: 'strong component lifecycle understanding transfers in <1 week'.
- missing: 'aws', candidate has: 'azure'. defense: 'cloud infrastructure patterns are identical'.

output requirements:
- do not be delusional. if a gap is real (e.g., missing 5 years of management), admit it.
"""


async def enthusiastAgent(auditor_data: dict, jd_text: str) -> EnthusiastOutput:
    llm = get_llm(temperature=0.3).with_structured_output(EnthusiastOutput)

    user_context = f"""
    --- auditor report (the prosecution) ---
    {json.dumps(auditor_data)}

    --- job description (the law) ---
    {jd_text}
    """

    try:
        return await llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_context),
        ])

    except Exception as e:
        logger.error(f"enthusiast agent failed: {e}")
        return EnthusiastOutput(
            defenses=[],
            hidden_value=[],
            culture_fit_prediction="unknown",
            hiring_recommendation="weak_invite",
        )