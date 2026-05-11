import json
import logging

#system messages sets the agent role/personality
#human message is the actualinput query

from langchain_core.messages import SystemMessage, HumanMessage

from core.llm import get_llm
from core.schemas import SkepticOutput

logger = logging.getLogger("skeptic")


#defines model thought process and behavior
SYSTEM_PROMPT = """
you are the 'technical risk' assessor. your job is to identify why this candidate might fail.

input context:
1. auditor_findings: the raw skills gap analysis.
2. job_description: the strict requirements.

your task:
analyze the gaps found by the auditor and calculate 'velocity risk' (how long until they are productive?).

rules:
- if a 'required' skill is missing, that is a high risk.
- if a skill is a 'partial match' (e.g., they know vue but need react), flag it as 'ramp-up required'.
- look for 'dependency chains': if they are missing python, they probably can't do the django work either.
- be direct. do not sugarcoat.
"""

async def skepticAgent(auditor_data:dict, jd_text:str) -> SkepticOutput:

    #langchain interface with setting up the model, with the schema we enforced
    llm = get_llm(temperature=0.2).with_structured_output(SkepticOutput)

    #supplies context necessary for judgement
    user_context = f"""
    --- auditor findings ---
    {json.dumps(auditor_data)}

    --- job description ---
    {jd_text}
    """

    try:
        return await llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_context)  
        ]) #ainvoke is the async cal for model, supply with bnoth system and human message

    except Exception as e:
        logger.error(f"skeptic agent failed: {e}")

        #defaults to a neutral judgement if it fails
        return SkepticOutput(
            red_flags=[],
            velocity_risk="medium_ramp_up",
            estimated_ramp_up_time="unknown",
            risk_score=50,
            verdict_recommendation="proceed_with_caution",
        )
    

