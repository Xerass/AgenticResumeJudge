import json
import logging

# standard logging pattern
logger = logging.getLogger("agent.skeptic")

async def skepticAgent(client, auditor_data: dict, jd_text: str, model_id: str) -> dict:
    # the system prompt defines the "personality" and the "mission"
    # here, we want a "technical gatekeeper" who protects the team from bad hires
    system_instruction = """
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

    # dump the input dict to string for the llm
    user_context = f"""
    --- auditor findings ---
    {json.dumps(auditor_data)}

    --- job description ---
    {jd_text}
    """

    # strict json schema. this allows us to mathematically weigh the risk later
    # e.g., if risk_score > 80, auto-reject
    schema_enforcement = """
    return this exact json structure:
    {
        "red_flags": [
            {
                "issue": "string (e.g., missing primary language)",
                "impact": "critical | major | minor",
                "reasoning": "string (why this hurts the team)"
            }
        ],
        "velocity_risk": "string (low_ramp_up | medium_ramp_up | high_ramp_up)",
        "estimated_ramp_up_time": "string (e.g., '2-4 weeks')",
        "risk_score": "int (0-100, where 100 is extremely risky)",
        "verdict_recommendation": "reject | proceed_with_caution | interview"
    }
    """

    try:
        # async call to keep the pipeline fast
        # lower temperature (0.2) because risk assessment should be cold and logical, not creative
        response = await client.models.generate_content_async(
            model=model_id,
            contents=[system_instruction, schema_enforcement, user_context],
            config={
                "response_mime_type": "application/json",
                "temperature": 0.2 
            }
        )

        return json.loads(response.text)

    except Exception as e:
        logger.error(f"skeptic agent failed: {e}")
        # fail safe: if the skeptic crashes, assume high risk to be safe
        return {
            "risk_score": 50,
            "verdict_recommendation": "manual_review",
            "error": "agent_crashed"
        }