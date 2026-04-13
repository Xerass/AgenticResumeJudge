import json
import logging

logger = logging.getLogger("agent.enthusiast")

async def enthusiastAgent(client, auditor_data: dict, jd_text: str, model_id: str) -> dict:
    # we want the agent to act like a defense attorney cross-examining the auditor's report
    # should counteract with skeptic, bring a brighter POV into the mix
    system_instruction = """
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
    - return raw json only.
    - do not be delusional. if a gap is real (e.g., missing 5 years of management), admit it.
    """

    user_context = f"""
    --- auditor report (the prosecution) ---
    {json.dumps(auditor_data)}

    --- job description (the law) ---
    {jd_text}
    """

    #schema again, but this time more on positive traits, wow so amazing
    schema_enforcement = """
    return this exact json structure:
    {
        "defenses": [
            {
                "gap_identified": "string (the skill missing)",
                "defense_argument": "string (why they can learn it fast)",
                "validity_score": "int (1-10 confidence in this transfer)"
            }
        ],
        "hidden_value": ["list of soft skills or outlier tech that adds value"],
        "culture_fit_prediction": "string (based on tone/projects)",
        "hiring_recommendation": "strong_invite | invite | weak_invite"
    }
    """

    #basically the same stuff as last time...
    try:

        response = await client.aio.models.generate_content(
            model=model_id,
            contents=[system_instruction, schema_enforcement, user_context],
            config={
                "response_mime_type": "application/json",
                "temperature": 0.3 # keep it creative enough to find connections but not hallucinate
            }
        )

        return json.loads(response.text)

    except Exception as e:
        logger.error(f"enthusiast agent failed: {e}")
        return {
            "error": "agent_failure", 
            "hiring_recommendation": "manual_review"
        }