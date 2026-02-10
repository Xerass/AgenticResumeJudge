import json
import logging

# set up a logger because print statements are for amateurs
logger = logging.getLogger("agent.enthusiast")

async def enthusiastAgent(client, auditor_data: dict, jd_text: str, model_id: str) -> dict:
    # prompt engineering: we don't just ask for a summary. we enforce a specific 'thinking path'
    # we want the agent to act like a defense attorney cross-examining the auditor's report
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

    # we dump the dict to a string for the prompt, but we keep the structure
    # this lets the model see the exact keys the auditor outputted
    user_context = f"""
    --- auditor report (the prosecution) ---
    {json.dumps(auditor_data)}

    --- job description (the law) ---
    {jd_text}
    """

    # pro tip: use a schema definition in the prompt or use the strictly typed generation config
    # here we explicitly describe the json schema we want back to ensure stability
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

    try:
        # the call is async so we don't block the event loop
        # we force the response mime type to json so the model doesn't yap
        response = await client.models.generate_content_async(
            model=model_id,
            contents=[system_instruction, schema_enforcement, user_context],
            config={
                "response_mime_type": "application/json",
                "temperature": 0.3 # keep it creative enough to find connections, but not hallucinate
            }
        )

        # parse immediately. if this fails, the agent is broken
        return json.loads(response.text)

    except Exception as e:
        # never let a single agent crash the whole orchestration
        logger.error(f"enthusiast agent failed: {e}")
        return {
            "error": "agent_failure", 
            "hiring_recommendation": "manual_review"
        }