import json
import logging

from database.driver import db

# standard logging pattern 
logger = logging.getLogger("agent.auditor")

async def auditorAgent(client, person_id: str, jd_text: str, model_id: str) -> dict:
    # the auditor is the fact checker of the pipeline
    # it cross-references the candidate's knowledge graph against the job description
    # every claim it makes is backed by a graph query, not llm hallucination
    system_instruction = """
    you are the 'technical skills auditor'. your job is to compare a candidate's verified skills
    against a job description and produce a structured gap analysis.

    input context:
    1. graph_evidence: real skill data pulled from a neo4j knowledge graph.
    2. job_description: the target role requirements.
    3. jd_skills: the extracted list of required hard skills.

    your task:
    - classify each required skill as 'direct_match', 'categorical_match', or 'missing'.
    - a 'direct_match' means the candidate has the exact skill.
    - a 'categorical_match' means the candidate has a related skill in the same category.
    - 'missing' means no evidence was found in the graph.
    - do not guess or infer. only use the graph evidence provided.

    rules:
    - be precise. this report feeds into the skeptic and enthusiast agents downstream.
    - do not add skills that are not in the graph evidence.
    - preserve the original skill names from the job description.
    """

    # strict json schema — same pattern as skeptic and enthusiast agents
    # this allows downstream agents to programmatically access findings
    schema_enforcement = """
    return this exact json structure:
    {
        "skill_analysis": [
            {
                "required_skill": "string (from the job description)",
                "status": "direct_match | categorical_match | missing",
                "candidate_skill": "string (what they actually have, or null)",
                "category": "string (shared category if categorical match, or null)"
            }
        ],
        "match_summary": {
            "direct_matches": "int",
            "categorical_matches": "int",
            "missing_skills": "int",
            "match_rate": "float (0.0 to 1.0)"
        },
        "assessment": "strong_match | partial_match | weak_match"
    }
    """

    try:
        # step 1: extract required hard skills from the jd using the llm
        # we keep this as a focused, single-purpose call
        jd_extraction_prompt = (
            "You are a technical recruiter. Extract a comma-separated list of "
            f"HARD technical skills only from this JD. No prose:\n\n{jd_text}"
        )

        extraction_response = await client.models.generate_content_async(
            model=model_id,
            contents=jd_extraction_prompt
        )

        jd_skills_raw = extraction_response.text
        jd_skills = [s.strip() for s in jd_skills_raw.split(",") if s.strip()]

        # step 2: query the knowledge graph for factual skill evidence
        # this is the auditor's superpower — it doesn't guess, it queries
        query = """
        MATCH (p:Person {uid: $pid})-[:HAS_EXPERIENCE]->(w:WorkExperience)-[:USED_SKILL]->(s:Skill)
        OPTIONAL MATCH (s)-[:RELATED_TO]->(related:Skill)
        RETURN s.name AS FoundSkill, related.name AS RelatedSkill
        """

        graph_results = db.run_query(query, {"pid": person_id})

        # step 3: format the graph evidence as context for the llm
        # we give the llm structured facts, not raw cypher output
        graph_evidence = []
        for record in graph_results:
            found = record.get("FoundSkill")
            related = record.get("RelatedSkill")
            entry = {"skill": found}
            if related:
                entry["related_to"] = related
            graph_evidence.append(entry)

        # build the user context with all three data sources
        user_context = f"""
        --- required skills (extracted from jd) ---
        {json.dumps(jd_skills)}

        --- graph evidence (candidate's verified skills) ---
        {json.dumps(graph_evidence)}

        --- job description ---
        {jd_text}
        """

        # step 4: ask the llm to produce the structured gap analysis
        # temperature is 0.1 because auditing must be deterministic
        response = await client.models.generate_content_async(
            model=model_id,
            contents=[system_instruction, schema_enforcement, user_context],
            config={
                "response_mime_type": "application/json",
                "temperature": 0.1
            }
        )

        # parse immediately — if this fails, the agent is broken
        return json.loads(response.text)

    except Exception as e:
        # never let a single agent crash the whole orchestration
        logger.error(f"auditor agent failed: {e}")
        return {
            "skill_analysis": [],
            "match_summary": {
                "direct_matches": 0,
                "categorical_matches": 0,
                "missing_skills": 0,
                "match_rate": 0.0
            },
            "assessment": "error",
            "error": "agent_failure"
        }