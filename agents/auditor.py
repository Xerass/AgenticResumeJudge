"""
Auditor's purpose is to discover via query related job skills based on Job Description
"""
import json
import logging

from database.driver import db

logger = logging.getLogger("agent.auditor")

async def auditorAgent(client, person_id: str, jd_text: str, model_id: str) -> dict:
    # the auditor is the fact checker of the pipeline
    # it cross-references the candidate's knowledge graph against the job description
    # yet another strict query prompt, so we need some railgaurds
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

    # strict json schema again 
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

        # as for the JD, we can just insert it as its standalone prompt
        jd_extraction_prompt = (
            "You are a technical recruiter. Extract a comma-separated list of "
            f"HARD technical skills only from this JD. No prose:\n\n{jd_text}"
        )

        # get the jd response
        extraction_response = await client.aio.models.generate_content(
            model=model_id,
            contents=jd_extraction_prompt
        )
        jd_skills_raw = extraction_response.text
        #csv style split to get skills
        jd_skills = [s.strip() for s in jd_skills_raw.split(",") if s.strip()]

        #ask the db
        #match person, if has experience, work experience, skill (get person data)
        #if skill has some relation to another skill get that too
        query = """
        MATCH (p:Person {uid: $pid})-[:HAS_EXPERIENCE]->(w:WorkExperience)-[:USED_SKILL]->(s:Skill)
        OPTIONAL MATCH (s)-[:RELATED_TO]->(related:Skill)
        RETURN s.name AS FoundSkill, related.name AS RelatedSkill
        """

        #run the query via the db  (just a search so its ok)
        graph_results = db.run_query(query, {"pid": person_id})

        # we give the llm structured facts
        graph_evidence = []

        #each record in graph results get extracted and dumped into a jason
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

        # temperature is 0.1 because auditing must be deterministic but we'll allow some creative interpretation regarding skills
        response = await client.aio.models.generate_content(
            model=model_id,
            contents=[system_instruction, schema_enforcement, user_context],
            config={
                "response_mime_type": "application/json",
                "temperature": 0.1
            }
        )

        return json.loads(response.text)

    except Exception as e:
        # never let a single agent crash the whole orchestration, if they ever do crash, just send a log and return a defaulted list.
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