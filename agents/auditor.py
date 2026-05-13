import json
import logging

from langchain_core.messages import SystemMessage, HumanMessage

#this agent needs access to the db so we need the driver
from core.llm import get_llm
from core.schemas import AuditorOutput
from database.driver import db

logger = logging.getlogger("agent.auditor")

#system prompt:
SYSTEM_PROMPT = """
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


#auditor agent call

async def auditorAgent(person_id: str, jd_text:str) -> AuditorOutput:
    #extract jd skills (simple unstructured call, should just list out skills as a list)

    extraction_llm = get_llm(temperature=0.0)
    extraction_response = await extraction_llm.ainvoke([
        HumanMessage(
            content=(
                "You are a technical recruiter. Extract a comma-separated list of "
                f"HARD technical skills only from this JD. No prose:\n\n{jd_text}"
            )
        )
    ])

    #simply parse the response into a list of strings, removing any bullet points or numbering
    jd_skills = [s.strip() for s in extraction_response.content.split(",") if s.strip()]

    #we need to query the graph next
    query = """
    MATCH (p:Person {uid: $pid})-[:HAS_EXPERIENCE]->(w:WorkExperience)-[:USED_SKILL]->(s:Skill)
    OPTIONAL MATCH (s)-[:RELATED_TO]->(related:Skill)
    RETURN s.name AS FoundSkill, related.name AS RelatedSkill
    """

    graph_results = db.run_query(query, {"pid": person_id})
    #match should find the person, check if they ahve experience connected to a skill, 
    #and also find related skills (category). these are the only things we can use.
    #insert them as evidence
    graph_evidence = []
    for record in graph_results:
        entry = {"skill": record.get("FoundSkill")}
        if record.get("RelatedSkill"):
            entry["related_to"] = record["RelatedSkill"]
        graph_evidence.append(entry)


    #force the schema
    audit_llm = get_llm(temperature=0.1).with_structured_output(AuditorOutput)

    user_context = f"""
    --- required skills (extracted from jd) ---
    {json.dumps(jd_skills)}

    --- graph evidence (candidate's verified skills) ---
    {json.dumps(graph_evidence)}

    --- job description ---
    {jd_text}
    """

    
    try:
        return await audit_llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_context),
        ])

    except Exception as e:
        logger.error(f"auditor agent failed: {e}")
        return AuditorOutput(
            skill_analysis=[],
            match_summary={"direct_matches": 0, "categorical_matches": 0, "missing_skills": 0, "match_rate": 0.0},
            assessment="weak_match",
        )