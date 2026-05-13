"""
The Constructor Agent
- Responsible for creating queries that will ADD new knowledge into the DB.
"""

import logging
import uuid

from langchain_core.messages import SystemMessage, HumanMessage

from core.llm import get_llm
from core.schemas import ConstructorOutput

logger = logging.getLogger("agent.constructor")


def escape_str(text):
    if not text: return ""
    return text.replace("'", "\\'").replace('"', '\\"')


SYSTEM_PROMPT = """
you are a data extraction engine. turn this resume text into strict json.

rules:
- extract work history with dates and specific tools used in each role.
- extract projects with their tech stacks.
- normalize skills (e.g., 'react.js' -> 'React').
- do not summarize. preserve detail.
"""


async def constructorAgent(resume_text: str, candidate_id: str) -> list[str]:
    llm = get_llm(temperature=0.0).with_structured_output(ConstructorOutput)

    try:
        data: ConstructorOutput = await llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"RESUME CONTENT:\n{resume_text}"),
        ])

        queries = []

        safe_name = escape_str(data.contact.name or "Unknown Candidate")
        safe_email = escape_str(data.contact.email or "")

        # MERGE = upsert find the Person by uid, create if doesn't exist
        # SET assigns name and email as properties on that node
        queries.append(
            f"MERGE (p:Person {{uid: '{candidate_id}'}}) "
            f"SET p.name = '{safe_name}', p.email = '{safe_email}'"
        )

        for job in data.experience:
            job_uid = str(uuid.uuid4())

            # MATCH finds the Person we just created
            # MERGE upserts a WorkExperience node keyed by its unique id
            # SET assigns role and company as properties
            # final MERGE draws the edge: Person -[:HAS_EXPERIENCE]-> WorkExperience
            q_job = (
                f"MATCH (p:Person {{uid: '{candidate_id}'}}) "
                f"MERGE (w:WorkExperience {{uid: '{job_uid}'}}) "
                f"SET w.role = '{escape_str(job.role)}', "
                f"    w.company = '{escape_str(job.company)}' "
                f"MERGE (p)-[:HAS_EXPERIENCE]->(w)"
            )
            queries.append(q_job)

            for skill in job.skills_used:
                clean_skill = escape_str(skill).upper()
                # MATCH finds the WorkExperience node we just created
                # MERGE upserts the Skill node (so "REACT" only exists once globally)
                # final MERGE draws the edge: WorkExperience -[:USED_SKILL]-> Skill
                # this links skills to specific jobs, letting us query experience-per-skill later
                queries.append(
                    f"MATCH (w:WorkExperience {{uid: '{job_uid}'}}) "
                    f"MERGE (s:Skill {{name: '{clean_skill}'}}) "
                    f"MERGE (w)-[:USED_SKILL]->(s)"
                )

        for proj in data.projects:
            proj_uid = str(uuid.uuid4())

            # same pattern upsert Project node, link it to Person via BUILT_PROJECT edge
            queries.append(
                f"MATCH (p:Person {{uid: '{candidate_id}'}}) "
                f"MERGE (pr:Project {{uid: '{proj_uid}'}}) "
                f"SET pr.name = '{escape_str(proj.title)}' "
                f"MERGE (p)-[:BUILT_PROJECT]->(pr)"
            )

            for tech in proj.tech_stack:
                # upsert Skill node, link to Project via USED_SKILL edge
                queries.append(
                    f"MATCH (pr:Project {{uid: '{proj_uid}'}}) "
                    f"MERGE (s:Skill {{name: '{escape_str(tech).upper()}'}}) "
                    f"MERGE (pr)-[:USED_SKILL]->(s)"
                )

        return queries

    except Exception as e:
        logger.error(f"constructor agent failed: {e}")
        return []