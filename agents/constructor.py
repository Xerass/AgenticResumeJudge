import json
import logging
import uuid

# setup logging to catch errors without crashing
logger = logging.getLogger("agent.constructor")

async def constructorAgent(client, resume_text: str, candidate_id: str, model_id: str):
    # we want the llm to act as a pure data extractor
    # no reasoning here, just structuring the mess that is a resume
    system_instruction = """
    you are a data extraction engine. turn this resume text into strict json.
    
    rules:
    - extract work history with dates and specific tools used in each role.
    - extract projects with their tech stacks.
    - normalize skills (e.g., 'react.js' -> 'React').
    - do not summarize. preserve detail.
    
    output schema:
    {
        "contact": {"name": "str", "email": "str"},
        "experience": [
            {
                "role": "str",
                "company": "str",
                "description": "str",
                "skills_used": ["str"] // this is the gold mine. map skills to roles.
            }
        ],
        "projects": [
            {
                "title": "str", 
                "tech_stack": ["str"]
            }
        ]
    }
    """

    try:
        # ask gemini for the data structure
        response = await client.models.generate_content_async(
            model=model_id,
            contents=[system_instruction, f"RESUME CONTENT:\n{resume_text}"],
            config={"response_mime_type": "application/json"}
        )
        
        data = json.loads(response.text)
        
        # now we switch to python mode to build the graph queries
        # this guarantees valid cypher and prevents injection attacks
        queries = []

        # 1. create the central person node
        # we use the candidate_id passed from the orchestrator for stability
        safe_name = escape_str(data['contact'].get('name', 'Unknown Candidate'))
        safe_email = escape_str(data['contact'].get('email', ''))
        
        queries.append(
            f"MERGE (p:Person {{uid: '{candidate_id}'}}) "
            f"SET p.name = '{safe_name}', p.email = '{safe_email}'"
        )

        # 2. loop through experience and link skills contextually
        for idx, job in enumerate(data.get('experience', [])):
            # create a unique id for this specific job instance
            job_uid = str(uuid.uuid4())
            
            # build the job node and link it to the person
            q_job = (
                f"MATCH (p:Person {{uid: '{candidate_id}'}}) "
                f"MERGE (w:WorkExperience {{uid: '{job_uid}'}}) "
                f"SET w.role = '{escape_str(job['role'])}', "
                f"    w.company = '{escape_str(job['company'])}' "
                f"MERGE (p)-[:HAS_EXPERIENCE]->(w)"
            )
            queries.append(q_job)

            # now the magic: link skills to THIS job
            # this lets us query 'years of experience' per skill later
            for skill in job.get('skills_used', []):
                clean_skill = escape_str(skill).upper() # always normalize to caps for easier matching
                q_skill = (
                    f"MATCH (w:WorkExperience {{uid: '{job_uid}'}}) "
                    f"MERGE (s:Skill {{name: '{clean_skill}'}}) "
                    f"MERGE (w)-[:USED_SKILL]->(s)"
                )
                queries.append(q_skill)

        # 3. loop through projects
        for idx, proj in enumerate(data.get('projects', [])):
            proj_uid = str(uuid.uuid4())
            
            q_proj = (
                f"MATCH (p:Person {{uid: '{candidate_id}'}}) "
                f"MERGE (pr:Project {{uid: '{proj_uid}'}}) "
                f"SET pr.name = '{escape_str(proj['title'])}' "
                f"MERGE (p)-[:BUILT_PROJECT]->(pr)"
            )
            queries.append(q_proj)

            # link project tech stack
            for tech in proj.get('tech_stack', []):
                clean_tech = escape_str(tech).upper()
                q_tech = (
                    f"MATCH (pr:Project {{uid: '{proj_uid}'}}) "
                    f"MERGE (s:Skill {{name: '{clean_tech}'}}) "
                    f"MERGE (pr)-[:USED_SKILL]->(s)"
                )
                queries.append(q_tech)

        return queries

    except Exception as e:
        logger.error(f"constructor agent failed: {e}")
        return []

def escape_str(text):
    # simple sanitizer to prevent cypher syntax errors
    if not text: return ""
    return text.replace("'", "\\'").replace('"', '\\"')