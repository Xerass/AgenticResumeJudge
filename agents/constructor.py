"""
The Constructor Agent
- Responsible for creating queries that will ADD new knowledge into the DB.
"""

import json
import logging
import uuid


logger = logging.getLogger("agent.constructor")

def escape_str(text):
    # simple sanitizer to prevent cypher syntax errors (stuff like ' messes up queries, we just need to make sure any text adaded passes through this to add 
    # the necessary \\)
    if not text: return ""
    return text.replace("'", "\\'").replace('"', '\\"')

async def constructorAgent(client, resume_text: str, candidate_id: str, model_id: str):
    #   we want the llm to act as a pure data extractor
    #   no reasoning here, just structuring the mess that is a resume
    #   within the query itself we try and order the llm to return it in a structured format so later logic 
    #   can operate without referencing a text blob
    
    system_instruction = """
    you are a data extraction engine. turn this resume text into strict json.
    
    rules:
    - extract work history with dates and specific tools used in each role.
    - extract projects with their tech stacks.
    - normalize skills (e.g., 'react.js' -> 'React').
    - do not summarize. preserve detail.
    
    """
    schema_enforcement= """{
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

    user_context = f"RESUME CONTENT:\n{resume_text}"

    try:
        # ask gemini for the data structure
        # we do this async via await and the async io (aio) method
        # we choose the model give it the payload (resume content)
        # then in config, we give it reposnse_mime_tyoe to application\json, which strictly enforces the "output as json" deal we have going
        # temp 0 simply means we are enforcing no outside thinking, strictly extract facts (important since we use this as the BASIS of knowledge)
        response = await client.aio.models.generate_content(
            model=model_id,
            contents=[system_instruction, schema_enforcement, user_context],
            config={
                "response_mime_type": "application/json",
                "temperature": 0.0 
            }
        )
        
        # since data is formatted as json already, we can just extract it via this
        data = json.loads(response.text)
        
        # now we switch to python mode to build the graph queries
        # this guarantees valid cypher and prevents injection attacks
        queries = []

        # create the central person node
        # we use the candidate_id passed from the orchestrator for stability
        # we enforce safe name extraction, with proper fallbacks via get
        safe_name = escape_str(data['contact'].get('name', 'Unknown Candidate'))
        safe_email = escape_str(data['contact'].get('email', ''))
        
        #query cheats:
        #MATCH is a strict search like SELECT, it will look for a pattern
        #OPTIONAL MATCH like a left outer join, goes for match, if none found return null
        # MERGE is an upsert/insert while also being a match, its a find if not found create, MERGE ((table/entity): value)
        # SET simply assigns a value to certain params 
        queries.append(
            f"MERGE (p:Person {{uid: '{candidate_id}'}}) "
            f"SET p.name = '{safe_name}', p.email = '{safe_email}'"
        )

        # loop through experience and link skills contextually
        for idx, job in enumerate(data.get('experience', [])):
            
            #create a unique id for this specific job instance
            job_uid = str(uuid.uuid4())
            
            #build the job node and link it to the person
            q_job = (
                #find person, add work experience, set workexperience role, company
                #add connection of person, to work experience
                f"MATCH (p:Person {{uid: '{candidate_id}'}}) "
                f"MERGE (w:WorkExperience {{uid: '{job_uid}'}}) "
                f"SET w.role = '{escape_str(job['role'])}', "
                f"    w.company = '{escape_str(job['company'])}' "
                f"MERGE (p)-[:HAS_EXPERIENCE]->(w)"
            )
            queries.append(q_job)

            # link specific skills found to this job, so this now acts as future reference
            # this lets us query 'years of experience' per skill later
            for skill in job.get('skills_used', []):
                clean_skill = escape_str(skill).upper() # always normalize to caps for easier matching
                q_skill = (
                    #match work experience to UID, add skill and create the connection
                    f"MATCH (w:WorkExperience {{uid: '{job_uid}'}}) "
                    f"MERGE (s:Skill {{name: '{clean_skill}'}}) "
                    f"MERGE (w)-[:USED_SKILL]->(s)"
                )
                queries.append(q_skill)

        # Project looop
        for idx, proj in enumerate(data.get('projects', [])):
            proj_uid = str(uuid.uuid4())
            
            #similar logic, create the person, add the project, create the connection
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

