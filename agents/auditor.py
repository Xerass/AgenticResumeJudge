#the auditor of the agents, this focuses on turning the unstructured pdf data into structured facts with nodes and relationships from the database and provides it to the judges

import time
from database.driver import db

def auditorAgent(client, personId, jdText, modelId):
    #a fact checker, queries neo4j to find direct and indirect skill matches

    #ask gemini to extract required skills
    #search the graph
    jdExtractionPrompt = (
        "You are a technical recruiter. Extract a comma-separated list of "
        f"HARD technical skills only from this JD. No prose:\n\n{jdText}"
    )

    #try-except or check for empty response to be safe
    response = client.models.generate_content(model=modelId, contents=jdExtractionPrompt)
    jdSkillsRaw = response.text

    jdSkills = [s.strip() for s in jdSkillsRaw.split(",") if s.strip()]

    #graph query
    query = """
    MATCH (p:Person {id: $pid})-[:HAS_RESUME]->(r)-[:MENTIONS_SKILL]->(s:Skill)
    OPTIONAL MATCH (s)-[:IS_PART_OF|LANGUAGE_FOR]->(cat:Category)<-[:IS_PART_OF|LANGUAGE_FOR]-(related:Skill)
    WHERE s.name IN $jdSkills OR related.name IN $jdSkills
    RETURN s.name as FoundSkill, cat.name as Category, related.name as RelatedRequirement
    """

    results = db.run_query(query, {"pid":personId, "jdSkills": jdSkills})

    #format the findings for other agents to use
    report = "Auditor Graph Report:\n"
    if not results:
        report += "No direct or categorical matches found in the knowledge graph."
    
    #set() to avoid duplicate lines in the report if a skill has multiple paths
    seen_entries = set()
    
    for rec in results:

        found = rec.get('FoundSkill')
        category = rec.get('Category')
        req = rec.get('RelatedRequirement')

        if found in jdSkills:
            line = f"-DIRECT MATCH: {found}"
        elif req and req in jdSkills:
            line = f"-CATEGORICAL MATCH: Candidate has **{found}**, which is related to required **{req}** via {category}"
        else:
            continue
        
        if line not in seen_entries:
            report += line + "\n"
            seen_entries.add(line)

    return report