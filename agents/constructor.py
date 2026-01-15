import time
import hashlib

def generate_fallback_id(text):
    return hashlib.sha256(text.encode()).hexdigest()[:10]

def constructorAgent(client, resumeText, modelId):
    #safety net
    safety_id = generate_fallback_id(resumeText)

    systemInstruction = f"""
    You are a Neo4j Graph Architect. Your goal is to map a resume to a graph.
    
    IDENTITY RULES:
    1. Look for the candidate's EMAIL. If found, use it as the Person 'id'.
    2. If NO email is found, use this safety ID: '{safety_id}'.
    3. Extract the candidate's Full Name.

    CONSTRUCTION PATTERN:
    MERGE (p:Person {{id: 'EXTRACTED_EMAIL_OR_SAFETY_ID'}})
    SET p.name = 'FULL_NAME'
    MERGE (r:Resume {{id: 'res_' + 'EXTRACTED_EMAIL_OR_SAFETY_ID'}})
    MERGE (p)-[:HAS_RESUME]->(r)
    
    SKILL RULES:
    - Extract tech skills (e.g., Python, Docker).
    - MERGE (s:Skill {{name: 'SkillName'}})
    - MERGE (r)-[:MENTIONS_SKILL]->(s)

    OUTPUT: Raw Cypher only. No markdown backticks.
    """

    time.sleep(2) 
    response = client.models.generate_content(
        model=modelId,
        contents=f"{systemInstruction}\n\nRESUME TEXT:\n{resumeText}"    
    )
    return response.text