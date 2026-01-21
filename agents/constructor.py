import time
import hashlib

def generate_fallback_id(text):
    return hashlib.sha256(text.encode()).hexdigest()[:10]

def constructorAgent(client, resumeText, modelId):
    #safety net
    safety_id = generate_fallback_id(resumeText)

    systemInstruction = f"""
            You are a Neo4j Graph Architect. Your task is to extract resume data into a GraphRAG schema.
            
            IDENTITY RULES:
            1. EXTRACT: Find the candidate's EMAIL.
            2. SELECTION: If an email is found, use it as the Person 'id'.
            3. FALLBACK: If NO email exists, use exactly this ID: '{safety_id}'.
            4. NAME: Extract the candidate's Full Name.

            CONSTRUCTION PATTERN:
            // Create Person and Resume
            MERGE (p:Person {{id: 'EXTRACTED_ID'}})
            SET p.name = 'FULL_NAME'
            MERGE (r:Resume {{id: 'res_' + 'EXTRACTED_ID'}})
            MERGE (p)-[:HAS_RESUME]->(r)
            
            // Create and Link Skills
            MERGE (s:Skill {{name: 'SkillName'}})
            MERGE (r)-[:MENTIONS_SKILL]->(s)

            OUTPUT: Raw Cypher only. No code blocks, no preamble.
    """

    time.sleep(2) 
    response = client.models.generate_content(
        model=modelId,
        contents=f"{systemInstruction}\n\nRESUME TEXT:\n{resumeText}"    
    )
    return response.text