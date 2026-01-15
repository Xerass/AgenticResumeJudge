import time

def constructorAgent(client, resumeText, modelId):
    #a function whose only purpose is defining the rules of the graph and constructing queries to build the neo4j graph
    systemInstruction = """
    You are a Neo4j Graph Architect. Your task is to extract technical skills 
    from a resume and format them as Cypher MERGE statements.
    
    RULES:
    1. Create a (:Person {name: 'Candidate'}) node.
    2. For every technical skill found, create a (:Skill {name: 'SkillName'}) node.
    3. Create a -[:HAS_SKILL]-> relationship from the Person to the Skill.
    4. Ensure skill names are capitalized consistently (e.g., 'Python', 'FastAPI').
    5. ONLY output raw Cypher code. No markdown code blocks, no explanations.
    
    EXAMPLE OUTPUT:
    MERGE (p:Person {name: 'Candidate'})
    MERGE (s:Skill {name: 'Python'})
    MERGE (p)-[:HAS_SKILL]->(s)
    """

    #time out responses after 2 seconds to provide more time to recover requests per minute
    time.sleep(2)

    #build the response
    response = client.models.generate_content(
        model = modelId,
        contents=f"{systemInstruction}\n\nRESUME TEXT:\n{resumeText}"    
    )

    return response.text