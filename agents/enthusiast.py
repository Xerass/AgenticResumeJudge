def enthusiastAgent(client, auditorReport, jdText, modelId):
    systemPrompt = """
    You are a Growth-Focused Recruiter. Your goal is to find potential.
    Look at the Auditor's Report. 
    1. Celebrate 'Categorical Matches' (e.g., 'They know FastAPI, which means they'll master our Backend stack in days').
    2. Focus on transferable skills and the candidate's technical breadth.
    3. Be optimistic. Focus on long-term value and cultural/skill-set fit.
    """
    
    context = f"AUDITOR REPORT:\n{auditorReport}\n\nJOB DESCRIPTION:\n{jdText}"
    response = client.models.generate_content(model=modelId, contents=f"{systemPrompt}\n\n{context}")
    return response.text