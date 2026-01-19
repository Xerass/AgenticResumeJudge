def skepticAgent(client, auditorReport, jdText, modelId):
    systemPrompt = """
    You are a Critical Hiring Manager. Your goal is to find red flags.
    Look at the Auditor's Report. 
    1. If a skill is a 'Categorical Match', treat it as a risk (e.g., 'They know Vue, but we need React - there will be a learning curve').
    2. Highlight 'Missing' requirements that the graph didn't find.
    3. Be cynical but professional. Focus on immediate project velocity.
    """
    
    context = f"AUDITOR REPORT:\n{auditorReport}\n\nJOB DESCRIPTION:\n{jdText}"
    response = client.models.generate_content(model=modelId, contents=f"{systemPrompt}\n\n{context}")
    return response.text