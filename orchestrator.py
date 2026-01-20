import os
import re
from google import genai
from dotenv import load_dotenv

from database.driver import db
from agents.constructor import constructorAgent
from agents.auditor import auditorAgent
from agents.skeptic import skepticAgent
from agents.enthusiast import enthusiastAgent


load_dotenv()

#the orchestrator should work as the project manager of the agents
class ResumeOrchestrator:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_id = "gemini-2.5-flash"
    

    #hidden func, sole purpose is to get the ID for a person
    def _extract_person_id(self, cypher_text):
        # Looks for: {id: 'something@email.com'}
        match = re.search(r"id:\s*['\"]([^'\"]+)['\"]", cypher_text)
        #return if match found else just give back none so constructor will make a uniqie Id instead
        return match.group(1) if match else None

    def lead_recruiter_agent(self, skeptic_view, enthusiast_view):
        #final decision maker
        prompt = f"""
        You are the Lead Recruiter. Review the following two conflicting reports:
        
        SKEPTIC REPORT:
        {skeptic_view}
        
        ENTHUSIAST REPORT:
        {enthusiast_view}
        
        DECISION RULES:
        1. Compare the risks vs. the growth potential.
        2. Give a final score (1-10).
        3. Make a final verdict: 'INVITE' or 'REJECT'.
        """

        response = self.client.models.generate_content(model = self.model_id, contents = prompt)
        return response.text