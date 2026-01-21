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

    def run_workflow(self, resume_text, jd_text):
        #orders the agents to do their work step by step
        print('[ORCHESTRATOR] Constructor Agent builing knowledge graph...')
        cypher_code = constructorAgent(self.client, resume_text, self.model_id)

        
        person_id = self._extract_person_id(cypher_code)
        
        if not person_id:
            print("Error: Could not extract Person ID from generated Cypher.")
            return None

        print(f"Writing candidate {person_id} to Neo4j...")
        for statement in cypher_code.split(';'):
            if statement.strip():
                db.run_query(statement)

        #ket auditor analyze the skills via knowledge graph
        print('[ORCHESTRATOR] Auditor Agent analyzing hard skills...')
        audit_report = auditorAgent(self.client, person_id, jd_text, self.model_id)

        print('[ORCHESTRATOR] Skeptic Agent evaluating red flags...')
        skeptic_report = skepticAgent(self.client, resume_text, jd_text, self.model_id)

        print('[ORCHESTRATOR] Enthusiast Agent highlighting growth potential...')
        enthusiast_report = enthusiastAgent(self.client, resume_text, jd_text, self.model_id)

        print('[ORCHESTRATOR] Lead Recruiter Agent making final verdict...')
        final_verdict = self.lead_recruiter_agent(skeptic_report, enthusiast_report)    

        return {
            "candidate_id": person_id,
            "verdict": final_verdict,
            "audit": audit_report
        }
    
    