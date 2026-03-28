import os
import uuid
import json
import asyncio
import logging
import time
from functools import wraps

from google import genai
from dotenv import load_dotenv

from database.driver import db
from agents.constructor import constructorAgent
from agents.auditor import auditorAgent
from agents.skeptic import skepticAgent
from agents.enthusiast import enthusiastAgent
from state import AgentState

load_dotenv()

# structured logging — every agent already uses this, the orchestrator should too
logger = logging.getLogger("orchestrator")

# Retry decorator with exponential backoff
# Wraps any async agent call so transient LLM / network failures don't kill
# the entire pipeline.  Defaults: 3 attempts, 2-second base delay.
def retry_async(max_retries: int = 3, base_delay: float = 2.0):
    """Decorator that retries an async function with exponential backoff."""
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_retries + 1):
                try:
                    return await fn(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** (attempt - 1))
                        logger.warning(
                            f"[retry] {fn.__name__} attempt {attempt}/{max_retries} "
                            f"failed: {e}  — retrying in {delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"[retry] {fn.__name__} exhausted {max_retries} attempts. "
                            f"Last error: {e}"
                        )
            raise last_exception  # re-raise so caller can handle the fallback
        return wrapper
    return decorator


class ResumeOrchestrator:
    """Manages the multi-agent resume analysis pipeline.

    Pipeline stages:
        1. Constructor  → builds a Neo4j knowledge graph from resume text
        2. Auditor      → cross-references graph skills against the JD
        3. Skeptic ‖ Enthusiast  → run in parallel for opposing analyses
        4. Lead Recruiter        → synthesises a final verdict
    """

    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_id = "gemini-2.5-flash"


    # Wrapped agent calls, each gets retry + backoff automatically


    @retry_async()
    async def _run_constructor(self, resume_text: str, candidate_id: str) -> list:
        """Parse the resume and return a list of Cypher queries."""
        return await constructorAgent(
            self.client, resume_text, candidate_id, self.model_id
        )

    @retry_async()
    async def _run_auditor(self, person_id: str, jd_text: str) -> dict:
        """Audit candidate skills against the job description."""
        return await auditorAgent(
            self.client, person_id, jd_text, self.model_id
        )

    @retry_async()
    async def _run_skeptic(self, auditor_data: dict, jd_text: str) -> dict:
        """Identify technical risk and red flags."""
        return await skepticAgent(
            self.client, auditor_data, jd_text, self.model_id
        )

    @retry_async()
    async def _run_enthusiast(self, auditor_data: dict, jd_text: str) -> dict:
        """Defend growth potential and transferable skills."""
        return await enthusiastAgent(
            self.client, auditor_data, jd_text, self.model_id
        )

    @retry_async()
    async def _run_lead_recruiter(self, skeptic_view: dict, enthusiast_view: dict) -> str:
        """Final decision maker — weighs risk vs. potential."""
        prompt = f"""
        You are the Lead Recruiter. Review the following two conflicting reports:

        SKEPTIC REPORT:
        {json.dumps(skeptic_view, indent=2)}

        ENTHUSIAST REPORT:
        {json.dumps(enthusiast_view, indent=2)}

        DECISION RULES:
        1. Compare the risks vs. the growth potential.
        2. Give a final score (1-10).
        3. Make a final verdict: 'INVITE' or 'REJECT'.
        """
        response = await self.client.aio.models.generate_content(
            model=self.model_id, contents=prompt
        )
        return response.text

    
    # Main pipeline
    

    async def run_workflow(self, resume_text: str, jd_text: str) -> AgentState:
        """Execute the full multi-agent pipeline and return the final state."""
        pipeline_start = time.perf_counter()

        # generate a stable candidate ID upfront so every agent references
        # the same node — no more brittle regex extraction from Cypher text
        candidate_id = str(uuid.uuid4())

        state: AgentState = {
            "resume_text": resume_text,
            "jd_text": jd_text,
            "candidate_id": candidate_id,
            "graph_queries": [],
            "auditor_report": {},
            "skeptic_critique": {},
            "enthusiast_pitch": {},
            "final_verdict": "",
        }

        # ── Stage 1: Constructor ──────────────────────────────────────
        logger.info("[1/4] Constructor Agent building knowledge graph...")
        t0 = time.perf_counter()

        try:
            queries = await self._run_constructor(resume_text, candidate_id)
            state["graph_queries"] = queries

            # execute each query the constructor produced
            for query in queries:
                if query and query.strip():
                    db.run_query(query)

            logger.info(
                f"[1/4] Constructor done — {len(queries)} queries executed "
                f"({time.perf_counter() - t0:.2f}s)"
            )
        except Exception as e:
            logger.error(f"[1/4] Constructor failed after retries: {e}")
            return state  # can't continue without a knowledge graph

        # ── Stage 2: Auditor ──────────────────────────────────────────
        logger.info("[2/4] Auditor Agent analyzing hard skills...")
        t0 = time.perf_counter()

        try:
            audit_report = await self._run_auditor(candidate_id, jd_text)
            state["auditor_report"] = audit_report
            logger.info(f"[2/4] Auditor done ({time.perf_counter() - t0:.2f}s)")
        except Exception as e:
            logger.error(f"[2/4] Auditor failed after retries: {e}")
            return state  # skeptic & enthusiast depend on the audit

        # ── Stage 3: Skeptic ‖ Enthusiast (parallel fan-out) ──────────
        logger.info("[3/4] Running Skeptic and Enthusiast in parallel...")
        t0 = time.perf_counter()

        skeptic_result, enthusiast_result = await asyncio.gather(
            self._safe_agent_call(
                self._run_skeptic(audit_report, jd_text),
                fallback={"risk_score": 50, "verdict_recommendation": "manual_review", "error": "agent_timeout"},
                label="Skeptic",
            ),
            self._safe_agent_call(
                self._run_enthusiast(audit_report, jd_text),
                fallback={"hiring_recommendation": "manual_review", "error": "agent_timeout"},
                label="Enthusiast",
            ),
        )

        state["skeptic_critique"] = skeptic_result
        state["enthusiast_pitch"] = enthusiast_result
        logger.info(f"[3/4] Parallel judges done ({time.perf_counter() - t0:.2f}s)")

        # ── Stage 4: Lead Recruiter ───────────────────────────────────
        logger.info("[4/4] Lead Recruiter making final verdict...")
        t0 = time.perf_counter()

        try:
            verdict = await self._run_lead_recruiter(skeptic_result, enthusiast_result)
            state["final_verdict"] = verdict
        except Exception as e:
            logger.error(f"[4/4] Lead Recruiter failed: {e}")
            state["final_verdict"] = "MANUAL REVIEW REQUIRED — lead recruiter agent unavailable."

        elapsed = time.perf_counter() - pipeline_start
        logger.info(f"Pipeline complete in {elapsed:.2f}s for candidate {candidate_id}")

        return state

    
    # Helpers


    @staticmethod
    async def _safe_agent_call(coro, *, fallback: dict, label: str) -> dict:
        """Run an agent coroutine; on failure return the fallback instead of crashing."""
        try:
            return await coro
        except Exception as e:
            logger.warning(f"{label} agent failed gracefully — using fallback. Error: {e}")
            return fallback