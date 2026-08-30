""" Screening Service, entry point into the LangGraph"""


from agenticresume.domain.models import AnalysisResult
from agenticresume.graph.build import build_graph
from agenticresume.settings import Settings

#async endpoint for screening, just calls the build graph
async def screen_resume(settings: Settings, resume_text: str, jd_text: str) -> AnalysisResult:
    """Screen a resume against a job description"""
    app = build_graph(settings = settings)
    final = await app.ainvoke(
        {
            "resume_text": resume_text,
            "jd_text": jd_text
        }
    )
    return final["result"]