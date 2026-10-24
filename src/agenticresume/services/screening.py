""" Screening Service, entry point into the LangGraph"""


from agenticresume.graph.build import build_graph
from agenticresume.graph.state import ScreeningState
from agenticresume.settings import Settings

#async endpoint for screening, just calls the build graph
async def screen_resume(settings: Settings, resume_text: str, jd_text: str) -> ScreeningState:
    """Screen a resume against a job description.

    Returns the full final state: one run now yields two products — the
    verdict (`result`) and, on a hold/reject, a tailoring plan (`tailoring`).
    An API layer can still pick out `state["result"]`.
    """
    app = build_graph(settings = settings)
    final = await app.ainvoke(
        {
            "resume_text": resume_text,
            "jd_text": jd_text
        }
    )
    return final