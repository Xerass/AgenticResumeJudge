"""
command line entry point for testing
"""

import asyncio
import logging

#cli building package
import typer 

from agenticresume.domain.models import AnalysisResult, CareerProfile, JobPost, Reframe
from agenticresume.graph.state import ScreeningState
from agenticresume.infra.documents import read_document
from agenticresume.services.screening import screen_resume
from agenticresume.settings import get_settings

app = typer.Typer(help = "Screens a resume against a job description")


#app.command creates a cli command in typer
@app.command(name = "screen", help = "Screens a resume against a job description")
def screen(resume: str, job:str) -> None:
    """
    Screen RESUME against JOB (paths to .pdf/.txt/.md files).
    """

    settings = get_settings() #config enters here
    logging.basicConfig(level=settings.log_level) #entry point configures logging


    resume_text = read_document(resume)
    jd_text = read_document(job)

    state = asyncio.run(screen_resume(settings, resume_text, jd_text))
    _print_verdict(state["result"])
    _print_tailoring(state)

def _print_verdict(r: AnalysisResult) -> None:
    typer.echo(f"\nDECISION: {r.decision.upper()}   (score {r.score:.0%})")
    typer.echo(f"\n{r.rationale}\n")
    for a in r.assessments:
        typer.echo(f"[{a.persona}] {a.summary}")
        for p in a.points:
            typer.echo(f"  - {p}")

def _print_tailoring(state: ScreeningState) -> None:
    """Render the tailoring plan, resolving UUID refs back to text via the state."""
    reframes: list[Reframe] = state.get("tailoring", [])
    if not reframes:
        return

    profile: CareerProfile = state["profile"]
    job_post: JobPost = state["job_post"]
    req_text = {r.id: r.text for r in job_post.requirements}
    fact_text = {f.id: f.text for f in profile.active_facts}

    typer.echo("\nTAILORING PLAN:")
    for rf in reframes:
        typer.echo(f"\n[{rf.move}] {req_text.get(rf.requirement_id, '?')}")
        if rf.source_fact_id is not None:
            typer.echo(f"  from: {fact_text.get(rf.source_fact_id, '?')}")
        if rf.proposed_text:
            typer.echo(f"  ->   {rf.proposed_text}")
        if rf.rationale:
            typer.echo(f"  ({rf.rationale})")


def main() -> None:
    app()