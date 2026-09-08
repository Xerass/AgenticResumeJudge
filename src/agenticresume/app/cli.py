"""
command line entry point for testing
"""

import asyncio
import logging

#cli building package
import typer 

from agenticresume.domain.models import AnalysisResult
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

    result = asyncio.run(screen_resume(settings, resume_text, jd_text))
    _print_verdict(result)

def _print_verdict(r: AnalysisResult) -> None:
    typer.echo(f"\nDECISION: {r.decision.upper()}   (score {r.score:.0%})")
    typer.echo(f"\n{r.rationale}\n")
    for a in r.assessments:
        typer.echo(f"[{a.persona}] {a.summary}")
        for p in a.points:
            typer.echo(f"  - {p}")


def main() -> None:
    app()