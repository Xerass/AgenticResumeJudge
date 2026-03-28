import asyncio
import logging

from pypdf import PdfReader
from orchestrator import ResumeOrchestrator

# configure logging so orchestrator + agent logs are visible in the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)


def read_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    return " ".join([page.extract_text() for page in reader.pages])


def read_text(file_path: str) -> str:
    with open(file_path, "r") as file:
        return file.read()


async def main():
    orchestrator = ResumeOrchestrator()

    # load data
    resume = read_pdf("data/sample_resume.pdf")
    jd = read_text("data/sample_jd.txt")

    # run the async pipeline
    results = await orchestrator.run_workflow(resume, jd)

    # print a summary
    print("\n" + "=" * 50)
    print(f"RESULT FOR: {results['candidate_id']}")
    print("=" * 50)
    print(results["final_verdict"])


if __name__ == "__main__":
    asyncio.run(main())