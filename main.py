from pypdf import PdfReader
from orchestrator import ResumeOrchestrator

def read_pdf(file_path):
    reader = PdfReader(file_path)
    return " ".join([page.extract_text() for page in reader.pages])


def read_text(file_path):
    with open(file_path, 'r') as file:
        return file.read()
    
if __name__ == "__main__":
    #start orch
    orchestrator = ResumeOrchestrator()

    #load data
    resume = read_pdf("data/sample_resume.pdf")
    jd = read_text ("data/sample_jd.txt")

    #run engine
    results = orchestrator.run_workflow(resume, jd)

    #print a summary
    print("\n" + "="*50)
    print(f"RESULT FOR: {results['id']}")
    print("="*50)
    print(results['verdict'])