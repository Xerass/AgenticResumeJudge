# agenticResume

An AI powered multi agent system for automated resume screening and deep candidate evaluation. It builds a persistent knowledge graph of a candidate's experience and employs a panel of specialized agents to provide a balanced hiring verdict.

## Core Architecture

The system uses an orchestrator to manage a four stage pipeline. Each stage is handled by an LLM powered agent with a specific persona and set of railguards.

1. **Constructor Agent**: Parses the resume and generates Cypher queries to build a Neo4j knowledge graph. This structures messy text into a queryable skill network.
2. **Auditor Agent**: Cross references the candidate knowledge graph against the job description to identify direct matches, categorical matches, and missing skills.
3. **Judge Agents (Parallel)**:
   * **Skeptic**: Acts as a technical gatekeeper focused on risks, ramp up time, and missing critical dependencies.
   * **Enthusiast**: Acts as a defense attorney finding conceptual equivalents for missing skills and highlighting growth potential.
4. **Lead Recruiter**: Synthesizes the conflicting judge reports to provide a final score and a verdict of INVITE or REJECT.

## Tech Stack

* **LLM**: Gemini 2.5 Flash
* **Graph Database**: Neo4j (Cypher)
* **Runtime**: Python 3.12+ (Asyncio)
* **Key Libraries**: google-genai, neo4j, pypdf, python-dotenv

## Installation

1. Clone the repository and navigate to the project directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables in a `.env` file:
   ```env
   GEMINI_API_KEY=your_api_key_here
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   ```

## Usage

### Running the Pipeline
Place a resume PDF and a job description text file in the `data/` directory, then run:
```bash
python main.py
```

### Verification
To test the end to end flow with internal sample data, execute the verification script:
```bash
python verify.py
```

## Project Structure

* `agents/`: Specialized logic for each agent (Constructor, Auditor, Skeptic, Enthusiast).
* `database/`: Neo4j driver and connection management.
* `data/`: Sample resumes and job descriptions for testing.
* `orchestrator.py`: Manages agent execution, state, and error handling.
* `state.py`: Defines the shared memory schema for the pipeline.
* `main.py`: Main entry point for processing external files.
