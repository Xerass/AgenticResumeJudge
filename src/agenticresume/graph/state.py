#for learning: a few key things, A LangGraph system requires a state.py file, which defines the state of the graph. 
# The state is a data structure that holds the current state of the graph, including the nodes and edges. 
#The state is used by the agents to make decisions and to update the graph.


#basic implementations are just a TypeDict

import operator
from typing import Annotated, TypedDict

from agenticresume.domain.models import (
    AnalysisResult,
    Assessment,
    CareerProfile,
    Coverage,
    JobPost,
)

class ScreeningState(TypedDict, total=False):
    #inputs
    resume_text: str
    jd_text: str

    #produced
    profile: CareerProfile                                    # extract node
    job_post: JobPost                                         # parse node
    coverages: list[Coverage]                                # audit node
    assessments: Annotated[list[Assessment], operator.add]   # 3 judges, pass to the add (concatenator) accumulator, side note: append wont work since the LangGraph Reducer sees it as null
    result: AnalysisResult                                   # recruiter node