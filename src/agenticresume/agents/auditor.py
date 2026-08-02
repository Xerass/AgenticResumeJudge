"""Auditor Agent: Asesses a candidate's CareerProfile against a JobPosts' 
requirements and produces one coverage item per requirement. Facts and requirements 
are ID'd"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from agenticresume.agents.mapping import coverages_from_audit
from agenticresume.agents.schemas import AuditorOutput
from agenticresume.domain.models import CareerProfile, Coverage, JobPost
from agenticresume.infra.llm import build_extractor
from agenticresume.settings import Settings