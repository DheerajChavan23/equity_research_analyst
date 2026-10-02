from typing import TypedDict, List, Dict, Any, Optional

class AgentScratchpad(TypedDict):
    ticker: str
    target_year: int
    user_goal: str
    # Planner outputs
    research_plan: List[str]
    # Quantitative findings (ratios, growth rates, margins)
    quant_metrics: Dict[str, Any]
    quant_summary: str
    # Qualitative findings (extracted 10-K sections & citations)
    qual_citations: List[Dict[str, str]]
    qual_summary: str
    # Synthesis & Audit
    draft_memo: Optional[str]
    critic_verdict: Optional[str]  # "APPROVED" | "REVISE"
    critique_notes: Optional[str]
    iteration_count: int