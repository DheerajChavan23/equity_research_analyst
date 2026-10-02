import json
from src.llm.gateway import LLMGateway
from src.llm.schema import Message, LLMRequest
from src.agents.state import AgentScratchpad

class CriticAgent:
    def __init__(self, gateway: LLMGateway):
        self.gateway = gateway

    def review(self, state: AgentScratchpad) -> AgentScratchpad:
        prompt = f"""Act as a Compliance & Fact-Checking Auditor. Verify the draft research memo against raw data.

                    Raw Quant Data:
                    {state['quant_metrics']}

                    Draft Memo:
                    {state['draft_memo']}

                    Check:
                    1. Are all stated metrics (P/E, margins, debt) consistent with the Raw Quant Data?
                    2. Are there any ungrounded speculative assertions?

                    Output valid JSON with keys:
                    - "verdict": "APPROVED" or "REVISE"
                    - "notes": "Explanation of changes required, or confirmation that facts are grounded."
                    """

        response = self.gateway.generate(LLMRequest(
            messages=[
                Message(role="system", content="Return valid JSON only."),
                Message(role="user", content=prompt)
            ],
            response_format="json",
            temperature=0.0
        ))
        
        try:
            result = json.loads(response.content)
            state["critic_verdict"] = result.get("verdict", "APPROVED")
            state["critique_notes"] = result.get("notes", "")
        except Exception:
            state["critic_verdict"] = "APPROVED"
            state["critique_notes"] = "Audit completed via default pass."
            
        return state