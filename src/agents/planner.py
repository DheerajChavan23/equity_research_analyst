import json
from typing import Optional
from src.llm.gateway import LLMGateway
from src.llm.schema import Message, LLMRequest
from src.agents.state import AgentScratchpad
from src.memory.sqlite_store import LocalMemoryStore

class PlannerAgent:
    def __init__(self, gateway: LLMGateway, memory: Optional[LocalMemoryStore] = None):
        self.gateway = gateway
        self.memory = memory

    def plan(self, state: AgentScratchpad) -> AgentScratchpad:
        # 1. Retrieve persistent user preferences from SQLite
        saved_rules = []
        if self.memory:
            memories = self.memory.get_all_memories(category="user_preference")
            saved_rules = [f"- {m['value']}" for m in memories]
            
        memory_context = ""
        if saved_rules:
            memory_context = "\nPersistent User Preferences & Rules to Enforce:\n" + "\n".join(saved_rules)

        # 2. Inject memories into the planning prompt
        prompt = f"""You are a Lead Financial Analyst. Break down the equity research request for {state['ticker']}.
Goal: {state['user_goal']}
{memory_context}

Output a JSON array of 3-4 specific research targets. If any persistent user preferences apply, incorporate them into the plan.
Format: ["target 1", "target 2", "target 3"]"""

        response = self.gateway.generate(LLMRequest(
            messages=[
                Message(role="system", content="Return valid JSON only."),
                Message(role="user", content=prompt)
            ],
            response_format="json",
            temperature=0.0
        ))
        
        try:
            plan_items = json.loads(response.content)
            state["research_plan"] = plan_items if isinstance(plan_items, list) else list(plan_items.values())[0]
        except Exception:
            state["research_plan"] = [
                "Analyze historical revenue growth and margins",
                "Review Item 1A risk factors for operational hazards",
                "Assess management commentary in MD&A"
            ]
        return state