import json
from src.llm.gateway import LLMGateway
from src.llm.schema import Message, LLMRequest
from src.memory.sqlite_store import LocalMemoryStore
from src.agents.state import AgentScratchpad

class MemoryGate:
    def __init__(self, gateway: LLMGateway, memory: LocalMemoryStore):
        self.gateway = gateway
        self.memory = memory

    def evaluate_and_store(self, state: AgentScratchpad):
        """Evaluates the session to find long-term user preferences."""
        prompt = f"""Review the user's research goal: "{state['user_goal']}"
                    Does this contain a reusable preference, formatting rule, or analytical constraint? (e.g., "Always use a DCF model", "Focus strictly on SaaS").
                    If YES, extract the core rule. If NO, return empty.

                    Output JSON:
                    {{
                        "has_preference": true/false,
                        "preference_key": "short_slug",
                        "preference_value": "The actual rule to remember"
                    }}"""

        response = self.gateway.generate(LLMRequest(
            messages=[
                Message(role="system", content="Return valid JSON only."),
                Message(role="user", content=prompt)
            ],
            response_format="json"
        ))
        
        try:
            result = json.loads(response.content)
            if result.get("has_preference"):
                # Save to SQLite
                with self.memory._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO memories (category, key, value)
                        VALUES (?, ?, ?)
                    """, ("user_preference", result["preference_key"], result["preference_value"]))
                    conn.commit()
                print(f"[Memory Gate] Saved new preference: {result['preference_value']}")
        except Exception:
            pass # Failsafe: if the LLM hallucinates non-JSON, we just don't save a memory.