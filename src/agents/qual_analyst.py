import json
from typing import List, Dict, Any

from src.llm.gateway import LLMGateway
from src.llm.schema import Message, LLMRequest, ToolCall
from src.rag.retriever import FinancialRetriever
from src.agents.state import AgentScratchpad

# Import ingestion tools to allow the agent to self-heal missing data
from src.tools.sec_fetcher import SECFetcher
from src.tools.parser import SECParser
from src.rag.store import FinancialVectorStore

class QualAnalystAgent:
    def __init__(self, gateway: LLMGateway, retriever: FinancialRetriever):
        self.gateway = gateway
        self.retriever = retriever

    def _get_tools(self) -> List[Dict[str, Any]]:
        """Define the JSON Schema for the tools the LLM can autonomously invoke."""
        return [
            {
                "name": "search_10k",
                "description": "Search the local vector database for 10-K filing excerpts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query (e.g., 'supply chain risks')"},
                        "section": {"type": "string", "description": "Optional section filter, e.g., 'item_1a' or 'item_7'"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "download_10k",
                "description": "Download the latest 10-K from the SEC and ingest it into the vector database. ONLY use this if search_10k returns no results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"}
                    },
                    "required": ["ticker"]
                }
            }
        ]

    def _execute_tool(self, tool_call: ToolCall, state: AgentScratchpad) -> str:
        """Executes the specific Python logic for the requested tool."""
        print(f"   [Agent Thought] Calling tool: {tool_call.name} with {tool_call.arguments}")
        
        if tool_call.name == "search_10k":
            query = tool_call.arguments.get("query")
            section = tool_call.arguments.get("section")
            chunks = self.retriever.query(state["ticker"], query, section, top_k=3)
            
            if not chunks:
                return json.dumps({"status": "error", "message": "No data found. The filing may not be downloaded yet."})
            
            # Save citations to state for the Critic to use later
            for c in chunks:
                state["qual_citations"].append({"section": c["section"], "text": c["content"]})
                
            return json.dumps({"status": "success", "data": chunks})
            
        elif tool_call.name == "download_10k":
            ticker = tool_call.arguments.get("ticker", state["ticker"])
            try:
                fetcher = SECFetcher()
                filing_path = fetcher.fetch_latest_10k(ticker)
                sections = SECParser.parse_filing(filing_path)
                store = FinancialVectorStore()
                store.ingest_sections(ticker, sections)
                return json.dumps({"status": "success", "message": f"{ticker} 10-K downloaded and vectorized successfully. You can now search it."})
            except Exception as e:
                return json.dumps({"status": "error", "message": str(e)})
                
        return json.dumps({"status": "error", "message": "Unknown tool."})

    def analyze(self, state: AgentScratchpad) -> AgentScratchpad:
        """The ReAct (Reason-Act) Loop."""
        state["qual_citations"] = []
        
        # 1. Initialize the conversation history with instructions
        messages = [
            Message(role="system", content=f"""You are an Autonomous Qualitative Financial Analyst researching {state['ticker']}.
            You must research the company using your tools.
            If your search returns empty, download the filing, then search again.
            Once you have extracted enough risk and strategic context:
            1. Write a concise 3-bullet summary strictly grounded on the retrieved 10-K text excerpts.
            2. Only state facts directly affirmed in the retrieved excerpts. Do NOT extrapolate or introduce external company/partner names (e.g. do not mention TSMC or outside entities unless explicitly named in the excerpt).
            3. Stop calling tools once you output the final 3-bullet summary."""),
            Message(role="user", content=f"Research {state['ticker']}. Focus on: {state['research_plan']}")
        ]

        max_steps = 5
        for step in range(max_steps):
            # 2. Ask the LLM what to do next
            response = self.gateway.generate(LLMRequest(
                messages=messages,
                tools=self._get_tools(),
                temperature=0.1
            ))

            # 3. If no tool calls, the LLM has generated its final summary
            if not response.tool_calls:
                state["qual_summary"] = response.content.strip()
                break

            # 4. If tool calls exist, append the assistant's decision to history
            messages.append(Message(
                role="assistant", 
                content=response.content or "", 
                tool_calls=response.tool_calls
            ))

            # 5. Execute each tool and return the Observation back to the LLM
            for tool_call in response.tool_calls:
                observation = self._execute_tool(tool_call, state)
                messages.append(Message(
                    role="tool", 
                    content=observation, 
                    tool_call_id=tool_call.id
                ))

        return state