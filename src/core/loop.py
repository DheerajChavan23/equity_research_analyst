from typing import Optional
from rich.console import Console

from src.llm.gateway import LLMGateway
from src.llm.schema import Message, LLMRequest
from src.rag.retriever import FinancialRetriever
from src.agents.state import AgentScratchpad
from src.agents.planner import PlannerAgent
from src.agents.quant_analyst import QuantAnalystAgent
from src.agents.qual_analyst import QualAnalystAgent
from src.agents.critic import CriticAgent
from src.memory.sqlite_store import LocalMemoryStore
from src.memory.memory_gate import MemoryGate
from src.tools.resolver import TickerResolver

console = Console()

class EquityResearchLoop:
    def __init__(self, gateway: LLMGateway, retriever: FinancialRetriever, memory: LocalMemoryStore):
        self.gateway = gateway
        self.memory = memory
        self.retriever = retriever
        self.resolver = TickerResolver(gateway=self.gateway)
        
        # Injects local SQLite storage into Planner to load saved preferences
        self.planner = PlannerAgent(gateway, memory=self.memory)
        self.quant = QuantAnalystAgent(gateway)
        self.qual = QualAnalystAgent(gateway, retriever)
        self.critic = CriticAgent(gateway)
        self.memory_gate = MemoryGate(gateway, self.memory)

    def _synthesize_memo(self, state: AgentScratchpad) -> str:
        prompt = f"""You are a Senior Wall Street Equity Research Director.
Compile a formal Investment Memorandum for {state['ticker']}.

Core Objective: {state['user_goal']}

Quantitative Summary:
{state['quant_summary']}

Qualitative 10-K Findings:
{state['qual_summary']}

Critique feedback from prior round (if any):
{state.get('critique_notes', 'None')}

Output a professional Markdown memorandum with:
1. Executive Summary & Thesis
2. Quantitative Valuation & Margins Table
3. Key 10-K Risks (with section references)
4. Investment Outlook
"""
        response = self.gateway.generate(LLMRequest(
            messages=[Message(role="user", content=prompt)],
            temperature=0.2
        ))
        return response.content.strip()

    def run(self, ticker: str, goal: str, max_iterations: int = 2) -> AgentScratchpad:
        resolved_ticker = self.resolver.resolve(ticker)
        state: AgentScratchpad = {
            "ticker": resolved_ticker,
            "target_year": 2026,
            "user_goal": goal,
            "research_plan": [],
            "quant_metrics": {},
            "quant_summary": "",
            "qual_citations": [],
            "qual_summary": "",
            "draft_memo": None,
            "critic_verdict": None,
            "critique_notes": None,
            "iteration_count": 0
        }

        # Step 1: Planning (loads saved episodic memory from SQLite)
        console.print(f"[bold cyan]▶ Planning research run for {resolved_ticker}...[/bold cyan]")
        state = self.planner.plan(state)

        # Step 2: Quantitative Analysis (Market Data / yfinance)
        console.print("[bold yellow]▶ Fetching quantitative multiples & pricing...[/bold yellow]")
        state = self.quant.analyze(state)

        # Step 3: Qualitative Analysis (ChromaDB RAG / SEC Fetcher)
        console.print("[bold yellow]▶ Running RAG retrieval on 10-K filing...[/bold yellow]")
        state = self.qual.analyze(state)

        # Step 4: Iterative Draft & Critic Review
        while state["iteration_count"] < max_iterations:
            state["iteration_count"] += 1
            console.print(f"[bold green]▶ Drafting memo (Round {state['iteration_count']})...[/bold green]")
            state["draft_memo"] = self._synthesize_memo(state)

            console.print("[bold magenta]▶ Running Critic guardrails & fact audit...[/bold magenta]")
            state = self.critic.review(state)

            if state["critic_verdict"] == "APPROVED":
                console.print("[bold green]✔ Critic approved report.[/bold green]")
                break
            else:
                console.print(f"[bold red]✘ Critic requested revision:[/bold red] {state['critique_notes']}")

        # Step 5: Memory Gate Evaluation (extracts & persists long-term user preferences)
        console.print("[bold blue]🧠 Evaluating session for reusable preferences...[/bold blue]")
        self.memory_gate.evaluate_and_store(state)

        # Step 6: Persist Full Session Scratchpad to SQLite
        session_id = f"{resolved_ticker}_{state['target_year']}"
        self.memory.save_scratchpad(session_id, resolved_ticker, state)
        
        return state