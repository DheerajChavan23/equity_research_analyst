import sys
from dotenv import load_dotenv

load_dotenv()

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.llm.gateway import LLMGateway
from src.rag.retriever import FinancialRetriever
from src.memory.sqlite_store import LocalMemoryStore
from src.core.loop import EquityResearchLoop
from src.tools.resolver import TickerResolver

app = typer.Typer(help="Autonomous Equity Research Agent")
console = Console()

@app.command()
def research(
    ticker: str = typer.Argument(..., help="Stock ticker symbol or company name (e.g., AAPL, Apple, Taiwan Semiconductor)"),
    goal: str = typer.Argument(..., help="What should the agent focus on?"),
    iterations: int = typer.Option(2, help="Max revision loops for the Critic agent")
):
    """Run an autonomous equity research loop on a specific company name or ticker."""
    # Initialize dependencies
    gateway = LLMGateway()
    memory = LocalMemoryStore()
    retriever = FinancialRetriever()
    
    # Resolve company name to ticker
    resolver = TickerResolver(gateway=gateway)
    resolved_ticker = resolver.resolve(ticker)

    if resolved_ticker != ticker.upper():
        console.print(Panel.fit(
            f"[bold white]Equity Research Agent[/bold white]\n"
            f"Query: [cyan]{ticker}[/cyan] ➔ Target: [bold green]${resolved_ticker}[/bold green]",
            border_style="cyan"
        ))
    else:
        console.print(Panel.fit(
            f"[bold white]Equity Research Agent[/bold white]\n"
            f"Target: [cyan]{resolved_ticker}[/cyan]",
            border_style="cyan"
        ))
    
    # Initialize Core Loop
    loop = EquityResearchLoop(gateway, retriever, memory)

    # Execute
    try:
        final_state = loop.run(resolved_ticker, goal, max_iterations=iterations)
        
        console.print("\n[bold green]✅ Research Complete. Final Memorandum:[/bold green]\n")
        console.print(Markdown(final_state["draft_memo"]))
        
    except Exception as e:
        console.print(f"[bold red]Fatal Error during execution:[/bold red] {str(e)}")

if __name__ == "__main__":
    app()