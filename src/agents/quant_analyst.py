from src.llm.gateway import LLMGateway
from src.llm.schema import Message, LLMRequest
from src.tools.market_data import MarketDataTool
from src.agents.state import AgentScratchpad

class QuantAnalystAgent:
    def __init__(self, gateway: LLMGateway):
        self.gateway = gateway

    def analyze(self, state: AgentScratchpad) -> AgentScratchpad:
        metrics = MarketDataTool.get_summary_metrics(state["ticker"])
        state["quant_metrics"] = metrics

        prompt = f"""Analyze the financial metrics for {state['ticker']}:
- Market Cap: {metrics.get('market_cap')}
- Trailing P/E: {metrics.get('pe_ratio_trailing')} | Forward P/E: {metrics.get('pe_ratio_forward')}
- YoY Revenue Growth: {metrics.get('revenue_growth_yoy')}
- Gross Margin: {metrics.get('gross_margins')} | Operating Margin: {metrics.get('operating_margins')}
- Total Debt: {metrics.get('total_debt')} | Free Cash Flow: {metrics.get('free_cashflow')}

Summarize financial health, valuation multiples, and margin trends in 3 concise bullet points."""

        response = self.gateway.generate(LLMRequest(
            messages=[Message(role="user", content=prompt)],
            temperature=0.1
        ))
        state["quant_summary"] = response.content.strip()
        return state