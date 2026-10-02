from typing import Dict, Any
import yfinance as yf
import pandas as pd

class MarketDataTool:
    @staticmethod
    def get_summary_metrics(ticker: str) -> Dict[str, Any]:
        """Fetches fundamental multiples, market cap, and profitability metrics."""
        t = yf.Ticker(ticker.upper())
        info = t.info
        
        return {
            "symbol": ticker.upper(),
            "short_name": info.get("shortName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap"),
            "pe_ratio_trailing": info.get("trailingPE"),
            "pe_ratio_forward": info.get("forwardPE"),
            "revenue_growth_yoy": info.get("revenueGrowth"),
            "gross_margins": info.get("grossMargins"),
            "operating_margins": info.get("operatingMargins"),
            "ebitda": info.get("ebitda"),
            "total_debt": info.get("totalDebt"),
            "free_cashflow": info.get("freeCashflow"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
        }

    @staticmethod
    def get_price_history(ticker: str, period: str = "5y") -> pd.DataFrame:
        """Pulls OHLCV time-series data for quantitative trend analysis."""
        t = yf.Ticker(ticker.upper())
        history = t.history(period=period)
        return history[["Open", "High", "Low", "Close", "Volume"]]