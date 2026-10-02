import os
import re
import json
import logging
from typing import Optional, Dict, Any
import yfinance as yf

from src.llm.gateway import LLMGateway
from src.llm.schema import Message, LLMRequest

logger = logging.getLogger(__name__)

class TickerResolver:
    """
    Resolves human-readable company names or informal queries into valid stock ticker symbols.
    
    Resolution hierarchy:
    1. Direct Ticker Recognition: Verifies if input is an already valid uppercase ticker (zero-latency).
    2. SEC EDGAR Official Ticker Map: Instant offline lookup across ~10,000 public companies.
    3. Real-Time Yahoo Finance Search: yfinance.Search queries Yahoo Finance auto-complete API.
    4. LLM Gateway Fallback: Uses LLM to resolve informal colloquial brand names.
    """

    def __init__(self, gateway: Optional[LLMGateway] = None, sec_cache_path: str = "data/sec_tickers.json"):
        self.gateway = gateway
        self.sec_cache_path = sec_cache_path
        self.sec_title_map: Dict[str, str] = {}
        self.sec_ticker_set: set = set()
        self._load_sec_cache()

    def _load_sec_cache(self) -> None:
        """Loads cached SEC EDGAR ticker-to-title map if available."""
        if os.path.exists(self.sec_cache_path):
            try:
                with open(self.sec_cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data.values():
                        title = item.get("title", "").strip().upper()
                        ticker = item.get("ticker", "").strip().upper()
                        if title and ticker:
                            self.sec_title_map[title] = ticker
                            self.sec_ticker_set.add(ticker)
            except Exception as e:
                logger.warning(f"Could not load SEC ticker cache from {self.sec_cache_path}: {e}")

    def resolve(self, query: str) -> str:
        """
        Resolves a company name or ticker query string to an uppercase ticker symbol.
        
        Args:
            query: User input (e.g. 'AAPL', 'Apple', 'Taiwan Semiconductor', '$MSFT')
            
        Returns:
            Resolved uppercase stock ticker symbol (e.g. 'AAPL', 'TSM', 'MSFT')
        """
        clean_query = query.strip().lstrip("$").strip()
        if not clean_query:
            return ""

        # 1. Direct Ticker check: verified against official SEC set or uppercase 1-5 letters
        if clean_query.isupper() and clean_query in self.sec_ticker_set:
            return clean_query

        if not self.sec_ticker_set and re.fullmatch(r"^[A-Za-z]{1,5}$", clean_query) and clean_query.isupper():
            return clean_query

        # 2. Check offline SEC EDGAR company titles
        norm_query = clean_query.upper()
        if norm_query in self.sec_title_map:
            return self.sec_title_map[norm_query]

        # 3. Real-Time Yahoo Finance Search API
        try:
            search = yf.Search(clean_query, max_results=5)
            quotes = getattr(search, "quotes", [])
            
            # Prioritize US equities in SEC registry
            for q in quotes:
                symbol = q.get("symbol", "").upper()
                quote_type = q.get("quoteType", "")
                if quote_type in ("EQUITY", "ETF") and "." not in symbol:
                    if not self.sec_ticker_set or symbol in self.sec_ticker_set:
                        return symbol
            
            # Standard equity filter
            for q in quotes:
                symbol = q.get("symbol", "").upper()
                quote_type = q.get("quoteType", "")
                if quote_type in ("EQUITY", "ETF") and "." not in symbol:
                    return symbol
                    
            if quotes:
                first_sym = quotes[0].get("symbol", "")
                if first_sym:
                    return first_sym.split(".")[0].upper()
        except Exception as e:
            logger.debug(f"Yahoo Finance search failed for '{clean_query}': {e}")

        # 4. Fallback: Ask LLM Gateway to resolve informal names
        if self.gateway:
            try:
                prompt = (
                    f'Identify the primary US stock exchange ticker (NYSE/NASDAQ) for the company: "{clean_query}".\n'
                    f'Return ONLY the raw uppercase ticker symbol (e.g. AAPL, MSFT, GOOGL, TSM). Do not include any explanations or punctuation.'
                )
                resp = self.gateway.generate(LLMRequest(
                    messages=[Message(role="user", content=prompt)],
                    temperature=0.0
                ))
                candidate = resp.content.strip().replace("$", "").upper()
                match = re.search(r"\b[A-Za-z]{1,5}\b", candidate)
                if match:
                    return match.group(0).upper()
            except Exception as e:
                logger.debug(f"LLM fallback failed for '{clean_query}': {e}")

        # 5. Fallback to raw uppercase query
        return clean_query.upper()
