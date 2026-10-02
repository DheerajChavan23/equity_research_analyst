import pytest
from src.tools.resolver import TickerResolver

def test_ticker_resolver_direct_tickers():
    resolver = TickerResolver()
    assert resolver.resolve("AAPL") == "AAPL"
    assert resolver.resolve("NVDA") == "NVDA"
    assert resolver.resolve("MSFT") == "MSFT"
    assert resolver.resolve("$TSLA") == "TSLA"

def test_ticker_resolver_company_names():
    resolver = TickerResolver()
    assert resolver.resolve("Apple") == "AAPL"
    assert resolver.resolve("Microsoft") == "MSFT"
    assert resolver.resolve("Taiwan Semiconductor") == "TSM"
    assert resolver.resolve("Tesla") == "TSLA"
    assert resolver.resolve("Amazon") == "AMZN"
