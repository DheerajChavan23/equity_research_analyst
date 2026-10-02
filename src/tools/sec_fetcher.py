import os
import glob
from pathlib import Path
from sec_edgar_downloader import Downloader

class SECFetcher:
    def __init__(self, download_dir: str = "data/raw_filings"):
        self.download_dir = download_dir
        # SEC EDGAR requires: "AppName AdminContact@domain.com"
        user_agent = os.getenv("SEC_USER_AGENT", "ResearchBot admin@example.com")
        
        # Split into company name and email
        parts = user_agent.split()
        company_name = parts[0] if len(parts) > 1 else "EquityResearchAgent"
        email = parts[-1] if "@" in parts[-1] else "contact@example.com"
        
        self.dl = Downloader(company_name, email, self.download_dir)

    def fetch_latest_10k(self, ticker: str, limit: int = 1) -> str:
        """
        Downloads the latest 10-K filing for the given ticker.
        Returns the file path of the downloaded document.
        """
        ticker = ticker.upper()
        self.dl.get("10-K", ticker, limit=limit)
        
        # Locate the downloaded file
        search_pattern = os.path.join(
            self.download_dir, "sec-edgar-filings", ticker, "10-K", "*", "*.txt"
        )
        files = glob.glob(search_pattern)
        if not files:
            # Fall back to checking for .htm / .html
            search_pattern_html = os.path.join(
                self.download_dir, "sec-edgar-filings", ticker, "10-K", "*", "*.htm*"
            )
            files = glob.glob(search_pattern_html)
            
        if not files:
            raise FileNotFoundError(f"No 10-K filing found for {ticker} in {self.download_dir}")
            
        # Return the most recent filing path
        return sorted(files, key=os.path.getmtime, reverse=True)[0]