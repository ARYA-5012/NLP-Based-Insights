"""
Financial Modeling Prep API Client for fetching earnings call transcripts.
Handles API communication, rate limiting, and local caching.
"""
import os
import json
import time
import requests
from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()


class FMPClient:
    """Client for Financial Modeling Prep API to fetch earnings call transcripts."""

    BASE_URL = "https://financialmodelingprep.com/api/v3"
    
    # S&P 500 target companies by sector
    TARGET_COMPANIES = {
        "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "CRM", "ORCL", "ADBE", "INTC",
                       "AMD", "QCOM", "AVGO", "TXN", "CSCO", "IBM", "NOW", "SHOP", "SNOW", "UBER",
                       "SQ", "PLTR", "NFLX", "PYPL", "ABNB", "SPOT", "TWLO", "ZS", "NET", "DDOG"],
        "Financials": ["JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW", "AXP", "V",
                       "MA", "COF", "USB", "PNC", "TFC", "BK", "STT", "CME", "ICE", "SPGI"],
        "Healthcare": ["UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "DHR", "BMY",
                       "AMGN", "GILD", "ISRG", "MDT", "SYK", "CVS", "CI", "HCA", "ELV", "ZTS"],
        "Consumer": ["WMT", "PG", "KO", "PEP", "COST", "NKE", "MCD", "SBUX", "TGT", "LOW",
                     "HD", "TJX", "ROST", "DG", "DLTR", "YUM", "CMG", "DPZ", "EL", "CL"],
        "Industrials": ["BA", "CAT", "GE", "HON", "UPS", "RTX", "LMT", "DE", "MMM", "EMR"],
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FMP_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FMP_API_KEY not found. Set it in .env or pass as argument.\n"
                "Get a free key at: https://financialmodelingprep.com/developer/docs/"
            )
        self._request_count = 0
        self._last_request_time = 0

    def _rate_limit(self):
        """Respect API rate limits (250 calls/day on free tier)."""
        elapsed = time.time() - self._last_request_time
        if elapsed < 0.5:  # Min 500ms between requests
            time.sleep(0.5 - elapsed)
        self._request_count += 1
        self._last_request_time = time.time()
        
        if self._request_count % 50 == 0:
            logger.info(f"API requests made: {self._request_count}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_transcript(self, ticker: str, quarter: int, year: int) -> Optional[Dict]:
        """Fetch a specific earnings call transcript."""
        self._rate_limit()
        url = f"{self.BASE_URL}/earning_call_transcript/{ticker}"
        params = {"quarter": quarter, "year": year, "apikey": self.api_key}

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data:
                logger.debug(f"No transcript: {ticker} Q{quarter} {year}")
                return None

            transcript = data[0] if isinstance(data, list) else data
            logger.info(f"Fetched: {ticker} Q{quarter} {year}")
            return transcript

        except requests.exceptions.RequestException as e:
            logger.error(f"API error for {ticker} Q{quarter} {year}: {e}")
            raise

    def save_transcript(self, transcript: Dict, save_dir: str) -> str:
        """Save transcript JSON to disk. Returns filepath."""
        if not transcript:
            return ""
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        ticker = transcript.get("symbol", "UNKNOWN")
        quarter = transcript.get("quarter", 0)
        year = transcript.get("year", 0)
        filename = f"{ticker}_Q{quarter}_{year}.json"
        filepath = os.path.join(save_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(transcript, f, indent=2)
        
        return filepath

    def collect_all(self, save_dir: str, years: range = range(2022, 2026),
                    quarters: range = range(1, 5)) -> int:
        """Collect transcripts for all target companies across years/quarters."""
        total_saved = 0
        all_tickers = [t for sector in self.TARGET_COMPANIES.values() for t in sector]
        total_requests = len(all_tickers) * len(years) * len(quarters)
        
        logger.info(f"Starting collection: {len(all_tickers)} companies, "
                     f"{len(years)} years, {len(quarters)} quarters = {total_requests} max requests")

        for ticker in all_tickers:
            for year in years:
                for quarter in quarters:
                    try:
                        transcript = self.get_transcript(ticker, quarter, year)
                        if transcript:
                            self.save_transcript(transcript, save_dir)
                            total_saved += 1
                    except Exception as e:
                        logger.warning(f"Skipping {ticker} Q{quarter} {year}: {e}")
                        continue

        logger.info(f"Collection complete: {total_saved} transcripts saved")
        return total_saved

    def get_sector_for_ticker(self, ticker: str) -> str:
        """Return the sector for a given ticker."""
        for sector, tickers in self.TARGET_COMPANIES.items():
            if ticker in tickers:
                return sector
        return "Unknown"


if __name__ == "__main__":
    client = FMPClient()
    # Test with one transcript
    transcript = client.get_transcript("AAPL", 3, 2024)
    if transcript:
        path = client.save_transcript(transcript, "data/raw")
        logger.info(f"Test transcript saved to: {path}")
