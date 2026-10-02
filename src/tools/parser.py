import re
from pathlib import Path
from typing import Dict, Optional
from bs4 import BeautifulSoup

class SECParser:
    # Target core qualitative sections in standard 10-K filings
    SECTION_PATTERNS = {
        "item_1": r"(?:item\s+1\.\s+business)(.*?)(?=item\s+1a\.\s+risk\s+factors)",
        "item_1a": r"(?:item\s+1a\.\s+risk\s+factors)(.*?)(?=item\s+1b\.\s+unresolved\s+staff\s+comments|item\s+2\.\s+properties)",
        "item_7": r"(?:item\s+7\.\s+management['’]?s\s+discussion\s+and\s+analysis)(.*?)(?=item\s+7a\.\s+quantitative\s+and\s+qualitative)",
    }

    @staticmethod
    def clean_html(raw_content: str) -> str:
        """Strips HTML tags, XML headers, scripts, and normalizes whitespace."""
        soup = BeautifulSoup(raw_content, "lxml")
        for tag in soup(["script", "style", "table"]):  # Remove tables and scripts for text RAG
            tag.decompose()
        text = soup.get_text(separator=" ")
        # Normalize multiple spaces, non-breaking spaces, and blank lines
        text = re.sub(r"[\xa0\s]+", " ", text)
        return text.strip()

    @classmethod
    def parse_filing(cls, file_path: str) -> Dict[str, str]:
        """
        Reads a raw 10-K (.txt or .html), cleans markup, 
        and extracts key items into a dictionary.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        clean_text = cls.clean_html(raw_text)

        extracted_sections: Dict[str, str] = {}
        lower_text = clean_text.lower()

        for section_key, pattern in cls.SECTION_PATTERNS.items():
            match = re.search(pattern, lower_text, re.DOTALL | re.IGNORECASE)
            if match:
                start, end = match.span(1)
                extracted_sections[section_key] = clean_text[start:end].strip()
            else:
                extracted_sections[section_key] = ""

        # Fallback: if regex missed individual sections, store full cleaned body
        if not any(extracted_sections.values()):
            extracted_sections["full_document"] = clean_text[:200000]

        return extracted_sections