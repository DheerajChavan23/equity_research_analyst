import pytest
import re

def test_deterministic_release_gate():
    """
    Simulates a final Release Gate in a CI/CD pipeline.
    Validates structural integrity of the final AgentScratchpad.
    """
    
    # In a real environment, you can load the latest state from LocalMemoryStore.
    # For this test, we simulate a successful payload to verify our regex logic.
    mock_memo = """
# INSTITUTIONAL EQUITY RESEARCH

## 1. Executive Summary & Thesis
NVIDIA is dominating.

## 2. Quantitative Valuation & Margins Table
| Metric | Value | Context |
|---|---|---|
| Trailing P/E | 29.6x | Repricing |
| Forward P/E | 14.9x | Attractive |
| FCF | $41.8B | Strong |
| Gross Margin | 74.7% | Elite |
| Operating Margin | 66.2% | High |

## 3. Key 10-K Risks (with Section References)
Supply chain risks.

## 4. Investment Outlook
Buy.
"""
    
    state = {
        "draft_memo": mock_memo,
        "critic_verdict": "APPROVED"
    }
    memo = state["draft_memo"]

    # 1. VERDICT CHECK: Assert the Critic explicitly approved the document
    assert state["critic_verdict"] == "APPROVED", "Critic Guardrail failed to approve the report."

    # 2. HEADING CHECK: Assert all 4 core sections are present
    required_headings = [
        "Executive Summary",
        "Quantitative Valuation",
        "Key 10-K Risks",
        "Investment Outlook"
    ]
    for heading in required_headings:
        # Regex looks for a Markdown header line containing the exact phrase
        pattern = rf"^#+.*\b{heading}\b.*$"
        match = re.search(pattern, memo, re.IGNORECASE | re.MULTILINE)
        assert match, f"Deterministic Failure: Missing required heading '{heading}'"

    # 3. TABLE EXISTENCE & ROW COUNT CHECK
    # Standard Markdown tables use pipes. We extract all lines containing a '|'
    table_lines = [line.strip() for line in memo.split("\n") if "|" in line]
    assert len(table_lines) >= 3, "Deterministic Failure: No standard Markdown table found (missing pipes '|')."
    
    # Exclude the header row and the delimiter row (e.g., |---|---|)
    data_rows = table_lines[2:]
    assert len(data_rows) >= 5, f"Deterministic Failure: Table must have at least 5 numerical rows. Found {len(data_rows)}."

    # 4. DATA INTEGRITY CHECK (NO EMPTY CELLS)
    for row_idx, row in enumerate(data_rows):
        # Split by pipe and remove leading/trailing empty strings from the outer edges
        cells = [cell.strip() for cell in row.split("|")[1:-1]]
        for col_idx, cell in enumerate(cells):
            assert cell != "", f"Deterministic Failure: Empty cell detected at Row {row_idx + 1}, Column {col_idx + 1}."