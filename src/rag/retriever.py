from typing import List, Dict, Any, Optional
from .store import FinancialVectorStore

class FinancialRetriever:
    def __init__(self, store: Optional[FinancialVectorStore] = None):
        self.store = store or FinancialVectorStore()

    def query(
        self,
        ticker: str,
        query_text: str,
        section: Optional[str] = None,
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic search over 10-K filings.
        Can optionally filter by section ('item_1', 'item_1a', 'item_7').
        """
        where_filter: Dict[str, Any] = {"ticker": ticker.upper()}
        if section:
            where_filter = {
                "$and": [
                    {"ticker": ticker.upper()},
                    {"section": section}
                ]
            }

        results = self.store.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where_filter
        )

        formatted_chunks = []
        if results and results.get("documents"):
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            for doc, meta in zip(docs, metas):
                formatted_chunks.append({
                    "content": doc,
                    "section": meta.get("section", "unknown"),
                    "chunk_index": meta.get("chunk_index", "0")
                })

        return formatted_chunks