import os
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

class FinancialVectorStore:
    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "data/vector_store")
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Local persistent client
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        
        # Default to lightweight local embedding function (no OpenAI token cost for embeddings)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        self.collection = self.client.get_or_create_collection(
            name="sec_10k_reports",
            embedding_function=self.embedding_fn
        )

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
        """Simple, deterministic character chunking with sliding overlap."""
        if not text:
            return []
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunks.append(text[start:end])
            if end == text_len:
                break
            start += (chunk_size - overlap)
        return chunks

    def ingest_sections(self, ticker: str, sections: Dict[str, str]):
        """Chunks and embeds each extracted 10-K section into ChromaDB."""
        ticker = ticker.upper()
        documents: List[str] = []
        metadatas: List[Dict[str, str]] = []
        ids: List[str] = []

        for section_name, content in sections.items():
            if not content:
                continue
            chunks = self.chunk_text(content, chunk_size=1200, overlap=200)
            for idx, chunk in enumerate(chunks):
                doc_id = f"{ticker}_{section_name}_{idx}"
                doc_content = f"[{ticker} 10-K {section_name.upper()}]: {chunk}"
                documents.append(doc_content)
                metadatas.append({
                    "ticker": ticker,
                    "section": section_name,
                    "chunk_index": str(idx)
                })
                ids.append(doc_id)

        if documents:
            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )