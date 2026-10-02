import os
import pytest
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness

from src.core.loop import EquityResearchLoop
from src.llm.gateway import LLMGateway
from src.rag.retriever import FinancialRetriever
from src.memory.sqlite_store import LocalMemoryStore

load_dotenv()

def test_memo_faithfulness():
    # 1. Run the agent to generate a draft
    gateway = LLMGateway()
    loop = EquityResearchLoop(gateway, FinancialRetriever(), LocalMemoryStore())
    
    state = loop.run(ticker="AAPL", goal="Summarize supply chain risks.", max_iterations=1)
    
    # 2. Format the agent's scratchpad into a HuggingFace Dataset (required by Ragas)
    # Using qual_summary (the direct 10-K qualitative synthesis) tested against retrieved 10-K contexts
    eval_data = {
        "user_input": [state["user_goal"]],
        "response": [state["qual_summary"]],
        "retrieved_contexts": [[c["text"] for c in state["qual_citations"]]],
        # Backward compatibility aliases:
        "question": [state["user_goal"]],
        "answer": [state["qual_summary"]],
        "contexts": [[c["text"] for c in state["qual_citations"]]]
    }
    dataset = Dataset.from_dict(eval_data)
    
    # 3. Configure evaluator LLM
    eval_kwargs = {}
    provider = os.getenv("RESEARCH_LLM_PROVIDER", "openai").lower()
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        from ragas.llms import LangchainLLMWrapper
        eval_llm = ChatGoogleGenerativeAI(
            model=os.getenv("DEFAULT_MODEL", "gemini-3-flash-preview"),
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        eval_kwargs["llm"] = LangchainLLMWrapper(eval_llm)

    # 4. Run the LLM-as-a-judge evaluation
    result = evaluate(
        dataset,
        metrics=[faithfulness],
        **eval_kwargs
    )
    
    score = result["faithfulness"]
    if isinstance(score, list):
        score = score[0]
    print(f"\nFaithfulness Score: {score:.2f} (1.0 = 100% grounded in 10-K)")
    
    # 5. The Release Gate: Fail the test if score is below 90%
    assert score >= 0.90, f"Hallucination detected! Score {score} is below the 0.90 threshold."