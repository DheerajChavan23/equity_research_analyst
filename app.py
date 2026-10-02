import streamlit as st
from rich.console import Console

from src.llm.gateway import LLMGateway
from src.rag.retriever import FinancialRetriever
from src.memory.sqlite_store import LocalMemoryStore
from src.core.loop import EquityResearchLoop
from src.tools.resolver import TickerResolver

# Page configuration
st.set_page_config(
    page_title="Autonomous Equity Research Analyst",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Autonomous FinTech Equity Research Agent")
st.caption("Model-Agnostic, Autonomous Multi-Agent Research System with SEC 10-K Retrieval & Guardrails")

# Sidebar - Settings & Episodic Memory Inspector
with st.sidebar:
    st.header("⚙️ Configuration & Memory")
    provider = st.selectbox("LLM Provider", ["gemini", "openai", "anthropic"], index=0)
    iterations = st.slider("Critic Max Iterations", min_value=1, max_value=3, value=2)
    
    st.markdown("---")
    st.subheader("🧠 Persistent User Rules")
    memory_store = LocalMemoryStore()
    active_memories = memory_store.get_all_memories(category="user_preference")
    if active_memories:
        for mem in active_memories:
            st.info(f"**{mem['key']}**: {mem['value']}")
    else:
        st.write("No persistent preferences stored yet.")

# Main input layout
col1, col2 = st.columns([1, 3])

with col1:
    company_or_ticker = st.text_input("Company or Ticker", value="Apple")

with col2:
    goal_input = st.text_input(
        "Research Directive", 
        value="Analyze product revenue and supply chain risks."
    )

run_button = st.button("🚀 Launch Autonomous Research", type="primary", use_container_width=True)

if run_button:
    if not company_or_ticker.strip():
        st.error("Please provide a valid company name or ticker.")
        st.stop()

    status_container = st.status("Initializing research workflow...", expanded=True)
    
    try:
        # Instantiate headless core dependencies
        gateway = LLMGateway(provider=provider)
        retriever = FinancialRetriever()
        memory = LocalMemoryStore()
        resolver = TickerResolver(gateway=gateway)

        # Step 0: Resolve Company Name to Stock Ticker
        resolved_ticker = resolver.resolve(company_or_ticker)
        if resolved_ticker != company_or_ticker.upper():
            st.info(f"Target resolved: **{company_or_ticker}** ➔ **${resolved_ticker}**")
            status_container.write(f"🎯 Target resolved: **{company_or_ticker}** ➔ **${resolved_ticker}**")
        else:
            status_container.write(f"🎯 Target: **${resolved_ticker}**")

        loop = EquityResearchLoop(gateway=gateway, retriever=retriever, memory=memory)

        status_container.write(f"🔍 Decomposing research goals for **{resolved_ticker}**...")
        # Step 1: Planner
        state = {
            "ticker": resolved_ticker,
            "target_year": 2026,
            "user_goal": goal_input,
            "research_plan": [],
            "quant_metrics": {},
            "quant_summary": "",
            "qual_citations": [],
            "qual_summary": "",
            "draft_memo": None,
            "critic_verdict": None,
            "critique_notes": None,
            "iteration_count": 0
        }
        state = loop.planner.plan(state)
        status_container.write(f"📋 **Research Plan**: {', '.join(state['research_plan'])}")

        # Step 2: Quant
        status_container.write("📊 Fetching market multiples and balance sheet figures via Yahoo Finance...")
        state = loop.quant.analyze(state)

        # Step 3: Qual ReAct Loop
        status_container.write("📑 Querying SEC 10-K filings via autonomous vector retrieval...")
        state = loop.qual.analyze(state)

        # Step 4: Synthesis & Critic Loop
        while state["iteration_count"] < iterations:
            state["iteration_count"] += 1
            status_container.write(f"✍️ Synthesizing formal memorandum (Iteration {state['iteration_count']})...")
            state["draft_memo"] = loop._synthesize_memo(state)

            status_container.write("🛡️ Critic evaluating draft against raw quantitative and filing facts...")
            state = loop.critic.review(state)

            if state["critic_verdict"] == "APPROVED":
                status_container.write("✅ **Critic Approved Report** (Fact checks passed).")
                break
            else:
                status_container.write(f"⚠️ Critic requested revisions: {state['critique_notes']}")

        # Step 5: Memory Gate & Session Persist
        status_container.write("🧠 Memory Gate evaluating session for long-term rules...")
        loop.memory_gate.evaluate_and_store(state)
        session_id = f"{resolved_ticker}_{state['target_year']}"
        loop.memory.save_scratchpad(session_id, resolved_ticker, state)

        status_container.update(label="Research Complete!", state="complete", expanded=False)

        # Display Metrics Bar
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        qm = state.get("quant_metrics", {})
        m_col1.metric("Market Cap", f"${qm.get('market_cap', 0):,}" if isinstance(qm.get('market_cap'), (int, float)) else str(qm.get('market_cap', 'N/A')))
        m_col2.metric("Trailing P/E", f"{qm.get('pe_ratio_trailing', 'N/A')}x")
        m_col3.metric("Gross Margin", f"{round(qm.get('gross_margins', 0) * 100, 2)}%" if isinstance(qm.get('gross_margins'), float) else str(qm.get('gross_margins', 'N/A')))
        m_col4.metric("Operating Margin", f"{round(qm.get('operating_margins', 0) * 100, 2)}%" if isinstance(qm.get('operating_margins'), float) else str(qm.get('operating_margins', 'N/A')))

        st.markdown("---")
        
        # Render the final Markdown report
        st.markdown(state["draft_memo"])

        # Display raw citations retrieved by the Qual agent
        if state.get("qual_citations"):
            with st.expander("📚 View Extracted 10-K Citations"):
                for idx, cite in enumerate(state["qual_citations"]):
                    st.markdown(f"**Citation {idx + 1}** (`{cite.get('section', 'Unknown')}`):")
                    st.caption(cite.get("text", ""))

    except Exception as e:
        status_container.update(label="Execution Failed", state="error")
        st.error(f"Error during research loop: {str(e)}")