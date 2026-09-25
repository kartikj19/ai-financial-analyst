import os
import streamlit as st
from dotenv import load_dotenv

from src.config import APP_TITLE, APP_SUBTITLE, DEFAULT_TOP_K
from src.rag import build_vector_store, answer_question, summarize_documents
from src.kpi import extract_financial_kpis

load_dotenv()
st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")

st.markdown("""
<style>
.main .block-container { max-width: 1200px; padding-top: 2rem; }
.hero { padding: 1.4rem 1.6rem; border-radius: 16px; background: linear-gradient(135deg,#111827,#1f2937); color: white; margin-bottom: 1rem; }
.hero h1 { margin-bottom: .25rem; }
.source-card { padding: .75rem 1rem; border-left: 4px solid #4f46e5; background: #f8fafc; border-radius: 8px; margin: .45rem 0; }
.small { color: #64748b; font-size: .88rem; }
</style>
""", unsafe_allow_html=True)

st.markdown(f'<div class="hero"><h1>📊 {APP_TITLE}</h1><div>{APP_SUBTITLE}</div></div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "files" not in st.session_state:
    st.session_state.files = []

with st.sidebar:
    st.header("Document Workspace")
    uploaded_files = st.file_uploader("Upload financial PDFs", type=["pdf"], accept_multiple_files=True)
    top_k = st.slider("Retrieved passages", 2, 8, DEFAULT_TOP_K)

    if st.button("🔎 Index Documents", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Upload at least one PDF first.")
        elif not os.getenv("GOOGLE_API_KEY"):
            st.error("GOOGLE_API_KEY is missing. Add it to .env and restart Streamlit.")
        else:
            with st.spinner("Extracting, chunking and embedding documents..."):
                try:
                    st.session_state.vector_store = build_vector_store(uploaded_files)
                    st.session_state.files = [f.name for f in uploaded_files]
                    st.session_state.messages = []
                    st.success(f"Indexed {len(uploaded_files)} document(s).")
                except Exception as exc:
                    st.exception(exc)

    if st.session_state.files:
        st.caption("Indexed documents")
        for name in st.session_state.files:
            st.write(f"• {name}")

    if st.button("🧹 Clear session", use_container_width=True):
        st.session_state.vector_store = None
        st.session_state.files = []
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**Recommended documents**")
    st.caption("Annual reports, quarterly results, investor presentations, earnings releases and financial statements.")

col1, col2 = st.columns([2, 1])
with col2:
    st.info("**How it works**\n\nPDF → text → chunks → embeddings → FAISS → relevant passages → Gemini → cited answer")
    if st.session_state.vector_store and st.button("📝 Summarize documents", use_container_width=True):
							st.divider()

st.subheader("📊 Financial KPI Dashboard")

if st.session_state.vector_store:
    if st.button(
        "🔍 Extract Financial KPIs",
        use_container_width=True,
    ):
        with st.spinner("Extracting financial KPIs..."):
            try:
                st.session_state.kpis = extract_financial_kpis(
                    st.session_state.vector_store
                )
            except Exception as exc:
                st.error(f"KPI extraction failed: {exc}")
                st.session_state.kpis = None

if "kpis" in st.session_state and st.session_state.kpis:
    kpis = st.session_state.kpis.get("kpis", [])

    if kpis:
        columns = st.columns(min(len(kpis), 4))

        for index, kpi in enumerate(kpis):
            with columns[index % 4]:
                st.metric(
                    label=kpi["name"],
                    value=kpi["value"],
                    help=(
                        f"Period: {kpi['period']} | "
                        f"Source: {kpi['source']} | "
                        f"Page: {kpi['page']}"
                    ),
                )

        st.caption(
            "KPIs are extracted from the uploaded documents. "
            "Always verify figures against the original report."
        )

        with st.expander("📑 KPI Sources"):
            for kpi in kpis:
                st.write(
                    f"**{kpi['name']}** — "
                    f"{kpi['source']}, page {kpi['page']}"
                )
    else:
        st.info(
            "No reliable financial KPIs were found in the uploaded documents."
        )
        with st.spinner("Generating summaries..."):
            try:
                st.session_state.summary = summarize_documents(st.session_state.vector_store)
            except Exception as exc:
                st.exception(exc)
    if "summary" in st.session_state:
        st.subheader("Document summary")
        st.write(st.session_state.summary)

with col1:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    for src in message["sources"]:
                        st.markdown(f'<div class="source-card"><b>{src["file"]}</b> — page {src["page"]}<br><span class="small">{src["preview"]}</span></div>', unsafe_allow_html=True)

    prompt = st.chat_input("Ask about revenue, profit, risk factors, guidance, cash flow...")
    if prompt:
        if not st.session_state.vector_store:
            st.warning("Index your PDFs first.")
            st.stop()
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and analyzing..."):
                try:
                    history = st.session_state.messages[:-1]
                    result = answer_question(st.session_state.vector_store, prompt, history, top_k)
                    st.markdown(result["answer"])
                    if result["sources"]:
                        with st.expander("Sources"):
                            for src in result["sources"]:
                                st.markdown(f'<div class="source-card"><b>{src["file"]}</b> — page {src["page"]}<br><span class="small">{src["preview"]}</span></div>', unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": result["answer"], "sources": result["sources"]})
                except Exception as exc:
                    st.error(str(exc))
