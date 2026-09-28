import os
import re

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.config import APP_TITLE, APP_SUBTITLE, DEFAULT_TOP_K
from src.rag import (
    build_vector_store,
    answer_question,
    summarize_documents,
    compare_documents,
)
from src.kpi import extract_financial_kpis


load_dotenv()

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout="wide",
)


st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
    }

    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #111827, #1f2937);
        color: white;
        margin-bottom: 1rem;
    }

    .hero h1 {
        margin-bottom: .25rem;
    }

    .source-card {
        padding: .75rem 1rem;
        border-left: 4px solid #4f46e5;
        background: #f8fafc;
        border-radius: 8px;
        margin: .45rem 0;
    }

    .small {
        color: #64748b;
        font-size: .88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def extract_metric_values(text, metric_keywords):
    """
    Extract FY 2025 and FY 2026 financial values from Gemini's
    comparison response.

    Returns:
        tuple[str | None, str | None]: FY 2025 and FY 2026 values.
    """

    # Gemini can sometimes return structured content as a list.
    if isinstance(text, list):
        text = "\n".join(
            item.get("text", str(item)) if isinstance(item, dict) else str(item)
            for item in text
        )

    text = str(text)

    keyword_pattern = "|".join(metric_keywords)

    # Find the metric heading/section and keep the search reasonably local.
    pattern = (
        r"(?is)"
        r"(?:###\s*)?"
        r"(?:\d+\.\s*)?"
        r"(?:"
        + keyword_pattern
        + r")"
        r".*?"
        r"(?=(?:\n\s*(?:###|\d+\.)\s)|\Z)"
    )

    match = re.search(pattern, text)

    if not match:
        return None, None

    section = match.group(0)

    value_pattern = (
        r"(?:₹|Rs\.?|INR)?\s*"
        r"([\d,]+(?:\.\d+)?)"
        r"\s*(?:crore|Cr)"
    )

    fy2025_match = re.search(
        r"FY\s*2025.*?" + value_pattern,
        section,
        re.IGNORECASE | re.DOTALL,
    )

    fy2026_match = re.search(
        r"FY\s*2026.*?" + value_pattern,
        section,
        re.IGNORECASE | re.DOTALL,
    )

    value_2025 = fy2025_match.group(1) if fy2025_match else None
    value_2026 = fy2026_match.group(1) if fy2026_match else None

    return value_2025, value_2026


def build_structured_comparison(answer):
    """
    Convert the AI comparison response into a structured DataFrame.

    Only figures explicitly present in the AI comparison response are used.
    Missing values remain unavailable rather than being invented.
    """

    if isinstance(answer, list):
        answer = "\n".join(
            item.get("text", str(item)) if isinstance(item, dict) else str(item)
            for item in answer
        )

    answer = str(answer)

    metrics = [
        ("Revenue", ["Revenue"]),
        ("Total Income", ["Total Income"]),
        ("Operating Profit", ["Operating Profit"]),
        (
            "Profit After Tax",
            [
                "Profit After Tax",
                "Profit for the Year",
                r"Profit After Tax \(PAT\)",
            ],
        ),
    ]

    comparison_rows = []

    for metric_name, keywords in metrics:
        fy2025, fy2026 = extract_metric_values(answer, keywords)

        if not fy2025 and not fy2026:
            continue

        value_2025 = (
            float(fy2025.replace(",", ""))
            if fy2025
            else None
        )

        value_2026 = (
            float(fy2026.replace(",", ""))
            if fy2026
            else None
        )

        percentage_change = None

        if (
            value_2025 is not None
            and value_2026 is not None
            and value_2025 != 0
        ):
            percentage_change = (
                (value_2026 - value_2025) / value_2025
            ) * 100

        comparison_rows.append(
            {
                "Metric": metric_name,
                "FY 2025 (₹ Cr)": value_2025,
                "FY 2026 (₹ Cr)": value_2026,
                "Change (%)": percentage_change,
            }
        )

    if not comparison_rows:
        return None

    return pd.DataFrame(comparison_rows)


# ---------------------------------------------------------
# Hero
# ---------------------------------------------------------

st.markdown(
    f"""
    <div class="hero">
        <h1>📊 {APP_TITLE}</h1>
        <div>{APP_SUBTITLE}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "files" not in st.session_state:
    st.session_state.files = []

if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = None


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.header("Document Workspace")

    uploaded_files = st.file_uploader(
        "Upload financial PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    top_k = st.slider(
        "Retrieved passages",
        2,
        8,
        DEFAULT_TOP_K,
    )

    if st.button(
        "🔎 Index Documents",
        width="stretch",
        type="primary",
    ):
        if not uploaded_files:
            st.warning("Upload at least one PDF first.")

        elif not os.getenv("GOOGLE_API_KEY"):
            st.error(
                "GOOGLE_API_KEY is missing. "
                "Add it to .env and restart Streamlit."
            )

        else:
            with st.spinner(
                "Extracting, chunking and embedding documents..."
            ):
                try:
                    st.session_state.vector_store = build_vector_store(
                        uploaded_files
                    )

                    st.session_state.files = [
                        f.name for f in uploaded_files
                    ]

                    st.session_state.messages = []
                    st.session_state.kpis = None
                    st.session_state.summary = None
                    st.session_state.comparison_result = None

                    st.success(
                        f"Indexed {len(uploaded_files)} document(s)."
                    )

                except Exception as exc:
                    st.exception(exc)

    if st.session_state.files:
        st.caption("Indexed documents")

        for name in st.session_state.files:
            st.write(f"• {name}")

    if st.button(
        "🧹 Clear session",
        width="stretch",
    ):
        st.session_state.vector_store = None
        st.session_state.files = []
        st.session_state.messages = []
        st.session_state.kpis = None
        st.session_state.summary = None
        st.session_state.comparison_result = None
        st.rerun()

    st.divider()

    st.markdown("**Recommended documents**")

    st.caption(
        "Annual reports, quarterly results, investor presentations, "
        "earnings releases and financial statements."
    )


# ---------------------------------------------------------
# Main layout
# ---------------------------------------------------------

col1, col2 = st.columns([2, 1])

with col2:
    st.info(
        "**How it works**\n\n"
        "PDF → text → chunks → embeddings → FAISS → "
        "relevant passages → Gemini → cited answer"
    )

    if st.session_state.vector_store:
        if st.button(
            "📝 Summarize documents",
            width="stretch",
        ):
            with st.spinner("Generating document summary..."):
                try:
                    st.session_state.summary = summarize_documents(
                        st.session_state.vector_store
                    )
                except Exception as exc:
                    st.error(f"Summary generation failed: {exc}")


# ---------------------------------------------------------
# Financial KPI Dashboard
# ---------------------------------------------------------

st.subheader("📊 Financial KPI Dashboard")

if st.session_state.vector_store:

    if st.button(
        "🔍 Extract Financial KPIs",
        width="stretch",
    ):
        with st.spinner("Extracting financial KPIs..."):
            try:
                st.session_state.kpis = extract_financial_kpis(
                    st.session_state.vector_store
                )
            except Exception as exc:
                st.error(f"KPI extraction failed: {exc}")
                st.session_state.kpis = None


if (
    "kpis" in st.session_state
    and st.session_state.kpis
):
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


if (
    "summary" in st.session_state
    and st.session_state.summary
):
    st.subheader("Document Summary")
    st.write(st.session_state.summary)


# ---------------------------------------------------------
# RAG Chat
# ---------------------------------------------------------

with col1:

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("Sources"):

                    for src in message["sources"]:
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <b>{src["file"]}</b> — page {src["page"]}<br>
                                <span class="small">
                                    {src["preview"]}
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    prompt = st.chat_input(
        "Ask about revenue, profit, risk factors, guidance, cash flow..."
    )

    if prompt:

        if not st.session_state.vector_store:
            st.warning("Index your PDFs first.")
            st.stop()

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching documents and analyzing..."
            ):

                try:
                    history = st.session_state.messages[:-1]

                    result = answer_question(
                        st.session_state.vector_store,
                        prompt,
                        history,
                        top_k,
                    )

                    st.markdown(result["answer"])

                    if result["sources"]:

                        with st.expander("Sources"):

                            for src in result["sources"]:
                                st.markdown(
                                    f"""
                                    <div class="source-card">
                                        <b>{src["file"]}</b> —
                                        page {src["page"]}<br>
                                        <span class="small">
                                            {src["preview"]}
                                        </span>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"],
                        }
                    )

                except Exception as exc:
                    st.error(str(exc))


# ---------------------------------------------------------
# Multi-PDF Financial Comparison
# ---------------------------------------------------------

st.divider()

st.subheader("📊 Multi-PDF Financial Comparison")


if (
    st.session_state.vector_store
    and len(st.session_state.files) >= 2
):

    comparison_question = st.text_input(
        "What would you like to compare?",
        placeholder=(
            "Example: Compare revenue and profit "
            "between the uploaded reports."
        ),
    )

    if st.button(
        "🔍 Compare Documents",
        width="stretch",
    ):

        if not comparison_question.strip():

            st.warning(
                "Please enter a comparison question."
            )

        else:

            with st.spinner(
                "Comparing the uploaded documents..."
            ):

                try:

                    comparison_result = compare_documents(
                        st.session_state.vector_store,
                        comparison_question,
                        top_k=10,
                    )

                    st.session_state.comparison_result = (
                        comparison_result
                    )

                except Exception as exc:

                    st.error(
                        f"Comparison failed: {exc}"
                    )

                    st.session_state.comparison_result = None

else:

    st.info(
        "Upload and index at least 2 PDFs to compare documents."
    )


# ---------------------------------------------------------
# Comparison Result
# ---------------------------------------------------------

if (
    st.session_state.comparison_result
):

    result = st.session_state.comparison_result

    st.markdown("### 📊 Comparison Result")

    # Original Gemini explanation
    st.markdown(result["answer"])

    # -----------------------------------------------------
    # Structured Financial Comparison
    # -----------------------------------------------------
# ---------------------------------------------
# Structured Financial Comparison
# ---------------------------------------------

import re

if "comparison_result" not in st.session_state or not st.session_state.comparison_result:
    st.stop()

result = st.session_state.comparison_result
answer_text = result["answer"]


# Gemini can occasionally return a list instead of a string
if isinstance(answer_text, list):
    answer_text = "\n".join(
        item.get("text", str(item)) if isinstance(item, dict) else str(item)
        for item in answer_text
    )

answer_text = str(answer_text)


def extract_metric_values(text, metric_names):
    """
    Extract FY 2025 and FY 2026 financial values from
    Gemini's comparison response.

    The parser intentionally searches a limited area around
    the requested metric instead of depending on Markdown
    heading formatting.
    """

    text_lower = text.lower()

    metric_position = -1

    for metric in metric_names:
        position = text_lower.find(metric.lower())

        if position != -1:
            metric_position = position
            break

    if metric_position == -1:
        return None, None

    # Look around the metric section.
    # This is deliberately generous because Gemini's formatting
    # can vary between responses.
    section_start = metric_position
    section_end = min(
        len(text),
        metric_position + 5000
    )

    section = text[section_start:section_end]

    # Stop at the next numbered section when possible.
    next_section = re.search(
        r"\n\s*(?:#{1,4}\s*)?\d+\.\s+",
        section
    )

    if next_section and next_section.start() > 100:
        section = section[:next_section.start()]

    # Financial value pattern:
    #
    # ₹255,324 crore
    # ₹255,324 Cr
    # Rs 255,324 crore
    # 255,324 crore
    # 49,454 crore
    #
    value_pattern = (
        r"(?:₹|Rs\.?|INR)?\s*"
        r"([\d,]+(?:\.\d+)?)"
        r"\s*(?:crore|cr)\b"
    )

    def find_year_value(year):
        pattern = (
            rf"FY\s*{year}"
            rf".{{0,1000}}?"
            rf"{value_pattern}"
        )

        match = re.search(
            pattern,
            section,
            flags=re.IGNORECASE | re.DOTALL
        )

        if match:
            return match.group(1)

        return None

    return (
        find_year_value(2025),
        find_year_value(2026)
    )


metrics = [
    (
        "Revenue",
        ["Revenue"]
    ),
    (
        "Total Income",
        ["Total Income"]
    ),
    (
        "Operating Profit",
        ["Operating Profit"]
    ),
    (
        "Profit After Tax",
        [
            "Profit After Tax",
            "Profit for the Year",
            "Profit After Tax (PAT)"
        ]
    ),
]


comparison_rows = []


for metric_name, metric_keywords in metrics:

    fy2025, fy2026 = extract_metric_values(
        answer_text,
        metric_keywords
    )

    if fy2025 is None and fy2026 is None:
        continue

    value_2025 = (
        float(fy2025.replace(",", ""))
        if fy2025
        else None
    )

    value_2026 = (
        float(fy2026.replace(",", ""))
        if fy2026
        else None
    )

    percentage_change = None

    if (
        value_2025 is not None
        and value_2026 is not None
        and value_2025 != 0
    ):
        percentage_change = (
            (value_2026 - value_2025)
            / value_2025
        ) * 100

    comparison_rows.append(
        {
            "Metric": metric_name,
            "FY 2025 (₹ Cr)": value_2025,
            "FY 2026 (₹ Cr)": value_2026,
            "Change (%)": percentage_change,
        }
    )


# ---------------------------------------------
# Financial Comparison Table
# ---------------------------------------------

if comparison_rows:

    comparison_df = pd.DataFrame(comparison_rows)

    st.markdown("### 📊 Financial Comparison Table")

    display_df = comparison_df.copy()

    for column in [
        "FY 2025 (₹ Cr)",
        "FY 2026 (₹ Cr)"
    ]:

        display_df[column] = display_df[column].apply(
            lambda value:
                f"₹{value:,.0f} Cr"
                if pd.notna(value)
                else "Not available"
        )

    display_df["Change (%)"] = display_df["Change (%)"].apply(
        lambda value:
            f"{value:+.2f}%"
            if pd.notna(value)
            else "Not available"
    )

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
    )


    # ---------------------------------------------
    # Visual Comparison
    # ---------------------------------------------

    chart_df = comparison_df[
        [
            "Metric",
            "FY 2025 (₹ Cr)",
            "FY 2026 (₹ Cr)"
        ]
    ].copy()

    # Keep only rows where at least one value exists
    chart_df = chart_df.dropna(
        subset=[
            "FY 2025 (₹ Cr)",
            "FY 2026 (₹ Cr)"
        ],
        how="all"
    )

    if not chart_df.empty:

        st.markdown("### 📈 Visual Comparison")

        chart_data = chart_df.set_index("Metric")

        st.bar_chart(
            chart_data,
            width="stretch"
        )

        st.caption(
            "Values are extracted from the uploaded financial reports. "
            "Always verify figures against the original documents."
        )

else:

    st.info(
        "No structured financial figures could be extracted "
        "from the comparison response. The detailed AI comparison "
        "above is still available."
    )
      # -----------------------------------------------------
    # Comparison Sources
    # -----------------------------------------------------

    if result.get("sources"):

        with st.expander(
            "📚 Comparison Sources"
        ):

            for src in result["sources"]:

                st.write(
                    f"**{src['file']}** — Page {src['page']}"
                )
