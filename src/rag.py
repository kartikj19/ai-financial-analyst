from __future__ import annotations

from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import EMBEDDING_MODEL, GEMINI_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TEMPERATURE

SYSTEM_PROMPT = """You are AI Financial Analyst, a document-grounded financial research assistant.

Answer ONLY from the supplied document context. If the answer is not supported by the context, say:
'I couldn't find enough information in the uploaded documents to answer that reliably.'

Rules:
- Do not invent figures, dates, company facts, or calculations.
- Distinguish reported facts from your interpretation.
- Preserve units, currencies, periods, and percentages exactly when possible.
- For comparisons, identify the relevant reporting periods.
- For calculations, show the formula and use only values present in the context.
- Do not provide personalized investment advice or tell the user to buy/sell a security.
- Mention the source document/page when useful.

Context:
{context}
"""


def _embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True})


def _llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=TEMPERATURE)


def build_vector_store(uploaded_files: list[Any]) -> FAISS:
    documents: list[Document] = []
    for uploaded in uploaded_files:
        path = f"/tmp/{uploaded.name}"
        with open(path, "wb") as handle:
            handle.write(uploaded.getbuffer())
        pages = PyPDFLoader(path).load()
        for page in pages:
            page.metadata["source_file"] = uploaded.name
            page.metadata["page_number"] = page.metadata.get("page", 0) + 1
        documents.extend(pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No extractable text was found in the uploaded PDFs.")
    return FAISS.from_documents(chunks, _embeddings())


def _context_and_sources(docs: list[Document]) -> tuple[str, list[dict[str, str]]]:
    context_parts = []
    sources = []
    for i, doc in enumerate(docs, start=1):
        filename = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page_number", "?")
        preview = " ".join(doc.page_content.strip().split())[:280]
        context_parts.append(f"[Source {i}: {filename}, page {page}]\n{doc.page_content}")
        sources.append({"file": filename, "page": str(page), "preview": preview})
    return "\n\n".join(context_parts), sources


def answer_question(vector_store: FAISS, question: str, history: list[dict[str, str]], top_k: int = 5) -> dict[str, Any]:
    docs = vector_store.similarity_search(question, k=top_k)
    context, sources = _context_and_sources(docs)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Conversation history:\n{history}\n\nQuestion:\n{question}"),
    ])
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:]) or "No previous conversation."
    chain = prompt | _llm()
    response = chain.invoke({"context": context, "history": history_text, "question": question})
    return {"answer": response.content, "sources": sources}


def summarize_documents(vector_store: FAISS) -> str:
    # Retrieve a representative set of chunks for a compact overview.
    docs = vector_store.similarity_search("executive summary revenue profit risks guidance cash flow outlook", k=8)
    context, _ = _context_and_sources(docs)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the supplied financial documents using only the context. Cover company/business overview, revenue/profit, cash flow, risks, guidance/outlook and notable changes. State when information is unavailable. Do not provide investment advice.\n\nContext:\n{context}"),
        ("human", "Create a concise analyst-style summary."),
    ])
    return (prompt | _llm()).invoke({"context": context}).content
